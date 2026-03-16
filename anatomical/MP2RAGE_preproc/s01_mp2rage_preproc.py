#!/usr/bin/env python
"""
run_mp2rage_preproc.py
======================
Run the MP2RAGE preprocessing pipeline sequentially (no nipype).

    Step 0   - SPM bias-field correction of the INV2 image
    Step 1   - MPRAGEise the UNI image using the bias-corrected INV2
    Step 1d  - Segmentation on MPRAGEised UNI (with skull):
                 · CAT12  →  <prefix>_UNI-mpragised_cat12seg/
                 · SPM    →  <prefix>_UNI-mpragised_spmseg/
    Step 1e  - Warp atlas sagittal sinus mask → T1w space (FLIRT affine;
               requires --atlas-sag-sinus)
    Step 1b  - Nighres skull stripping → brain mask
    Step 1c  - Apply brain mask:
                 · raw UNI              → UNI-brain            (→ MGDM)
                 · MPRAGEised UNI       → UNI-mpragised-brain   (→ FreeSurfer)
                 · bias-corrected INV2  → INV2-spmbc-brain      (→ QC)
                 · T1map (optional)     → T1map-brain           (→ MGDM)
    Step 2   - Nighres MGDM segmentation
    Step 3   - Nighres dura estimation
    Step 4a  - Combine nighres + CAT12 brain masks with dura/MGDM-guided
               erosion → brain-mask-combined.nii.gz
    Step 4b  - Refine SSS mask: atlas prior × INV2 dark signal × dura proba
               → SSS-mask-refined.nii.gz
               [QC checkpoint 1: review SSS mask before carving brain mask]
    Step 4c  - Subtract dilated SSS from combined mask
               → brain-mask-final.nii.gz
               [QC checkpoint 2: review final mask before FreeSurfer]
    Step 4d  - Apply final brain mask to MPRAGEised UNI
               → UNI-mpragised-brain-final.nii.gz  (FreeSurfer input)

Overwrite behaviour
-------------------
Existence is checked against the final BIDS-named files in *outdir*.
If an output already exists and overwrite is False, the step is skipped
and the file is copied back into *workdir* for downstream use.

Valid step names for --overwrite:
    spmbc        Step 0   - SPM bias-field correction
    mpragise     Step 1   - MPRAGEise
    cat12seg     Step 1d  - CAT12 segmentation
    spmseg       Step 1d  - SPM segmentation
    sagsinus     Step 1e  - Atlas sagittal sinus warp
    skullstrip   Step 1b  - Nighres skull stripping
    applymask    Step 1c  - Apply brain mask
    mgdm         Step 2   - Nighres MGDM segmentation
    dura         Step 3   - Nighres dura estimation
    combinemasks Step 4a  - Combine nighres + CAT12 masks
    refineSss    Step 4b  - Refine SSS mask
    finalMask    Step 4c+4d - Subtract SSS + apply final mask

Usage example
-------------
python run_mp2rage_preproc.py \\
    --uni         /data/sub-01/ses-01/anat/sub-01_ses-01_UNI.nii.gz \\
    --inv2        /data/sub-01/ses-01/anat/sub-01_ses-01_INV2.nii.gz \\
    --t1map       /data/sub-01/ses-01/anat/sub-01_ses-01_T1map.nii.gz \\
    --outdir      /out/sub-01/ses-01/anat \\
    --subject     sub-01 \\
    --session     ses-01 \\
    --workdir     /tmp/mp2rage_work \\
    --mp2rage-script-dir /opt/mp2rage_scripts \\
    --spm-standalone     /opt/spm12/run_spm12.sh \\
    --mcr-path           /opt/mcr/v99 \\
    --atlas-sag-sinus    /opt/fsl/data/standard/MNI152_T1_1mm_Dil3_sagsinus_mask.nii.gz \\
    --skip-qc
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from preproc_utils import (
    check_skip,
    get_stem,
    spm_bias_correct,
    mprage_ise,
    cat12_seg,
    spm_seg,
    warp_atlas_sag_sinus,
    nighres_skull_strip,
    apply_mask,
    nighres_mgdm,
    nighres_dura_estimation,
    combine_brain_masks,
    refine_sss_mask,
    make_brain_mask_nosss,
    launch_freeview,
)

STEP_KEYS = [
    'spmbc',
    'mpragise',
    'cat12seg',
    'spmseg',
    'sagsinus',
    'skullstrip',
    'applymask',
    'mgdm',
    'dura',
    'combinemasks',
    'refineSss',
    'finalMask',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_output_name(outdir, subject, session, suffix, extension='.nii.gz'):
    """
    Build a BIDS-style output path.

    >>> build_output_name('/out', 'sub-01', 'ses-01', 'T1w-mpragised')
    '/out/sub-01_ses-01_T1w-mpragised.nii.gz'
    >>> build_output_name('/out', 'sub-01', None, 'T1w-mpragised')
    '/out/sub-01_T1w-mpragised.nii.gz'
    """
    tokens = [t for t in [subject, session, suffix] if t]
    return os.path.join(outdir, '_'.join(tokens) + extension)


def _qc_checkpoint(description, freeview_args, skip_qc):
    """
    Pause for manual review in freeview unless --skip-qc is set.
    Prompts for Enter after viewing so any edits are read back before
    the pipeline continues.
    """
    print('\n' + '=' * 62)
    print('  [QC] {}'.format(description))
    print('=' * 62)

    if skip_qc:
        print('  --skip-qc set: skipping manual review.')
        return

    launch_freeview(*freeview_args)

    print('\n  Review the images in freeview.')
    print('  If you edit the mask:')
    print('    File → Save Volume As ... → overwrite the same file')
    print('  Press Enter here when done (or Ctrl-C to abort pipeline).')
    try:
        input('  [waiting] Press Enter to continue > ')
    except KeyboardInterrupt:
        print('\nPipeline aborted by user.')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    uni,
    inv2,
    outdir,
    subject,
    mp2rage_script_dir,
    session=None,
    workdir='/tmp/mp2rage_work',
    t1map=None,
    spm_script='s01_spmbc',
    spm_standalone=None,
    mcr_path=None,
    cat12_script='preproc_cat12seg',
    spm_seg_script='preproc_spmseg',
    atlas_sag_sinus=None,
    fsl_dir=None,
    nighres_docker='nighres/nighres:latest',
    mgdm_contrast='Mp2rage7T',
    mgdm_atlas=None,
    dura_background_distance=5.0,
    dura_threshold=0.7,
    inv2_sss_percentile=15.0,
    sss_dilation_mm=3.5,
    skip_qc=False,
    overwrite=None,
):
    """
    Run the full MP2RAGE preprocessing pipeline.

    Outputs are written to BIDS-named files in *outdir*. Intermediate files
    live in *workdir*. Skipped steps restore their outputs from *outdir* into
    *workdir* so downstream steps always read from a consistent location.

    Parameters
    ----------
    dura_threshold       : Dura probability threshold for mask erosion in
                           Step 4a (default 0.7 — conservative).
    inv2_sss_percentile  : INV2 percentile defining 'dark' signal in the SSS
                           ROI for Step 4b (default 15.0).
    sss_dilation_mm      : SSS mask dilation radius before subtraction from
                           brain mask in Step 4c (default 3.5 mm).
    skip_qc              : Skip manual freeview checkpoints (batch use).
    overwrite            : Dict mapping step keys → bool. Missing keys default
                           to False. Valid keys: see STEP_KEYS.

    Returns
    -------
    dict mapping output labels to their final paths in outdir.
    """
    ow = {k: False for k in STEP_KEYS}
    if overwrite:
        unknown = set(overwrite) - set(STEP_KEYS)
        if unknown:
            raise ValueError(
                'Unknown overwrite key(s): {}. Valid keys: {}'.format(
                    sorted(unknown), STEP_KEYS)
            )
        ow.update(overwrite)

    os.makedirs(outdir,  exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    workdir = str(Path(workdir).resolve())
    outdir  = str(Path(outdir).resolve())

    prefix = '_'.join(t for t in [subject, session] if t)

    def _final(suffix, ext='.nii.gz'):
        return build_output_name(outdir, subject, session, suffix,
                                 extension=ext)

    def _work(final_path):
        return os.path.join(workdir, os.path.basename(final_path))

    # ------------------------------------------------------------------
    # Step 0 - SPM bias-field correction of INV2
    # ------------------------------------------------------------------
    print('\n[Step 0] SPM bias-field correction of INV2...')

    inv2_bc_final = _final('INV2-spmbc')
    inv2_bc_work  = _work(inv2_bc_final)

    if not check_skip(
        {'inv2_bc': inv2_bc_final},
        ow['spmbc'],
        'Step 0: SPM bias-field correction',
        workdir_paths={'inv2_bc': inv2_bc_work},
    ):
        inv2_bc_work = spm_bias_correct(
            input_image=inv2,
            out_dir=workdir,
            mp2rage_script_dir=mp2rage_script_dir,
            spm_script=spm_script,
            spm_standalone=spm_standalone,
            mcr_path=mcr_path,
        )
        shutil.copy(inv2_bc_work, inv2_bc_final)

    print('  -> {}'.format(inv2_bc_work))

    # ------------------------------------------------------------------
    # Step 1 - MPRAGEise UNI with bias-corrected INV2
    # ------------------------------------------------------------------
    print('\n[Step 1] MPRAGEising UNI...')

    uni_mpragised_final = _final('UNI-mpragised')
    uni_mpragised_work  = _work(uni_mpragised_final)

    if not check_skip(
        {'uni_mpragised': uni_mpragised_final},
        ow['mpragise'],
        'Step 1: MPRAGEise',
        workdir_paths={'uni_mpragised': uni_mpragised_work},
    ):
        uni_mpragised_work = mprage_ise(
            uni_file=uni,
            inv2_file=inv2_bc_work,
            out_dir=workdir,
        )
        shutil.copy(uni_mpragised_work, uni_mpragised_final)

    print('  -> {}'.format(uni_mpragised_work))

    # ------------------------------------------------------------------
    # Step 1d - CAT12 segmentation (MPRAGEised UNI, with skull)
    # ------------------------------------------------------------------
    print('\n[Step 1d] CAT12 segmentation...')

    uni_mpragised_stem    = get_stem(Path(uni_mpragised_final))
    cat12_out_final       = os.path.join(outdir, '{}_UNI-mpragised_cat12seg'.format(prefix))
    cat12_sentinel_final  = os.path.join(cat12_out_final, '{}_cat12seg_batch.mat'.format(uni_mpragised_stem))
    cat12_brainmask_final = os.path.join(cat12_out_final, '{}_brainmask.nii'.format(uni_mpragised_stem))

    if not check_skip(
        {'cat12seg_batch': cat12_sentinel_final},
        ow['cat12seg'],
        'Step 1d: CAT12 segmentation',
    ):
        cat12_work_dir = cat12_seg(
            input_image=uni_mpragised_work,
            out_dir=workdir,
            mp2rage_script_dir=mp2rage_script_dir,
            spm_script=cat12_script,
            spm_standalone=spm_standalone,
            mcr_path=mcr_path,
        )
        if os.path.exists(cat12_out_final):
            shutil.rmtree(cat12_out_final)
        shutil.copytree(cat12_work_dir, cat12_out_final)

    print('  -> {}'.format(cat12_out_final))

    # ------------------------------------------------------------------
    # Step 1d - SPM segmentation (MPRAGEised UNI, with skull)
    # ------------------------------------------------------------------
    print('\n[Step 1d] SPM segmentation...')

    spm_out_final       = os.path.join(outdir, '{}_UNI-mpragised_spmseg'.format(prefix))
    spm_sentinel_final  = os.path.join(spm_out_final, '{}_spmseg_batch.mat'.format(uni_mpragised_stem))
    spm_brainmask_final = os.path.join(spm_out_final, '{}_brainmask.nii'.format(uni_mpragised_stem))

    if not check_skip(
        {'spmseg_batch': spm_sentinel_final},
        ow['spmseg'],
        'Step 1d: SPM segmentation',
    ):
        spm_work_dir = spm_seg(
            input_image=uni_mpragised_work,
            out_dir=workdir,
            mp2rage_script_dir=mp2rage_script_dir,
            spm_script=spm_seg_script,
            spm_standalone=spm_standalone,
            mcr_path=mcr_path,
        )
        if os.path.exists(spm_out_final):
            shutil.rmtree(spm_out_final)
        shutil.copytree(spm_work_dir, spm_out_final)

    print('  -> {}'.format(spm_out_final))

    # ------------------------------------------------------------------
    # Step 1e - Warp atlas sagittal sinus mask → T1w space
    # ------------------------------------------------------------------
    print('\n[Step 1e] Warping atlas sagittal sinus mask to T1w space...')

    sss_atlas_final = _final('SSS-atlas-in-T1')
    sss_atlas_work  = _work(sss_atlas_final)

    if atlas_sag_sinus:
        if not check_skip(
            {'sss_atlas': sss_atlas_final},
            ow['sagsinus'],
            'Step 1e: atlas sagittal sinus warp',
            workdir_paths={'sss_atlas': sss_atlas_work},
        ):
            sss_atlas_work = warp_atlas_sag_sinus(
                t1w_image=uni_mpragised_work,
                atlas_mask=atlas_sag_sinus,
                out_dir=workdir,
                fsl_dir=fsl_dir,
            )
            shutil.copy(sss_atlas_work, sss_atlas_final)
        print('  -> {}'.format(sss_atlas_work))
    else:
        print('  [skip] --atlas-sag-sinus not provided.')
        sss_atlas_final = None
        sss_atlas_work  = None

    # ------------------------------------------------------------------
    # Step 1b - Nighres skull stripping → brain mask
    # ------------------------------------------------------------------
    print('\n[Step 1b] Nighres skull stripping...')

    brain_mask_final = _final('brain-mask')
    brain_mask_work  = _work(brain_mask_final)

    if not check_skip(
        {'brain_mask': brain_mask_final},
        ow['skullstrip'],
        'Step 1b: Nighres skull stripping',
        workdir_paths={'brain_mask': brain_mask_work},
    ):
        brain_mask_work = nighres_skull_strip(
            inv2_image=inv2_bc_work,
            uni_image=uni,
            out_dir=workdir,
            t1map_image=t1map,
            docker_image=nighres_docker,
        )
        shutil.copy(brain_mask_work, brain_mask_final)

    print('  -> {}'.format(brain_mask_work))

    # ------------------------------------------------------------------
    # Step 1c - Apply brain mask to every image
    # ------------------------------------------------------------------
    print('\n[Step 1c] Applying brain mask...')

    uni_brain_final           = _final('UNI-brain')
    uni_mpragised_brain_final = _final('UNI-mpragised-brain')
    inv2_brain_final          = _final('INV2-spmbc-brain')
    t1map_brain_final         = _final('T1map-brain') if t1map else None

    outdir_applymask  = {
        'uni_brain':           uni_brain_final,
        'uni_mpragised_brain': uni_mpragised_brain_final,
        'inv2_brain':          inv2_brain_final,
    }
    workdir_applymask = {k: _work(v) for k, v in outdir_applymask.items()}

    if t1map:
        outdir_applymask['t1map_brain']  = t1map_brain_final
        workdir_applymask['t1map_brain'] = _work(t1map_brain_final)

    if not check_skip(
        outdir_applymask,
        ow['applymask'],
        'Step 1c: apply brain mask',
        workdir_paths=workdir_applymask,
    ):
        uni_brain_work = apply_mask(
            input_image=uni,
            mask_image=brain_mask_work,
            out_dir=workdir,
            out_suffix='_brain',
        )
        uni_mpragised_brain_work = apply_mask(
            input_image=uni_mpragised_work,
            mask_image=brain_mask_work,
            out_dir=workdir,
            out_suffix='_brain',
        )
        inv2_brain_work = apply_mask(
            input_image=inv2_bc_work,
            mask_image=brain_mask_work,
            out_dir=workdir,
            out_suffix='_brain',
        )
        shutil.copy(uni_brain_work,           uni_brain_final)
        shutil.copy(uni_mpragised_brain_work, uni_mpragised_brain_final)
        shutil.copy(inv2_brain_work,          inv2_brain_final)

        t1map_brain_work = None
        if t1map:
            t1map_brain_work = apply_mask(
                input_image=t1map,
                mask_image=brain_mask_work,
                out_dir=workdir,
                out_suffix='_brain',
            )
            shutil.copy(t1map_brain_work, t1map_brain_final)
    else:
        uni_brain_work           = workdir_applymask['uni_brain']
        uni_mpragised_brain_work = workdir_applymask['uni_mpragised_brain']
        inv2_brain_work          = workdir_applymask['inv2_brain']
        t1map_brain_work         = workdir_applymask.get('t1map_brain')

    print('  UNI brain           -> {}'.format(uni_brain_work))
    print('  UNI mpragised brain -> {}'.format(uni_mpragised_brain_work))
    print('  INV2 brain          -> {}'.format(inv2_brain_work))
    if t1map_brain_work:
        print('  T1map brain         -> {}'.format(t1map_brain_work))

    # ------------------------------------------------------------------
    # Step 2 - Nighres MGDM segmentation
    # ------------------------------------------------------------------
    print('\n[Step 2] Nighres MGDM segmentation...')

    mgdm_seg_final  = _final('mgdm-seg')
    mgdm_dist_final = _final('mgdm-dist')
    mgdm_lbls_final = _final('mgdm-lbls')
    mgdm_mems_final = _final('mgdm-mems')

    outdir_mgdm  = {
        'segmentation': mgdm_seg_final,
        'distance':     mgdm_dist_final,
        'labels':       mgdm_lbls_final,
        'memberships':  mgdm_mems_final,
    }
    workdir_mgdm = {k: _work(v) for k, v in outdir_mgdm.items()}

    if not check_skip(
        outdir_mgdm,
        ow['mgdm'],
        'Step 2: Nighres MGDM segmentation',
        workdir_paths=workdir_mgdm,
    ):
        mgdm_outputs = nighres_mgdm(
            input_image=uni_brain_work,
            out_dir=workdir,
            docker_image=nighres_docker,
            contrast_type=mgdm_contrast,
            t1map_image=t1map_brain_work,
            atlas=mgdm_atlas,
        )
        for key, final in outdir_mgdm.items():
            shutil.copy(mgdm_outputs[key], final)
        mgdm_work = mgdm_outputs
    else:
        mgdm_work = workdir_mgdm

    print('  segmentation -> {}'.format(mgdm_work['segmentation']))

    # ------------------------------------------------------------------
    # Step 3 - Nighres dura estimation
    # ------------------------------------------------------------------
    print('\n[Step 3] Nighres dura estimation...')

    dura_final = _final('dura-proba')
    dura_work  = _work(dura_final)

    if not check_skip(
        {'dura_proba': dura_final},
        ow['dura'],
        'Step 3: Nighres dura estimation',
        workdir_paths={'dura_proba': dura_work},
    ):
        dura_work = nighres_dura_estimation(
            inv2_image=inv2_bc_work,
            brain_mask=brain_mask_work,
            out_dir=workdir,
            docker_image=nighres_docker,
            background_distance=dura_background_distance,
        )
        shutil.copy(dura_work, dura_final)

    print('  dura probability -> {}'.format(dura_work))

    # ------------------------------------------------------------------
    # Step 4a - Combine nighres + CAT12 brain masks
    # ------------------------------------------------------------------
    print('\n[Step 4a] Combining brain masks...')

    combined_mask_final = _final('brain-mask-combined')
    combined_mask_work  = _work(combined_mask_final)

    if not check_skip(
        {'combined_mask': combined_mask_final},
        ow['combinemasks'],
        'Step 4a: combine brain masks',
        workdir_paths={'combined_mask': combined_mask_work},
    ):
        if not Path(cat12_brainmask_final).exists():
            raise FileNotFoundError(
                'CAT12 brain mask not found: {}\n'
                'Ensure Step 1d completed successfully.'.format(
                    cat12_brainmask_final)
            )
        combined_mask_work = combine_brain_masks(
            nighres_mask=brain_mask_work,
            cat12_mask=cat12_brainmask_final,
            dura_proba=dura_work,
            mgdm_memberships=mgdm_work['memberships'],
            out_dir=workdir,
            dura_threshold=dura_threshold,
        )
        shutil.copy(combined_mask_work, combined_mask_final)

    print('  -> {}'.format(combined_mask_work))

    # ------------------------------------------------------------------
    # Step 4b - Refine SSS mask (atlas × INV2 dark signal × dura proba)
    # ------------------------------------------------------------------
    print('\n[Step 4b] Refining SSS mask...')

    sss_refined_final = _final('SSS-mask-refined')
    sss_refined_work  = _work(sss_refined_final)

    if sss_atlas_final and Path(sss_atlas_final).exists():
        if not check_skip(
            {'sss_refined': sss_refined_final},
            ow['refineSss'],
            'Step 4b: refine SSS mask',
            workdir_paths={'sss_refined': sss_refined_work},
        ):
            sss_refined_work = refine_sss_mask(
                atlas_sss_in_t1=sss_atlas_work,
                inv2_image=inv2_bc_work,
                dura_proba=dura_work,
                brain_mask=combined_mask_work,
                out_dir=workdir,
                inv2_percentile=inv2_sss_percentile,
            )
            shutil.copy(sss_refined_work, sss_refined_final)

        print('  -> {}'.format(sss_refined_work))

        _qc_checkpoint(
            description=(
                'Review the refined SSS mask before it carves the brain mask.\n\n'
                '  What to check:\n'
                '    - Does the mask follow the superior sagittal sinus?\n'
                '    - Under-filled → sinus remains in brain mask\n'
                '    - Over-filled  → clips medial cortex\n\n'
                '  Save edits in place. Pipeline reads back from:\n'
                '    {}'.format(sss_refined_final)
            ),
            freeview_args=[
                uni_mpragised_work,
                '{}:colormap=lut:opacity=0.6'.format(sss_refined_final),
                '{}:colormap=heat:opacity=0.4'.format(dura_work),
            ],
            skip_qc=skip_qc,
        )
        shutil.copy(sss_refined_final, sss_refined_work)

    else:
        print('  [skip] No atlas SSS mask available.')
        sss_refined_final = None
        sss_refined_work  = None

    # ------------------------------------------------------------------
    # Step 4c - Subtract dilated SSS → final brain mask
    # ------------------------------------------------------------------
    print('\n[Step 4c] Producing final brain mask...')

    final_mask_final = _final('brain-mask-final')
    final_mask_work  = _work(final_mask_final)

    if not check_skip(
        {'final_mask': final_mask_final},
        ow['finalMask'],
        'Step 4c: final brain mask',
        workdir_paths={'final_mask': final_mask_work},
    ):
        if sss_refined_work and Path(sss_refined_work).exists():
            final_mask_work = make_brain_mask_nosss(
                brain_mask=combined_mask_work,
                sss_mask=sss_refined_work,
                out_dir=workdir,
                sss_dilation_mm=sss_dilation_mm,
            )
        else:
            print('  No SSS mask available; final mask = combined mask.')
            final_mask_work = os.path.join(workdir, 'brain-mask-final.nii.gz')
            shutil.copy(combined_mask_work, final_mask_work)

        shutil.copy(final_mask_work, final_mask_final)

    print('  -> {}'.format(final_mask_work))

    _qc_checkpoint(
        description=(
            'Review the FINAL brain mask before FreeSurfer.\n\n'
            '  What to check:\n'
            '    - Inferior temporal poles and orbitofrontal cortex\n'
            '    - SSS cavity (not too aggressive / not under-carved)\n'
            '    - Cerebellum / brainstem inferior boundary\n'
            '    - No stray islands outside the brain\n\n'
            '  Use coronal + sagittal views. Save edits in place.\n'
            '  Pipeline reads back from:\n'
            '    {}'.format(final_mask_final)
        ),
        freeview_args=[
            uni_mpragised_work,
            '{}:colormap=lut:opacity=0.5'.format(final_mask_final),
            '{}:colormap=lut:opacity=0.4'.format(combined_mask_final),
            '{}:colormap=heat:opacity=0.3'.format(dura_work),
        ],
        skip_qc=skip_qc,
    )
    shutil.copy(final_mask_final, final_mask_work)

    # ------------------------------------------------------------------
    # Step 4d - Apply final brain mask to MPRAGEised UNI
    # ------------------------------------------------------------------
    print('\n[Step 4d] Applying final mask to MPRAGEised UNI...')

    uni_mpragised_brain_fs_final = _final('UNI-mpragised-brain-final')
    uni_mpragised_brain_fs_work  = _work(uni_mpragised_brain_fs_final)

    if not check_skip(
        {'uni_mpragised_brain_final': uni_mpragised_brain_fs_final},
        ow['finalMask'],
        'Step 4d: apply final brain mask',
        workdir_paths={'uni_mpragised_brain_final': uni_mpragised_brain_fs_work},
    ):
        uni_mpragised_brain_fs_work = apply_mask(
            input_image=uni_mpragised_work,
            mask_image=final_mask_work,
            out_dir=workdir,
            out_suffix='_brain_final',
        )
        shutil.copy(uni_mpragised_brain_fs_work, uni_mpragised_brain_fs_final)

    print('  FreeSurfer input -> {}'.format(uni_mpragised_brain_fs_work))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n[Done] All outputs in {}'.format(outdir))
    print('  → FreeSurfer input : {}'.format(uni_mpragised_brain_fs_final))
    print('  → Brain mask       : {}'.format(final_mask_final))
    if sss_refined_final:
        print('  → SSS mask         : {}'.format(sss_refined_final))

    results = {
        'inv2_biascorrected':        inv2_bc_final,
        'inv2_brain':                inv2_brain_final,
        'uni_mpragised':             uni_mpragised_final,
        'uni_mpragised_brain':       uni_mpragised_brain_final,
        'uni_brain':                 uni_brain_final,
        'brain_mask_nighres':        brain_mask_final,
        'cat12_seg_dir':             cat12_out_final,
        'spm_seg_dir':               spm_out_final,
        'sss_atlas_in_t1':           sss_atlas_final,
        'mgdm_segmentation':         mgdm_seg_final,
        'mgdm_distance':             mgdm_dist_final,
        'mgdm_labels':               mgdm_lbls_final,
        'mgdm_memberships':          mgdm_mems_final,
        'dura_probability':          dura_final,
        'brain_mask_combined':       combined_mask_final,
        'sss_mask_refined':          sss_refined_final,
        'brain_mask_final':          final_mask_final,
        'uni_mpragised_brain_final': uni_mpragised_brain_fs_final,
    }
    if t1map:
        results['t1map_brain'] = t1map_brain_final

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description='MP2RAGE preprocessing pipeline (no nipype)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--uni',    required=True, help='UNI image (.nii/.nii.gz)')
    p.add_argument('--inv2',   required=True, help='INV2 image (.nii/.nii.gz)')
    p.add_argument('--t1map',  default=None,
                   help='T1 map (.nii/.nii.gz) — recommended for 7T')
    p.add_argument('--outdir',  required=True, help='Output directory')
    p.add_argument('--subject', required=True, help='BIDS subject label, e.g. sub-01')
    p.add_argument('--session', default=None,  help='BIDS session label, e.g. ses-01')
    p.add_argument('--workdir', default='/tmp/mp2rage_work',
                   help='Working directory for intermediate files')
    p.add_argument('--mp2rage-script-dir', required=True,
                   help='Directory containing SPM/CAT12 m-scripts')
    p.add_argument('--spm-script',     default='s01_spmbc',
                   help='SPM bias correction m-script name')
    p.add_argument('--cat12-script',   default='preproc_cat12seg',
                   help='CAT12 segmentation m-script name')
    p.add_argument('--spm-seg-script', default='preproc_spmseg',
                   help='SPM segmentation m-script name')
    p.add_argument('--spm-standalone', default=None,
                   help='Path to SPM standalone executable')
    p.add_argument('--mcr-path', default=None,
                   help='MATLAB MCR path (required with --spm-standalone)')
    p.add_argument('--atlas-sag-sinus', default=None,
                   help='Dilated atlas SSS mask in MNI space. '
                        'If omitted, Steps 1e / 4b / 4c are skipped.')
    p.add_argument('--fsl-dir', default=None,
                   help='FSL root directory (default: $FSLDIR). '
                        'Required for Step 1e.')
    p.add_argument('--nighres-docker', default='nighres/nighres:latest',
                   help='Nighres Docker image tag')
    p.add_argument('--mgdm-contrast',  default='Mp2rage7T',
                   help='MGDM contrast type')
    p.add_argument('--mgdm-atlas',     default=None,
                   help='MGDM atlas file (uses nighres default if unset)')
    p.add_argument('--dura-background-distance', type=float, default=5.0,
                   help='Max distance within brain mask for dura estimation (mm)')
    p.add_argument('--dura-threshold', type=float, default=0.7,
                   help='Dura probability threshold for brain mask erosion (Step 4a)')
    p.add_argument('--inv2-sss-percentile', type=float, default=15.0,
                   help='INV2 percentile defining dark signal in SSS ROI (Step 4b)')
    p.add_argument('--sss-dilation-mm', type=float, default=3.5,
                   help='SSS mask dilation radius before subtraction (Step 4c, mm)')
    p.add_argument('--skip-qc', action='store_true', default=False,
                   help='Skip manual freeview QC checkpoints (batch use)')

    ow = p.add_argument_group(
        'overwrite options',
        'By default, steps whose outputs exist in outdir are skipped. '
        'Valid step names: ' + ', '.join(STEP_KEYS),
    )
    ow.add_argument(
        '--overwrite',
        nargs='+', metavar='STEP', default=[], choices=STEP_KEYS,
        help='Force re-run for one or more named steps.',
    )
    ow.add_argument(
        '--overwrite-all',
        action='store_true', default=False,
        help='Force re-run for all steps.',
    )

    return p


def main():
    args = _build_parser().parse_args()

    if args.overwrite_all:
        overwrite = {k: True for k in STEP_KEYS}
    else:
        overwrite = {k: (k in args.overwrite) for k in STEP_KEYS}

    run_pipeline(
        uni=args.uni,
        inv2=args.inv2,
        outdir=args.outdir,
        subject=args.subject,
        session=args.session,
        workdir=args.workdir,
        mp2rage_script_dir=args.mp2rage_script_dir,
        t1map=args.t1map,
        spm_script=args.spm_script,
        spm_standalone=args.spm_standalone,
        mcr_path=args.mcr_path,
        cat12_script=args.cat12_script,
        spm_seg_script=args.spm_seg_script,
        atlas_sag_sinus=args.atlas_sag_sinus,
        fsl_dir=args.fsl_dir,
        nighres_docker=args.nighres_docker,
        mgdm_contrast=args.mgdm_contrast,
        mgdm_atlas=args.mgdm_atlas,
        dura_background_distance=args.dura_background_distance,
        dura_threshold=args.dura_threshold,
        inv2_sss_percentile=args.inv2_sss_percentile,
        sss_dilation_mm=args.sss_dilation_mm,
        skip_qc=args.skip_qc,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()