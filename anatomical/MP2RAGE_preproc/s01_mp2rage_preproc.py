#!/usr/bin/env python
"""
run_mp2rage_preproc.py
======================
Run the MP2RAGE preprocessing pipeline sequentially (no nipype).

    Step 0   – SPM bias-field correction of the INV2 image
    Step 1   – MPRAGEise the UNI image using the bias-corrected INV2
    Step 1b  – Nighres skull stripping → brain mask only
    Step 1c  – Apply brain mask to every image:
                 · raw UNI              → UNI_brain           (→ MGDM)
                 · MPRAGEised UNI       → UNI_mpragised_brain  (→ FreeSurfer)
                 · bias-corrected INV2  → INV2_brain           (→ QC)
                 · T1map (optional)     → T1map_brain          (→ MGDM)
    Step 2   – Nighres MGDM segmentation:
                 · contrast_image1 = UNI_brain
                 · contrast_image2 = T1map_brain (when available)
    Step 3   – Nighres dura estimation:
                · second_inversion = bias-corrected INV2
                · skullstrip_mask  = brain mask

Overwrite behaviour
-------------------
Existence is checked against the final BIDS-named files in *outdir*.
If an output already exists there and overwrite is False, the step is
skipped and the existing file is copied back into *workdir* so that
downstream steps can use it as normal.

Pass overwrite flags to force specific steps to re-run:

    # Re-run only MGDM and dura estimation
    python run_mp2rage_preproc.py ... --overwrite mgdm dura

    # Re-run everything
    python run_mp2rage_preproc.py ... --overwrite-all

Valid step names for --overwrite:
    spmbc       Step 0  – SPM bias-field correction
    mpragise    Step 1  – MPRAGEise
    skullstrip  Step 1b – Nighres skull stripping
    applymask   Step 1c – Apply brain mask (all masked images together)
    mgdm        Step 2  – Nighres MGDM segmentation
    dura        Step 3  – Nighres dura estimation

Usage examples
--------------
# SPM standalone mode
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
    --mcr-path           /opt/mcr/v99

# MATLAB mode (omit --spm-standalone and --mcr-path)
python run_mp2rage_preproc.py \\
    --uni         /data/sub-01/ses-01/anat/sub-01_ses-01_UNI.nii.gz \\
    --inv2        /data/sub-01/ses-01/anat/sub-01_ses-01_INV2.nii.gz \\
    --outdir      /out \\
    --subject     sub-01 \\
    --mp2rage-script-dir /opt/mp2rage_scripts
"""

import argparse
import os
import shutil
from pathlib import Path

from preproc_utils import (
    check_skip,
    spm_bias_correct,
    mprage_ise,
    nighres_skull_strip,
    apply_mask,
    nighres_mgdm,
    nighres_dura_estimation,
)

# All valid step keys, in pipeline order
STEP_KEYS = ['spmbc', 'mpragise', 'skullstrip', 'applymask', 'mgdm', 'dura']


# ---------------------------------------------------------------------------
# Output filename helper
# ---------------------------------------------------------------------------

def build_output_name(outdir: str, subject: str, session: str,
                      suffix: str, extension: str = '.nii.gz') -> str:
    """
    Build a BIDS-style output filename.

    Examples
    --------
    >>> build_output_name('/out', 'sub-01', 'ses-01', 'T1w-mpragised')
    '/out/sub-01_ses-01_T1w-mpragised.nii.gz'
    >>> build_output_name('/out', 'sub-01', None, 'T1w-mpragised')
    '/out/sub-01_T1w-mpragised.nii.gz'
    """
    tokens = [t for t in [subject, session, suffix] if t]
    return os.path.join(outdir, '_'.join(tokens) + extension)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    uni: str,
    inv2: str,
    outdir: str,
    subject: str,
    mp2rage_script_dir: str,
    session: str = None,
    workdir: str = '/tmp/mp2rage_work',
    t1map: str = None,
    spm_script: str = 's01_spmbc',
    spm_standalone: str = None,
    mcr_path: str = None,
    nighres_docker: str = 'nighres/nighres:latest',
    mgdm_contrast: str = 'Mp2rage7T',
    mgdm_atlas: str = None,
    dura_background_distance: float = 5.0,
    overwrite: dict = None,
) -> dict:
    """
    Run the full MP2RAGE preprocessing pipeline and copy outputs to *outdir*.

    Existence checks are performed against final BIDS-named files in *outdir*.
    Skipped steps have their outputs restored from *outdir* into *workdir* so
    that all downstream steps can continue to read from *workdir* as normal.

    Parameters
    ----------
    overwrite : dict mapping step keys to booleans, e.g.
                {'spmbc': False, 'mgdm': True}.
                Missing keys default to False (don't overwrite).
                Valid keys: 'spmbc', 'mpragise', 'skullstrip',
                            'applymask', 'mgdm', 'dura'.

    Returns a dict mapping output names to their final paths in outdir.
    """
    ow = {k: False for k in STEP_KEYS}
    if overwrite:
        unknown = set(overwrite) - set(STEP_KEYS)
        if unknown:
            raise ValueError(
                'Unknown overwrite key(s): {}. Valid keys are: {}'.format(
                    sorted(unknown), STEP_KEYS)
            )
        ow.update(overwrite)

    os.makedirs(outdir,  exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    workdir = str(Path(workdir).resolve())
    outdir  = str(Path(outdir).resolve())

    # Convenience: build a final (outdir) path for a given suffix
    def _final(suffix):
        return build_output_name(outdir, subject, session, suffix)

    # Convenience: workdir path with the same basename as a final path
    def _work(final_path):
        return os.path.join(workdir, os.path.basename(final_path))

    # ------------------------------------------------------------------
    # Step 0 – SPM bias-field correction of INV2
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
    # Step 1 – MPRAGEise UNI with bias-corrected INV2
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
    # Step 1b – Nighres skull stripping → brain mask only
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
    # Step 1c – Apply brain mask to every image
    # ------------------------------------------------------------------
    print('\n[Step 1c] Applying brain mask...')

    uni_brain_final           = _final('UNI-brain')
    uni_mpragised_brain_final = _final('UNI-mpragised-brain')
    inv2_brain_final          = _final('INV2-spmbc-brain')
    t1map_brain_final         = _final('T1map-brain') if t1map else None

    outdir_applymask = {
        'uni_brain':          uni_brain_final,
        'uni_mpragised_brain': uni_mpragised_brain_final,
        'inv2_brain':         inv2_brain_final,
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
        # Restored from outdir by check_skip; read back workdir paths
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
    # Step 2 – Nighres MGDM segmentation
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
    # Step 3 – Nighres dura estimation
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
    # Return final outdir paths
    # ------------------------------------------------------------------
    print('\n[Done] All outputs in {}'.format(outdir))

    results = {
        'inv2_biascorrected':  inv2_bc_final,
        'inv2_brain':          inv2_brain_final,
        'uni_mpragised':       uni_mpragised_final,
        'uni_mpragised_brain': uni_mpragised_brain_final,
        'uni_brain':           uni_brain_final,
        'brain_mask':          brain_mask_final,
        'mgdm_segmentation':   mgdm_seg_final,
        'mgdm_distance':       mgdm_dist_final,
        'mgdm_labels':         mgdm_lbls_final,
        'mgdm_memberships':    mgdm_mems_final,
        'dura_probability':    dura_final,
    }
    if t1map:
        results['t1map_brain'] = t1map_brain_final

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='MP2RAGE preprocessing pipeline (no nipype)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--uni',    required=True, help='UNI image (.nii/.nii.gz)')
    p.add_argument('--inv2',   required=True, help='INV2 image (.nii/.nii.gz)')
    p.add_argument('--t1map',  default=None,
                   help='T1 map (.nii/.nii.gz) — strongly recommended for 7T')
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--subject', required=True,
                   help='BIDS subject label e.g. sub-01')
    p.add_argument('--session', default=None,
                   help='BIDS session label e.g. ses-01')
    p.add_argument('--workdir', default='/tmp/mp2rage_work',
                   help='Working directory for intermediate files')
    p.add_argument('--mp2rage-script-dir', required=True,
                   help='Directory containing SPM m-script')
    p.add_argument('--spm-script', default='s01_spmbc',
                   help='SPM m-script name')
    p.add_argument('--spm-standalone', default=None,
                   help='Path to SPM standalone executable')
    p.add_argument('--mcr-path', default=None,
                   help='Path to MATLAB MCR (required if --spm-standalone set)')
    p.add_argument('--nighres-docker', default='nighres/nighres:latest',
                   help='Nighres Docker image tag')
    p.add_argument('--mgdm-contrast', default='Mp2rage7T',
                   help='MGDM contrast type')
    p.add_argument('--mgdm-atlas', default=None,
                   help='MGDM atlas file (uses nighres default if unset)')
    p.add_argument('--dura-background-distance', type=float, default=5.0,
                   help='Max distance within brain mask for dura estimation (mm)')

    ow_group = p.add_argument_group(
        'overwrite options',
        'By default, steps whose outputs already exist in outdir are skipped '
        'and their files are restored to workdir for downstream use. '
        'Use the flags below to force specific steps to re-run.\n'
        'Valid step names: ' + ', '.join(STEP_KEYS),
    )
    ow_group.add_argument(
        '--overwrite',
        nargs='+',
        metavar='STEP',
        default=[],
        choices=STEP_KEYS,
        help='Force re-run for one or more named steps.',
    )
    ow_group.add_argument(
        '--overwrite-all',
        action='store_true',
        default=False,
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
        nighres_docker=args.nighres_docker,
        mgdm_contrast=args.mgdm_contrast,
        mgdm_atlas=args.mgdm_atlas,
        dura_background_distance=args.dura_background_distance,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()
    # next step
    # recon-all -subjid sub-03mprageisenighresbmask -i sub-03_ses-1_UNI-mpragised-brain.nii.gz  -autorecon1 -noskullstrip -hires