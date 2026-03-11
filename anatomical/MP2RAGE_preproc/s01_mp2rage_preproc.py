#!/usr/bin/env python
"""
run_mp2rage_preproc.py
======================
Run the MP2RAGE preprocessing pipeline sequentially (no nipype).

    Step 0   – SPM bias-field correction of the INV2 image
    Step 1   – MPRAGEise the UNI image using the bias-corrected INV2
    Step 1b  – Nighres skull stripping → brain mask only
    Step 1c  – Apply brain mask to every image:
                 · raw UNI              → UNI_brain         (→ MGDM)
                 · MPRAGEised UNI       → UNI_mpragised_brain (→ FreeSurfer)
                 · bias-corrected INV2  → INV2_brain        (→ QC)
                 · T1map (optional)     → T1map_brain       (→ MGDM)
    Step 2   – Nighres MGDM segmentation:
                 · contrast_image1 = UNI_brain
                 · contrast_image2 = T1map_brain (when available)

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
    spm_bias_correct,
    mprage_ise,
    nighres_skull_strip,
    apply_mask,
    nighres_mgdm,
)


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
) -> dict:
    """
    Run the full MP2RAGE preprocessing pipeline and copy outputs to *outdir*.

    Returns a dict mapping output names to their final paths.
    """
    os.makedirs(outdir,  exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0 – SPM bias-field correction of INV2
    # ------------------------------------------------------------------
    print('\n[Step 0] SPM bias-field correction of INV2...')
    inv2_bc = spm_bias_correct(
        input_image=inv2,
        out_dir=workdir,
        mp2rage_script_dir=mp2rage_script_dir,
        spm_script=spm_script,
        spm_standalone=spm_standalone,
        mcr_path=mcr_path,
    )
    print('  -> {}'.format(inv2_bc))

    # ------------------------------------------------------------------
    # Step 1 – MPRAGEise UNI with bias-corrected INV2
    # ------------------------------------------------------------------
    print('\n[Step 1] MPRAGEising UNI...')
    uni_mpragised = mprage_ise(
        uni_file=uni,
        inv2_file=inv2_bc,
        out_dir=workdir,
    )
    print('  -> {}'.format(uni_mpragised))

    # ------------------------------------------------------------------
    # Step 1b – Nighres skull stripping → brain mask only
    # ------------------------------------------------------------------
    print('\n[Step 1b] Nighres skull stripping...')
    brain_mask = nighres_skull_strip(
        inv2_image=inv2_bc,
        uni_image=uni,
        out_dir=workdir,
        t1map_image=t1map,
        docker_image=nighres_docker,
    )
    print('  -> {}'.format(brain_mask))

    # ------------------------------------------------------------------
    # Step 1c – Apply brain mask to every image
    # ------------------------------------------------------------------
    print('\n[Step 1c] Applying brain mask...')

    uni_brain = apply_mask(
        input_image=uni,
        mask_image=brain_mask,
        out_dir=workdir,
        out_suffix='_brain',
    )
    print('  UNI brain          -> {}'.format(uni_brain))

    uni_mpragised_brain = apply_mask(
        input_image=uni_mpragised,
        mask_image=brain_mask,
        out_dir=workdir,
        out_suffix='_brain',
    )
    print('  UNI mpragised brain -> {}'.format(uni_mpragised_brain))

    inv2_brain = apply_mask(
        input_image=inv2_bc,
        mask_image=brain_mask,
        out_dir=workdir,
        out_suffix='_brain',
    )
    print('  INV2 brain         -> {}'.format(inv2_brain))

    t1map_brain = None
    if t1map:
        t1map_brain = apply_mask(
            input_image=t1map,
            mask_image=brain_mask,
            out_dir=workdir,
            out_suffix='_brain',
        )
        print('  T1map brain        -> {}'.format(t1map_brain))

    # ------------------------------------------------------------------
    # Step 2 – Nighres MGDM segmentation
    # ------------------------------------------------------------------
    print('\n[Step 2] Nighres MGDM segmentation...')
    mgdm_outputs = nighres_mgdm(
        input_image=uni_brain,
        out_dir=workdir,
        docker_image=nighres_docker,
        contrast_type=mgdm_contrast,
        t1map_image=t1map_brain,
        atlas=mgdm_atlas,
    )
    print('  segmentation -> {}'.format(mgdm_outputs['segmentation']))

    # ------------------------------------------------------------------
    # Copy outputs to outdir with BIDS-style names
    # ------------------------------------------------------------------
    print('\n[Output] Copying results to {}...'.format(outdir))

    def _copy(src, suffix):
        dst = build_output_name(outdir, subject, session, suffix)
        shutil.copy(src, dst)
        print('  {} -> {}'.format(suffix, dst))
        return dst

    results = {
        'inv2_biascorrected':    _copy(inv2_bc,              'INV2-spmbc'),
        'inv2_brain':            _copy(inv2_brain,            'INV2-spmbc-brain'),
        'uni_mpragised':         _copy(uni_mpragised,         'UNI-mpragised'),
        'uni_mpragised_brain':   _copy(uni_mpragised_brain,   'UNI-mpragised-brain'),
        'uni_brain':             _copy(uni_brain,             'UNI-brain'),
        'brain_mask':            _copy(brain_mask,            'brain-mask'),
        'mgdm_segmentation':     _copy(mgdm_outputs['segmentation'], 'mgdm-seg'),
        'mgdm_distance':         _copy(mgdm_outputs['distance'],     'mgdm-dist'),
        'mgdm_labels':           _copy(mgdm_outputs['labels'],       'mgdm-lbls'),
        'mgdm_memberships':      _copy(mgdm_outputs['memberships'],  'mgdm-mems'),
    }
    if t1map_brain:
        results['t1map_brain'] = _copy(t1map_brain, 'T1map-brain')

    print('\n[Done] All outputs written to {}'.format(outdir))
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
    return p


def main():
    args = _build_parser().parse_args()
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
    )


if __name__ == '__main__':
    main()
    # next step
    # recon-all -subjid sub-03mprageisenighresbmask -i sub-03_ses-1_UNI-mpragised-brain.nii.gz  -autorecon1 -noskullstrip -hires 