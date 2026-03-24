#!/usr/bin/env python
"""
run_mp2rage_preproc.py
======================
Run the MP2RAGE preprocessing pipeline sequentially (no nipype).

    Step 0   - SPM bias-field correction of the INV2 image
    Step 1   - MPRAGEise the UNI image using the bias-corrected INV2
    Step 2   - spm segmentation on INV2

Overwrite behaviour
-------------------
Existence is checked against the final BIDS-named files in *outdir*.
If an output already exists there and overwrite is False, the step is
skipped and the existing file is copied back into *workdir* so that
downstream steps can use it as normal.

Pass overwrite flags to force specific steps to re-run:

    # Re-run only the mask combination and everything downstream
    python run_mp2rage_preproc.py ... --overwrite mprageise

    # Re-run everything
    python run_mp2rage_preproc.py ... --overwrite-all

Valid step names for --overwrite:
    spmbc        
    mpragise     
    spm12seg     
    applymask  

Usage examples
--------------
# SPM standalone mode
python run_mp2rage_preproc.py \\
    --uni         /data/sub-01/ses-01/anat/sub-01_ses-01_UNI.nii.gz \\
    --inv2        /data/sub-01/ses-01/anat/sub-01_ses-01_INV2.nii.gz \\
    --outdir      /out/sub-01/ses-01/anat \\
    --subject     sub-01 \\
    --session     ses-01 \\
    --workdir     /tmp/mp2rage_work \\
    --mp2rage-script-dir /opt/mp2rage_scripts \\
    --spm-standalone     /opt/spm12/run_spm12.sh \\
    --mcr-path           /opt/mcr/v99 \\


"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from preproc_utils import (
    check_skip,
    spm_bias_correct,
    mprage_ise,
    spm_seg,
    get_stem
)

# All valid step keys, in pipeline order
STEP_KEYS = [
    'spmbc',
    'mpragise',
    'spmseg',
]


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
    spm_script: str = 'preproc_spmbc',
    spm_standalone: str = None,
    mcr_path: str = None,
    overwrite: dict = None,
) -> dict:
    """
    Run the full MP2RAGE preprocessing pipeline and copy outputs to *outdir*.

    Existence checks are performed against final BIDS-named files in *outdir*.
    Skipped steps have their outputs restored from *outdir* into *workdir* so
    that all downstream steps can continue to read from *workdir* as normal.

    Parameters
    ----------
    overwrite            : dict mapping step keys → bool.  Missing keys
                           default to False.  Valid keys: see STEP_KEYS.

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

    def _final(suffix, ext='.nii.gz'):
        return build_output_name(outdir, subject, session, suffix,
                                 extension=ext)

    def _work(final_path):
        return os.path.join(workdir, os.path.basename(final_path))

    prefix = '_'.join(t for t in [subject, session] if t)

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
    # SPM segmentation (MPRAGEised UNI, with skull)
    # ------------------------------------------------------------------
    print('\n[Step *] SPM segmentation...')
    k = inv2_bc_work
    _t_stem = get_stem(Path(k))
    spm_seg_out_final   = os.path.join(
        outdir, '{}_{}_spmseg'.format(prefix, _t_stem))

    # Gate on c1 (GM map) as the sentinel — written last by SPM unified seg
    spm_seg_gm_final = os.path.join(
        spm_seg_out_final,
        '{}_GM_native.nii'.format(_t_stem))
    if not check_skip(
        {'spm_seg_gm': spm_seg_gm_final},
        ow['spmseg'],
        'Step 1d-ii: SPM segmentation',
    ):
        spm_seg_outputs = spm_seg(
            input_image=k,
            out_dir=workdir,
            mp2rage_script_dir=mp2rage_script_dir,
            spm_script='preproc_spmseg',
            spm_standalone=spm_standalone,
            mcr_path=mcr_path,
        )
        if os.path.exists(spm_seg_out_final):
            shutil.rmtree(spm_seg_out_final)
        shutil.copytree(spm_seg_outputs, spm_seg_out_final)

    print('  -> {}'.format(spm_seg_out_final))

    # Copy brainmask to parent folder
    brainmask_src = os.path.join(spm_seg_out_final, '{}_stripbrainmask.nii'.format(_t_stem))
    brainmask_dst = os.path.join(outdir, f'{subject}_{session}_brainmask.nii')
    print(brainmask_src)
    if os.path.exists(brainmask_src):
        shutil.copy(brainmask_src, brainmask_dst)

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
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--subject', required=True,
                   help='BIDS subject label e.g. sub-01')
    p.add_argument('--session', default=None,
                   help='BIDS session label e.g. ses-01')
    p.add_argument('--workdir', default='/tmp/mp2rage_work',
                   help='Working directory for intermediate files')
    p.add_argument('--mp2rage-script-dir', required=True,
                   help='Directory containing SPM m-scripts '
                        '(preproc_spmbc.m, preproc_cat12seg.m)')
    p.add_argument('--spm-standalone', default=None,
                   help='Path to SPM standalone executable')
    p.add_argument('--mcr-path', default=None,
                   help='Path to MATLAB MCR (required if --spm-standalone set)')
    p.add_argument('--fsl-dir', default=None,
                   help='FSL installation root (default: $FSLDIR). '
                        'Required for Step 1e.')

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
        spm_standalone=args.spm_standalone,
        mcr_path=args.mcr_path,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()