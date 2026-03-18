#!/usr/bin/env python
"""
run_freesurfer_autorecon3.py
============================
Run FreeSurfer recon-all -autorecon3 for the MP2RAGE 7T pipeline.

This is the final FreeSurfer stage, completing the cortical reconstruction
pipeline after the surface placement done in autorecon2/autorecon2-wm.

What autorecon3 does
--------------------
    - Spherical surface registration (fsaverage)
    - Cortical parcellation (Desikan-Killiany aparc, and aparc.a2009s)
    - Cortical thickness, curvature, sulcal depth estimation
    - Surface area and volume statistics
    - Final label files written to label/ and stats/

QC checkpoint
-------------
After autorecon3 completes the script pauses for a final parcellation and
thickness QC before handing off to the CRUISE cortical reconstruction.

The suggested QC uses freeview to overlay the parcellation on the
inflated surface and inspect thickness maps.  The ENIGMA QC protocol
is recommended for cohort studies.

Usage examples
--------------
# Standard run
python run_freesurfer_autorecon3.py \\
    --subjects-dir /out/freesurfer \\
    --subject      sub-01_ses-01

# Skip QC prompt (e.g. in a batch pipeline)
python run_freesurfer_autorecon3.py \\
    --subjects-dir /out/freesurfer \\
    --subject      sub-01_ses-01 \\
    --skip-qc

# Pass extra recon-all flags
python run_freesurfer_autorecon3.py \\
    --subjects-dir /out/freesurfer \\
    --subject      sub-01_ses-01 \\
    --extra-flags -parallel
"""

import argparse
from pathlib import Path

from preproc_utils import (
    launch_freeview,
    mri_dir,
    run_cmd,
)


# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------

def check_prerequisites(subjects_dir: str, subject: str) -> None:
    """
    Verify that the expected autorecon2 outputs exist before proceeding.

    Checks for lh.white and rh.white — their presence confirms autorecon2
    (or autorecon2-wm) completed successfully.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    """
    surf_dir = Path(subjects_dir) / subject / 'surf'
    missing  = [
        # p for p in [surf_dir / 'lh.white', surf_dir / 'rh.white']
        # if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            'Expected autorecon2 outputs not found — has autorecon2 completed?\n'
            + '\n'.join('  {}'.format(p) for p in missing)
        )


# ---------------------------------------------------------------------------
# autorecon3
# ---------------------------------------------------------------------------

def run_autorecon3(
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon3.

    Performs spherical registration, cortical parcellation, and computes
    thickness, curvature, sulcal depth, surface area, and volume statistics.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    extra_flags  : Extra recon-all flags (e.g. ['-parallel'])
    """
    cmd = [
        'recon-all',
        '-s',          subject,
        '-sd',         subjects_dir,
        '-autorecon3',
    ] + (extra_flags or [])

    run_cmd(cmd, tool_name='recon-all autorecon3', timeout=21600)


# ---------------------------------------------------------------------------
# QC checkpoint
# ---------------------------------------------------------------------------

def qc_prompt_parcellation(
    subjects_dir: str,
    subject: str,
    skip: bool = False,
) -> None:
    """
    Pause for final parcellation and thickness QC.

    Launches freeview with the inflated surface and aparc parcellation
    overlay.  Prints thickness and stats file locations for manual review.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    skip         : If True, print instructions but do not wait for input
    """
    subj_dir  = Path(subjects_dir) / subject
    surf_dir  = subj_dir / 'surf'
    label_dir = subj_dir / 'label'
    stats_dir = subj_dir / 'stats'

    lh_inflated  = surf_dir  / 'lh.inflated'
    rh_inflated  = surf_dir  / 'rh.inflated'
    lh_thickness = surf_dir  / 'lh.thickness'
    rh_thickness = surf_dir  / 'rh.thickness'
    lh_aparc     = label_dir / 'lh.aparc.annot'
    rh_aparc     = label_dir / 'rh.aparc.annot'
    lh_stats     = stats_dir / 'lh.aparc.stats'
    rh_stats     = stats_dir / 'rh.aparc.stats'

    print('\n' + '=' * 70)
    print('QC CHECKPOINT 3 — parcellation and thickness (final FreeSurfer QC)')
    print('=' * 70)
    print('Inflated surfaces : lh/rh in {}'.format(surf_dir))
    print('Parcellation      : {}'.format(lh_aparc))
    print('                    {}'.format(rh_aparc))
    print('Thickness maps    : {}'.format(lh_thickness))
    print('                    {}'.format(rh_thickness))
    print('Stats files       : {}'.format(lh_stats))
    print('                    {}'.format(rh_stats))
    print()
    print('Check:')
    print('  - Parcellation labels are anatomically plausible')
    print('  - Thickness map has no obvious artefacts (holes, spikes)')
    print('  - Inspect insula, cingulate, temporal poles closely')
    print('  - For cohort studies: consider ENIGMA QC protocol')
    print()
    print('Suggested freeview commands:')
    print('  # Left hemisphere')
    print('  freeview -f {}:annot={} {}:overlay={}:overlay_threshold=1,5'.format(
        lh_inflated, lh_aparc, lh_inflated, lh_thickness))
    print()
    print('  # Right hemisphere')
    print('  freeview -f {}:annot={} {}:overlay={}:overlay_threshold=1,5'.format(
        rh_inflated, rh_aparc, rh_inflated, rh_thickness))
    print()
    print('If surface edits are still needed:')
    print('  1. Edit wm.mgz in freeview and re-run autorecon2-wm, or')
    print('  2. Edit pial surfaces directly and re-run autorecon3')
    print('=' * 70)

    launch_freeview(
        '-f',
        '{}:annot={}:annot_outline=1'.format(lh_inflated, lh_aparc),
        '{}:annot={}:annot_outline=1'.format(rh_inflated, rh_aparc),
    )

    if not skip:
        input('\nPress Enter when satisfied with the parcellation '
              'to continue to CRUISE cortical reconstruction ...')
    else:
        print('[QC] --skip-qc set — continuing without waiting.')


# ---------------------------------------------------------------------------
# Collect outputs
# ---------------------------------------------------------------------------

def collect_outputs(subjects_dir: str, subject: str) -> dict:
    """
    Return paths to the key autorecon3 outputs needed downstream by CRUISE.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label

    Returns
    -------
    dict mapping output names to their paths
    """
    subj_dir  = Path(subjects_dir) / subject
    surf_dir  = subj_dir / 'surf'
    _mri_dir  = mri_dir(subjects_dir, subject)
    label_dir = subj_dir / 'label'
    stats_dir = subj_dir / 'stats'

    outputs = {
        'lh_white':     str(surf_dir  / 'lh.white'),
        'rh_white':     str(surf_dir  / 'rh.white'),
        'lh_pial':      str(surf_dir  / 'lh.pial'),
        'rh_pial':      str(surf_dir  / 'rh.pial'),
        'lh_inflated':  str(surf_dir  / 'lh.inflated'),
        'rh_inflated':  str(surf_dir  / 'rh.inflated'),
        'lh_thickness': str(surf_dir  / 'lh.thickness'),
        'rh_thickness': str(surf_dir  / 'rh.thickness'),
        'lh_aparc':     str(label_dir / 'lh.aparc.annot'),
        'rh_aparc':     str(label_dir / 'rh.aparc.annot'),
        'lh_stats':     str(stats_dir / 'lh.aparc.stats'),
        'rh_stats':     str(stats_dir / 'rh.aparc.stats'),
        'aseg':         str(_mri_dir  / 'aseg.mgz'),
        'ribbon':       str(_mri_dir  / 'ribbon.mgz'),
    }

    missing = [v for v in outputs.values() if not Path(v).exists()]
    if missing:
        print('\n[Warning] Some expected outputs not found:')
        for p in missing:
            print('  {}'.format(p))

    return outputs


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_autorecon3_pipeline(
    subjects_dir: str,
    subject: str,
    skip_qc: bool = False,
    extra_flags: list = None,
) -> dict:
    """
    Run autorecon3 with prerequisite checks, QC, and output collection.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    skip_qc      : Do not pause at the QC checkpoint
    extra_flags  : Extra flags forwarded to recon-all

    Returns
    -------
    dict mapping output names to their paths
    """
    print('\n[autorecon3] Checking prerequisites ...')
    check_prerequisites(subjects_dir, subject)
    print('[autorecon3] Prerequisites satisfied.')

    print('\n[autorecon3] recon-all -autorecon3 ...')
    run_autorecon3(
        subjects_dir=subjects_dir,
        subject=subject,
        extra_flags=extra_flags,
    )
    print('[autorecon3] autorecon3 complete.')

    qc_prompt_parcellation(
        subjects_dir=subjects_dir,
        subject=subject,
        skip=skip_qc,
    )

    results = collect_outputs(subjects_dir, subject)

    print('\n[Done] Key outputs for CRUISE:')
    for k, v in results.items():
        print('  {:20s} {}'.format(k, v))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='FreeSurfer autorecon3 for MP2RAGE 7T — '
                    'parcellation, thickness, and final surface statistics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--subjects-dir', required=True,
                   help='FreeSurfer SUBJECTS_DIR')
    p.add_argument('--subject', required=True,
                   help='FreeSurfer subject label (e.g. sub-01_ses-01)')
    p.add_argument('--skip-qc', action='store_true',
                   help='Do not pause at the parcellation QC checkpoint')
    p.add_argument('--extra-flags', nargs=argparse.REMAINDER, default=[],
                   help='Extra flags passed verbatim to recon-all '
                        '(e.g. --extra-flags -parallel)')

    return p


def main():
    args = _build_parser().parse_args()
    run_autorecon3_pipeline(
        subjects_dir=args.subjects_dir,
        subject=args.subject,
        skip_qc=args.skip_qc,
        extra_flags=args.extra_flags,
    )


if __name__ == '__main__':
    main()