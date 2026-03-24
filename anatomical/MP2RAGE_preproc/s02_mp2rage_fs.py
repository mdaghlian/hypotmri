#!/usr/bin/env python
"""
run_freesurfer_recon.py
=======================
Run FreeSurfer recon-all stages for the MP2RAGE 7T pipeline.

Stage ordering
--------------
    Stage 3   – recon-all -autorecon1 -noskullstrip
                  Conforms the full-head UNI input, runs Talairach registration.
                  FreeSurfer's own skull stripping is skipped — the nighres
                  mask injected in Stage 4a is superior for 7T MP2RAGE data.
                  The full-head image is required here so that Talairach
                  registration has correct head geometry to work from.

    Stage 4a  – Inject nighres brain mask → brainmask.mgz
                  Resamples the nighres mask to T1.mgz space (nearest-neighbour),
                  then computes:  brainmask = (nighres_mask > 0) * T1
                  This zeros non-brain voxels while preserving T1 intensities —
                  the format FreeSurfer expects.  Also writes
                  brain.finalsurfs.manedit.mgz.  Backups created before any
                  file is overwritten.

    QC #1     – Inspect brainmask.mgz before surface generation.
                  Pipeline pauses here (pass --skip-qc-1 to bypass).

    Stage 4b  – recon-all -autorecon2
                  FreeSurfer computes wm.mgz, tessellates surfaces, and places
                  white + pial surfaces.

    QC #2     – Inspect wm.mgz and white/pial surfaces.
                  Pipeline pauses here (pass --skip-qc-2 to bypass).

    Stage 4c  – recon-all -autorecon3
                  Cortical parcellation, thickness, curvature stats, etc.

Why this order?
---------------
autorecon1 must receive a full-head image — Talairach registration relies on
overall head shape and proportions to find the AC-PC line.  Passing a
skull-stripped image can degrade or break this registration.  The nighres brain
mask is injected immediately after autorecon1 so that autorecon2 uses our mask
rather than FreeSurfer's skull-strip result.  autorecon3 then runs on the
finished surfaces.

Overwrite behaviour
-------------------
Each stage checks whether its sentinel output already exists in the FreeSurfer
subject directory.  If it does and overwrite is False, the stage is skipped.

Pass overwrite flags to force specific stages to re-run:

    # Re-run brain mask injection and autorecon2 only
    python run_freesurfer_recon.py ... --overwrite inject_brainmask autorecon2

    # Re-run everything
    python run_freesurfer_recon.py ... --overwrite-all

Valid stage keys for --overwrite:
    autorecon1        Stage 3  – sentinel: mri/T1.mgz
    inject_brainmask  Stage 4a – sentinel: mri/brainmask.mgz
    autorecon2        Stage 4b – sentinel: mri/wm.mgz
    autorecon3        Stage 4c – sentinel: surf/lh.thickness

Usage examples
--------------
# Minimal run
python run_freesurfer_recon.py \\
    --uni-mpragised    /out/sub-01_ses-01_UNI-mpragised.nii.gz \\
    --brain-mask       /out/sub-01_ses-01_brain-mask.nii.gz \\
    --subjects-dir     /out/freesurfer \\
    --subject          sub-01_ses-01

# With manually edited brain mask
python run_freesurfer_recon.py \\
    --uni-mpragised    /out/sub-01_ses-01_UNI-mpragised.nii.gz \\
    --brain-mask       /out/sub-01_ses-01_brain-mask.nii.gz \\
    --brain-mask-edited /out/sub-01_ses-01_brain-mask-edited.nii.gz \\
    --subjects-dir     /out/freesurfer \\
    --subject          sub-01_ses-01

# Re-entry after manual brainmask edits — re-inject and re-run from autorecon2
python run_freesurfer_recon.py \\
    --uni-mpragised    /out/sub-01_ses-01_UNI-mpragised.nii.gz \\
    --brain-mask       /out/sub-01_ses-01_brain-mask.nii.gz \\
    --brain-mask-edited /out/sub-01_ses-01_brain-mask-edited.nii.gz \\
    --subjects-dir     /out/freesurfer \\
    --subject          sub-01_ses-01 \\
    --skip-autorecon1 --skip-qc-1 \\
    --overwrite inject_brainmask autorecon2 autorecon3
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from preproc_utils import (
    backup_file,
    check_skip,
    launch_freeview,
    mri_dir,
    resample_to_mgh,
    run_cmd,
)


# All valid stage keys, in pipeline order
STAGE_KEYS = [
    'autorecon1',
    'inject_brainmask',
    'autorecon2',
    'autorecon3',
]


# ---------------------------------------------------------------------------
# Stage 3 – autorecon1
# ---------------------------------------------------------------------------

def run_autorecon1(
    uni_mpragised: str,
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon1 -noskullstrip.

    Uses the full-head MPRAGEised UNI image.  FreeSurfer's own skull stripping
    is deliberately skipped — the nighres mask injected in Stage 4a is superior
    for 7T MP2RAGE data.  The full-head image is required so that Talairach
    registration has correct head geometry.

    Parameters
    ----------
    uni_mpragised : Full-head MPRAGEised UNI (.nii.gz)
    subjects_dir  : FreeSurfer SUBJECTS_DIR
    subject       : FreeSurfer subject label
    extra_flags   : Extra recon-all flags (e.g. ['-parallel'])
    """
    cmd = [
        'recon-all',
        '-i',            uni_mpragised,
        '-s',            subject,
        '-sd',           subjects_dir,
        '-autorecon1',
        '-noskullstrip',
    ] + (extra_flags or [])

    run_cmd(cmd, tool_name='recon-all autorecon1', timeout=7200)


# ---------------------------------------------------------------------------
# Stage 4a – inject brain mask
# ---------------------------------------------------------------------------

def inject_brain_mask(
    brain_mask: str,
    subjects_dir: str,
    subject: str,
    brain_mask_edited: str = None,
) -> Path:
    """
    Inject the nighres brain mask into the FreeSurfer subject directory.

    Workflow
    --------
    1. Backup existing brainmask.mgz (if present).
    2. Resample the nighres mask (or edited override) to T1.mgz space using
       nearest-neighbour interpolation.
    3. Compute:  brainmask = (nighres_mask > 0) * T1
       Zeros non-brain voxels while preserving T1 intensities inside the mask —
       the format FreeSurfer expects.  The T1.mgz affine and header are
       preserved so FreeSurfer sees a valid MGH volume.
    4. Write brainmask.mgz and brain.finalsurfs.manedit.mgz (checked by
       FreeSurfer during pial surface refinement).
    5. Save the resampled nighres mask as brainmask_nighres.mgz for audit.

    Parameters
    ----------
    brain_mask        : nighres brain mask (.nii.gz)
    subjects_dir      : FreeSurfer SUBJECTS_DIR
    subject           : FreeSurfer subject label
    brain_mask_edited : Manually edited mask — overrides brain_mask if given

    Returns
    -------
    Path to the written brainmask.mgz
    """
    mri_path       = mri_dir(subjects_dir, subject)
    t1_mgz         = mri_path / 'T1.mgz'
    brainmask_mgz  = mri_path / 'brainmask.mgz'
    finalsurfs_mgz = mri_path / 'brain.finalsurfs.manedit.mgz'
    nighres_mgz    = mri_path / 'brainmask_nighres.mgz'

    if not t1_mgz.exists():
        raise FileNotFoundError(
            'T1.mgz not found — has autorecon1 completed?\n'
            '  Expected: {}'.format(t1_mgz)
        )

    if brainmask_mgz.exists():
        backup_file(brainmask_mgz)

    mask_to_use = brain_mask_edited if brain_mask_edited else brain_mask
    print('\n[inject_brain_mask] Using mask: {}'.format(mask_to_use))

    # Resample mask to T1.mgz space (nearest-neighbour) and save audit copy
    mask_mgh = resample_to_mgh(mask_to_use, t1_mgz)
    mask_mgh.to_filename(str(nighres_mgz))

    # Load T1 to get data + geometry; load resampled mask for its data
    t1_img   = nib.load(str(t1_mgz))
    mask_img = nib.load(str(nighres_mgz))

    brain_data = (
        (mask_img.get_fdata() > 0).astype(np.float32)
        * t1_img.get_fdata().astype(np.float32)
    )

    # Wrap in T1's MGH header so FreeSurfer sees correct vox2ras / TR etc.
    new_brainmask_mgh = nib.freesurfer.MGHImage(
        brain_data,
        affine=t1_img.affine,
        header=t1_img.header,
    )

    new_brainmask_mgh.to_filename(str(brainmask_mgz))
    new_brainmask_mgh.to_filename(str(finalsurfs_mgz))

    print('[inject_brain_mask] Written: {}'.format(brainmask_mgz))
    print('[inject_brain_mask] Written: {}'.format(finalsurfs_mgz))
    return brainmask_mgz


# ---------------------------------------------------------------------------
# Stage 4b – autorecon2
# ---------------------------------------------------------------------------

def run_autorecon2(
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon2.

    FreeSurfer computes wm.mgz, tessellates the surfaces, and places the
    white and pial surfaces.

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
        '-autorecon2',
    ] + (extra_flags or [])
    run_cmd(cmd, tool_name='recon-all autorecon2', timeout=21600)


# ---------------------------------------------------------------------------
# Stage 4c – autorecon3
# ---------------------------------------------------------------------------

def run_autorecon3(
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon3.

    Cortical parcellation, thickness, curvature stats, and all remaining
    FreeSurfer outputs.

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
    run_cmd(cmd, tool_name='recon-all autorecon3', timeout=14400)


# ---------------------------------------------------------------------------
# QC prompts
# ---------------------------------------------------------------------------

def qc_prompt_brainmask(
    subjects_dir: str,
    subject: str,
    skip: bool = False,
) -> None:
    """
    Pause the pipeline for brain mask QC before autorecon2.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    skip         : If True, print instructions but do not wait for input
    """
    if skip:
        print('[QC] --skip-qc-1 set — continuing without waiting.')
        return

    mri_path      = mri_dir(subjects_dir, subject)
    t1_mgz        = mri_path / 'T1.mgz'
    brainmask_mgz = mri_path / 'brainmask.mgz'

    print('\n' + '=' * 70)
    print('QC CHECKPOINT 1 — brain mask (before autorecon2)')
    print('=' * 70)
    print('T1          : {}'.format(t1_mgz))
    print('Brain mask  : {}'.format(brainmask_mgz))
    print()
    print('Check: full cortex coverage, no dura/skull, no holes.')
    print()
    print('Suggested freeview command:')
    print('  freeview {} {}:colormap=heat:opacity=0.4'.format(
        t1_mgz, brainmask_mgz))
    print()
    print('If edits are needed:')
    print('  1. Edit the mask in freeview or ITK-SNAP')
    print('  2. Save as brain-mask-edited.nii.gz')
    print('  3. Re-run with: --brain-mask-edited <path> '
          '--overwrite inject_brainmask autorecon2 autorecon3')
    print('=' * 70)

    launch_freeview(
        str(t1_mgz),
        '{}:colormap=heat:opacity=0.4'.format(brainmask_mgz),
    )

    input('\nPress Enter when satisfied with the brain mask '
          'to continue to autorecon2 ...')


def qc_prompt_surfaces(
    subjects_dir: str,
    subject: str,
    skip: bool = False,
) -> None:
    """
    Pause the pipeline for surface + WM QC before autorecon3.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    skip         : If True, print instructions but do not wait for input
    """
    if skip:
        print('[QC] --skip-qc-2 set — continuing without waiting.')
        return

    mri_path = mri_dir(subjects_dir, subject)
    surf_dir = Path(subjects_dir) / subject / 'surf'
    t1_mgz   = mri_path / 'T1.mgz'
    wm_mgz   = mri_path / 'wm.mgz'
    lh_white = surf_dir / 'lh.white'
    rh_white = surf_dir / 'rh.white'
    lh_pial  = surf_dir / 'lh.pial'
    rh_pial  = surf_dir / 'rh.pial'

    print('\n' + '=' * 70)
    print('QC CHECKPOINT 2 — surfaces + WM mask (before autorecon3)')
    print('=' * 70)
    print('T1       : {}'.format(t1_mgz))
    print('WM mask  : {}'.format(wm_mgz))
    print('Surfaces : lh/rh white + pial in {}'.format(surf_dir))
    print()
    print('Check: white surface at WM/GM boundary, pial at GM/CSF boundary.')
    print('       Pay attention to insula, cingulate, and occipital poles.')
    print()
    print('Suggested freeview command:')
    print('  freeview {} {}:colormap=heat:opacity=0.3 '
          '-f {}:edgecolor=yellow {}:edgecolor=red '
          '{}:edgecolor=yellow {}:edgecolor=red'.format(
              t1_mgz, wm_mgz,
              lh_white, lh_pial,
              rh_white, rh_pial))
    print()
    print('If surface edits are needed:')
    print('  1. Edit wm.mgz directly in freeview (Voxel Edit mode)')
    print('  2. Save, then press Enter — autorecon3 will proceed.')
    print('     To re-run autorecon2 with your edits instead, Ctrl-C here')
    print('     and re-run with: --skip-autorecon1 --skip-qc-1 '
          '--overwrite autorecon2 autorecon3')
    print('=' * 70)

    launch_freeview(str(t1_mgz), str(wm_mgz))

    input('\nPress Enter when satisfied with surfaces and WM mask '
          'to continue to autorecon3 ...')


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_freesurfer_stages(
    uni_mpragised: str,
    brain_mask: str,
    subjects_dir: str,
    subject: str,
    brain_mask_edited: str = None,
    skip_autorecon1: bool = False,
    skip_autorecon2: bool = False,
    skip_autorecon3: bool = False,
    skip_qc_1: bool = False,
    skip_qc_2: bool = False,
    extra_flags: list = None,
    quit_point: str = '',
    overwrite: dict = None,
) -> dict:
    """
    Full pipeline:
        autorecon1 → inject_brainmask → QC1 → autorecon2 → QC2 → autorecon3

    Existence of each stage's sentinel output is checked in the FreeSurfer
    subject directory.  Completed stages are skipped unless overwrite is set.

    Parameters
    ----------
    uni_mpragised     : Full-head MPRAGEised UNI (.nii.gz)
    brain_mask        : nighres brain mask (.nii.gz)
    subjects_dir      : FreeSurfer SUBJECTS_DIR
    subject           : FreeSurfer subject label
    brain_mask_edited : Manually edited brain mask — overrides brain_mask
                        at the injection step if supplied
    skip_autorecon1   : Force-skip autorecon1 (legacy re-entry flag)
    skip_autorecon2   : Force-skip autorecon2 (legacy re-entry flag)
    skip_autorecon3   : Force-skip autorecon3 (legacy re-entry flag)
    skip_qc_1         : Do not pause at brain mask QC checkpoint
    skip_qc_2         : Do not pause at surface/WM QC checkpoint
    extra_flags       : Extra flags forwarded to all recon-all calls
    quit_point        : Stop after a named stage ('autorecon1', 'brainmask')
    overwrite         : dict mapping stage keys → bool.
                        Missing keys default to False.
                        Valid keys: 'autorecon1', 'inject_brainmask',
                        'autorecon2', 'autorecon3'.

    Returns
    -------
    dict mapping output names to their paths
    """
    ow = {k: False for k in STAGE_KEYS}
    if overwrite:
        unknown = set(overwrite) - set(STAGE_KEYS)
        if unknown:
            raise ValueError(
                'Unknown overwrite key(s): {}. Valid keys are: {}'.format(
                    sorted(unknown), STAGE_KEYS)
            )
        ow.update(overwrite)

    os.makedirs(subjects_dir, exist_ok=True)

    subj_dir = Path(subjects_dir) / subject
    mri_path = mri_dir(subjects_dir, subject)
    surf_dir = subj_dir / 'surf'

    # ------------------------------------------------------------------ #
    # Stage 3 – autorecon1                                                #
    # ------------------------------------------------------------------ #
    print('\n[Stage 3] autorecon1 ...')
    if skip_autorecon1:
        print('  [skip] autorecon1 — --skip-autorecon1 flag set.')
    elif check_skip(
        {'T1': mri_path / 'T1.mgz'},
        ow['autorecon1'],
        'Stage 3: autorecon1',
    ):
        pass
    else:
        run_autorecon1(
            uni_mpragised=uni_mpragised,
            subjects_dir=subjects_dir,
            subject=subject,
            extra_flags=extra_flags,
        )
        print('[Stage 3] autorecon1 complete.')

    if 'autorecon1' in quit_point:
        print('Quitting at autorecon1')
        return {}

    # ------------------------------------------------------------------ #
    # Stage 4a – inject brain mask                                        #
    # ------------------------------------------------------------------ #
    print('\n[Stage 4a] Injecting brain mask ...')
    if check_skip(
        {'brainmask': mri_path / 'brainmask.mgz'},
        ow['inject_brainmask'],
        'Stage 4a: inject brain mask',
    ):
        brainmask_mgz = mri_path / 'brainmask.mgz'
    else:
        brainmask_mgz = inject_brain_mask(
            brain_mask=brain_mask,
            subjects_dir=subjects_dir,
            subject=subject,
            brain_mask_edited=brain_mask_edited,
        )
        print('[Stage 4a] Brain mask injected: {}'.format(brainmask_mgz))

    if 'brainmask' in quit_point:
        print('Quitting at brainmask')
        return {}

    # ------------------------------------------------------------------ #
    # QC checkpoint 1 — brainmask                                        #
    # ------------------------------------------------------------------ #
    qc_prompt_brainmask(
        subjects_dir=subjects_dir,
        subject=subject,
        skip=skip_qc_1,
    )

    # ------------------------------------------------------------------ #
    # Stage 4b – autorecon2                                               #
    # ------------------------------------------------------------------ #
    print('\n[Stage 4b] autorecon2 ...')
    if skip_autorecon2:
        print('  [skip] autorecon2 — --skip-autorecon2 flag set.')
    elif check_skip(
        {'wm': mri_path / 'wm.mgz'},
        ow['autorecon2'],
        'Stage 4b: autorecon2',
    ):
        pass
    else:
        run_autorecon2(
            subjects_dir=subjects_dir,
            subject=subject,
            extra_flags=extra_flags,
        )
        print('[Stage 4b] autorecon2 complete.')

    # ------------------------------------------------------------------ #
    # QC checkpoint 2 — surfaces + WM                                    #
    # ------------------------------------------------------------------ #
    qc_prompt_surfaces(
        subjects_dir=subjects_dir,
        subject=subject,
        skip=skip_qc_2,
    )

    # ------------------------------------------------------------------ #
    # Stage 4c – autorecon3                                               #
    # ------------------------------------------------------------------ #
    print('\n[Stage 4c] autorecon3 ...')
    if skip_autorecon3:
        print('  [skip] autorecon3 — --skip-autorecon3 flag set.')
    elif check_skip(
        {'lh_thickness': surf_dir / 'lh.thickness'},
        ow['autorecon3'],
        'Stage 4c: autorecon3',
    ):
        pass
    else:
        run_autorecon3(
            subjects_dir=subjects_dir,
            subject=subject,
            extra_flags=extra_flags,
        )
        print('[Stage 4c] autorecon3 complete.')

    # ------------------------------------------------------------------ #
    # Collect outputs                                                     #
    # ------------------------------------------------------------------ #
    results = {
        'subject_dir':   str(subj_dir),
        'brainmask_mgz': str(brainmask_mgz),
        'wm_mgz':        str(mri_path / 'wm.mgz'),
        'lh_white':      str(surf_dir / 'lh.white'),
        'rh_white':      str(surf_dir / 'rh.white'),
        'lh_pial':       str(surf_dir / 'lh.pial'),
        'rh_pial':       str(surf_dir / 'rh.pial'),
        'lh_thickness':  str(surf_dir / 'lh.thickness'),
        'rh_thickness':  str(surf_dir / 'rh.thickness'),
    }

    print('\n[Done] Key outputs:')
    for k, v in results.items():
        print('  {:20s} {}'.format(k, v))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='FreeSurfer recon stages for MP2RAGE 7T: '
                    'autorecon1 → brainmask inject → autorecon2 → autorecon3.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--uni-mpragised', required=True,
                   help='Full-head MPRAGEised UNI (.nii.gz) — '
                        'brain mask is injected after autorecon1')
    p.add_argument('--brain-mask', required=True,
                   help='nighres brain mask (.nii.gz)')
    p.add_argument('--subjects-dir', required=True,
                   help='FreeSurfer SUBJECTS_DIR')
    p.add_argument('--subject', required=True,
                   help='FreeSurfer subject label (e.g. sub-01_ses-01)')

    p.add_argument('--brain-mask-edited', default=None,
                   help='Manually edited brain mask (.nii.gz) — '
                        'overrides --brain-mask at injection step')

    p.add_argument('--skip-autorecon1', action='store_true',
                   help='Force-skip autorecon1 regardless of overwrite setting')
    p.add_argument('--skip-autorecon2', action='store_true',
                   help='Force-skip autorecon2 regardless of overwrite setting')
    p.add_argument('--skip-autorecon3', action='store_true',
                   help='Force-skip autorecon3 regardless of overwrite setting')
    p.add_argument('--quit-point', default='',
                   help='Stop pipeline after named stage '
                        '(autorecon1 | brainmask)')

    p.add_argument('--skip-qc-1', action='store_true',
                   help='Do not pause at the brainmask QC checkpoint')
    p.add_argument('--skip-qc-2', action='store_true',
                   help='Do not pause at the surface/WM QC checkpoint')

    p.add_argument('--extra-flags', nargs=argparse.REMAINDER, default=[],
                   help='Extra flags passed verbatim to all recon-all calls')

    ow_group = p.add_argument_group(
        'overwrite options',
        'By default, stages whose sentinel outputs already exist are skipped. '
        'Use the flags below to force specific stages to re-run.\n'
        'Valid stage keys: ' + ', '.join(STAGE_KEYS),
    )
    ow_group.add_argument(
        '--overwrite',
        nargs='+',
        metavar='STAGE',
        default=[],
        choices=STAGE_KEYS,
        help='Force re-run for one or more named stages.',
    )
    ow_group.add_argument(
        '--overwrite-all',
        action='store_true',
        default=False,
        help='Force re-run for all stages.',
    )

    return p


def main():
    args = _build_parser().parse_args()

    if args.overwrite_all:
        overwrite = {k: True for k in STAGE_KEYS}
    else:
        overwrite = {k: (k in args.overwrite) for k in STAGE_KEYS}

    run_freesurfer_stages(
        uni_mpragised=args.uni_mpragised,
        brain_mask=args.brain_mask,
        subjects_dir=args.subjects_dir,
        subject=args.subject,
        brain_mask_edited=args.brain_mask_edited,
        skip_autorecon1=args.skip_autorecon1,
        skip_autorecon2=args.skip_autorecon2,
        skip_autorecon3=args.skip_autorecon3,
        skip_qc_1=args.skip_qc_1,
        skip_qc_2=args.skip_qc_2,
        extra_flags=args.extra_flags,
        quit_point=args.quit_point,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()