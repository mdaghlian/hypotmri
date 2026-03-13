#!/usr/bin/env python
"""
run_freesurfer_recon.py
=======================
Run FreeSurfer recon-all stages for the MP2RAGE 7T pipeline.

Stage ordering
--------------
    Stage 3   – recon-all -autorecon1 -noskullstrip
                  Conforms the input, runs Talairach registration.
                  Skips FreeSurfer skull stripping (nighres mask is better).

    Stage 4a  – Inject nighres brain mask → brainmask.mgz
                  Merges nighres mask against brainmask.auto.mgz, multiplied
                  by T1.mgz intensities.  Also writes brain.finalsurfs.manedit.mgz.
                  Backups created before any file is overwritten.

    QC #1     – Inspect brainmask.mgz before surface generation.
                  Pipeline pauses here (pass --skip-qc-1 to bypass).

    Stage 4b  – recon-all -autorecon2
                  FreeSurfer computes its own wm.mgz, runs surface tessellation
                  and places white + pial surfaces.

    Stage 4c  – Inject MGDM WM mask → wm.mgz  (optional)
                  Merges MGDM WM with FreeSurfer's existing wm.mgz using the
                  formula: ((fs_wm + mgdm_wm) > 0) * 255, then takes the
                  largest connected component to remove floating islands.
                  Skipped if --mgdm-seg is not supplied.

    QC #2     – Inspect wm.mgz and white/pial surfaces.
                  Pipeline pauses here (pass --skip-qc-2 to bypass).

    Stage 4d  – recon-all -autorecon2-wm
                  Re-runs from the WM segmentation stage onwards to incorporate
                  the injected WM mask.  Skipped if --mgdm-seg was not supplied.

Why this order?
---------------
WM injection must happen *after* autorecon2 so that FreeSurfer's own wm.mgz
exists to merge against.  Injecting before autorecon2 would write MGDM WM
cold — FreeSurfer would then overwrite it during its own WM segmentation step.

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
    inject_wm         Stage 4c – sentinel: mri/wm_mgdm.mgz
    autorecon2_wm     Stage 4d – sentinel: surf/lh.white + surf/rh.white

The existing --skip-autorecon1 / --skip-autorecon2 / --quit-point flags are
preserved — they serve a different purpose (re-entry / debugging) and are
evaluated after the overwrite check.

Usage examples
--------------
# Minimal — brain mask only, no MGDM WM injection
python run_freesurfer_recon.py \\
    --uni-mpragised-brain /out/sub-01_ses-01_UNI-mpragised-brain.nii.gz \\
    --brain-mask          /out/sub-01_ses-01_brain-mask.nii.gz \\
    --subjects-dir        /out/freesurfer \\
    --subject             sub-01_ses-01

# Full — with manually edited brain mask and MGDM WM injection
python run_freesurfer_recon.py \\
    --uni-mpragised-brain /out/sub-01_ses-01_UNI-mpragised-brain.nii.gz \\
    --brain-mask          /out/sub-01_ses-01_brain-mask.nii.gz \\
    --brain-mask-edited   /out/sub-01_ses-01_brain-mask-edited.nii.gz \\
    --mgdm-seg            /out/sub-01_ses-01_mgdm-seg.nii.gz \\
    --subjects-dir        /out/freesurfer \\
    --subject             sub-01_ses-01

# Re-entry after manual wm.mgz edits — skip to autorecon2-wm,
# overwriting the inject_wm and autorecon2_wm stages
python run_freesurfer_recon.py \\
    --uni-mpragised-brain /out/sub-01_ses-01_UNI-mpragised-brain.nii.gz \\
    --brain-mask          /out/sub-01_ses-01_brain-mask.nii.gz \\
    --mgdm-seg            /out/sub-01_ses-01_mgdm-seg.nii.gz \\
    --subjects-dir        /out/freesurfer \\
    --subject             sub-01_ses-01 \\
    --skip-autorecon1 --skip-autorecon2 --skip-qc-1 \\
    --overwrite inject_wm autorecon2_wm
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image as nli

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
    'inject_wm',
    'autorecon2_wm',
]

# Label value used by nighres MGDM for cerebral white matter.
# Adjust if your atlas uses a different convention.
MGDM_WM_LABEL = 32


# ---------------------------------------------------------------------------
# Stage 3 – autorecon1
# ---------------------------------------------------------------------------

def run_autorecon1(
    uni_mpragised_brain: str,
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon1 -noskullstrip.

    Uses the MPRAGEised skull-stripped UNI image.  FreeSurfer's own skull
    stripping is deliberately skipped — the nighres mask injected in Stage 4a
    is superior for 7T MP2RAGE data.

    Parameters
    ----------
    uni_mpragised_brain : Skull-stripped MPRAGEised UNI (.nii.gz)
    subjects_dir        : FreeSurfer SUBJECTS_DIR
    subject             : FreeSurfer subject label
    extra_flags         : Extra recon-all flags (e.g. ['-parallel'])
    """
    cmd = [
        'recon-all',
        '-i',            uni_mpragised_brain,
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
    3. Compute:  new_brainmask = (nighres_mask > 0) * T1
       This zeros out non-brain voxels while preserving T1 intensities inside
       the mask — the format FreeSurfer expects.
    4. Write brainmask.mgz and brain.finalsurfs.manedit.mgz (the latter is
       checked by FreeSurfer during pial surface refinement).
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

    if not t1_mgz.exists():
        raise FileNotFoundError(
            'T1.mgz not found — has autorecon1 completed?\n'
            '  Expected: {}'.format(t1_mgz)
        )

    if brainmask_mgz.exists():
        backup_file(brainmask_mgz)

    mask_to_use = brain_mask_edited if brain_mask_edited else brain_mask
    print('\n[inject_brain_mask] Using mask: {}'.format(mask_to_use))

    mask_mgh = resample_to_mgh(mask_to_use, t1_mgz)

    nighres_mgz = mri_path / 'brainmask_nighres.mgz'
    mask_mgh.to_filename(str(nighres_mgz))

    new_brainmask = nli.math_img(
        '(nighres_mask > 0) * t1',
        nighres_mask=str(nighres_mgz),
        t1=str(t1_mgz),
    )
    new_brainmask_mgh = nib.freesurfer.MGHImage(
        new_brainmask.get_fdata().astype(np.float32),
        affine=new_brainmask.affine,
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

    FreeSurfer computes its own WM segmentation (wm.mgz), tessellates the
    surfaces, and places the white and pial surfaces.  The resulting wm.mgz
    is then available for merging with the MGDM WM mask in Stage 4c.

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
    print(cmd)
    # run_cmd(cmd, tool_name='recon-all autorecon2', timeout=21600)


# ---------------------------------------------------------------------------
# Stage 4c – inject MGDM WM mask (post-autorecon2)
# ---------------------------------------------------------------------------

def inject_wm_mask(
    mgdm_seg: str,
    subjects_dir: str,
    subject: str,
    wm_label: int = MGDM_WM_LABEL,
) -> Path:
    """
    Merge the MGDM WM mask with FreeSurfer's existing wm.mgz.

    Must be called *after* autorecon2 so that wm.mgz exists to merge against.

    Workflow
    --------
    1. Backup existing wm.mgz.
    2. Extract WM voxels from MGDM segmentation (voxels == wm_label).
    3. Resample MGDM WM mask to wm.mgz space (nearest-neighbour).
    4. Merge:  new_wm = ((fs_wm + mgdm_wm) > 0) * 255
    5. Take the largest connected component to remove floating WM islands.
    6. Write the result as wm.mgz.  Save intermediate mgdm_wm.mgz for audit.

    Parameters
    ----------
    mgdm_seg     : MGDM segmentation image (.nii.gz)
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    wm_label     : Integer WM label in the MGDM segmentation

    Returns
    -------
    Path to the written wm.mgz
    """
    mri_path = mri_dir(subjects_dir, subject)
    wm_mgz   = mri_path / 'wm.mgz'

    if not wm_mgz.exists():
        raise FileNotFoundError(
            'wm.mgz not found — has autorecon2 completed?\n'
            '  Expected: {}'.format(wm_mgz)
        )

    backup_file(wm_mgz)

    seg_img  = nib.load(str(Path(mgdm_seg).resolve()))
    seg_data = np.round(seg_img.get_fdata()).astype(np.int32)
    wm_mask  = (seg_data == wm_label).astype(np.float32)

    if wm_mask.sum() == 0:
        raise ValueError(
            'No voxels with WM label {} found in MGDM segmentation.\n'
            '  Check --mgdm-wm-label.  Segmentation: {}'.format(
                wm_label, mgdm_seg)
        )

    mgdm_wm_nii = nib.Nifti1Image(wm_mask, seg_img.affine, seg_img.header)

    mgdm_wm_mgh = resample_to_mgh(mgdm_wm_nii, wm_mgz)
    mgdm_wm_mgz = mri_path / 'wm_mgdm.mgz'
    mgdm_wm_mgh.to_filename(str(mgdm_wm_mgz))

    merged = nli.math_img(
        '((fs_wm + mgdm_wm) > 0) * 255',
        fs_wm=str(wm_mgz),
        mgdm_wm=str(mgdm_wm_mgz),
    )

    merged_lcc   = nli.largest_connected_component_img(merged)
    merged_final = nli.math_img('img * 255', img=merged_lcc)

    wm_mgh = nib.freesurfer.MGHImage(
        merged_final.get_fdata().astype(np.float32),
        affine=merged_final.affine,
    )
    wm_mgh.to_filename(str(wm_mgz))

    print('[inject_wm_mask] Written: {}'.format(wm_mgz))
    print('[inject_wm_mask] Written: {}'.format(mgdm_wm_mgz))
    return wm_mgz


# ---------------------------------------------------------------------------
# Stage 4d – autorecon2-wm
# ---------------------------------------------------------------------------

def run_autorecon2_wm(
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon2-wm.

    Re-runs FreeSurfer from the WM segmentation stage onwards to incorporate
    the injected wm.mgz from Stage 4c.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    extra_flags  : Extra recon-all flags (e.g. ['-parallel'])
    """
    cmd = [
        'recon-all',
        '-s',             subject,
        '-sd',            subjects_dir,
        '-autorecon2-wm',
    ] + (extra_flags or [])

    run_cmd(cmd, tool_name='recon-all autorecon2-wm', timeout=21600)


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
          '--overwrite inject_brainmask autorecon2 inject_wm autorecon2_wm')
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
    Pause the pipeline for surface + WM QC before autorecon2-wm.

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
    print('QC CHECKPOINT 2 — surfaces + WM mask (before autorecon2-wm)')
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
    print('If WM edits are needed:')
    print('  1. Edit wm.mgz directly in freeview (Voxel Edit mode)')
    print('  2. Save, then press Enter — autorecon2-wm will re-run.')
    print('=' * 70)

    launch_freeview(str(t1_mgz), str(wm_mgz))

    input('\nPress Enter when satisfied with surfaces and WM mask '
          'to continue to autorecon2-wm ...')


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_freesurfer_stages(
    uni_mpragised_brain: str,
    brain_mask: str,
    subjects_dir: str,
    subject: str,
    brain_mask_edited: str = None,
    mgdm_seg: str = None,
    mgdm_wm_label: int = MGDM_WM_LABEL,
    skip_autorecon1: bool = False,
    skip_autorecon2: bool = False,
    skip_qc_1: bool = False,
    skip_qc_2: bool = False,
    extra_flags: list = None,
    quit_point: str = '',
    overwrite: dict = None,
) -> dict:
    """
    Full pipeline:
        autorecon1 → brainmask inject → QC1 → autorecon2
        → WM inject → QC2 → autorecon2-wm

    Existence of each stage's sentinel output is checked in the FreeSurfer
    subject directory.  Completed stages are skipped unless overwrite is set.

    Parameters
    ----------
    uni_mpragised_brain : Skull-stripped MPRAGEised UNI (.nii.gz)
    brain_mask          : nighres brain mask (.nii.gz)
    subjects_dir        : FreeSurfer SUBJECTS_DIR
    subject             : FreeSurfer subject label
    brain_mask_edited   : Manually edited brain mask (overrides brain_mask)
    mgdm_seg            : MGDM segmentation for WM injection (optional)
    mgdm_wm_label       : WM label integer in the MGDM segmentation
    skip_autorecon1     : Skip autorecon1 regardless of overwrite setting
                          (legacy re-entry flag — takes precedence)
    skip_autorecon2     : Skip autorecon2 regardless of overwrite setting
                          (legacy re-entry flag — takes precedence)
    skip_qc_1           : Do not pause at brainmask QC checkpoint
    skip_qc_2           : Do not pause at surface/WM QC checkpoint
    extra_flags         : Extra flags forwarded to all recon-all calls
    quit_point          : Stop after a named stage ('autorecon1', 'brainmask')
    overwrite           : dict mapping stage keys to booleans.
                          Missing keys default to False (don't overwrite).
                          Valid keys: 'autorecon1', 'inject_brainmask',
                          'autorecon2', 'inject_wm', 'autorecon2_wm'.

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
            uni_mpragised_brain=uni_mpragised_brain,
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
    # Stage 4c – inject MGDM WM mask                                     #
    # ------------------------------------------------------------------ #
    wm_mgz = mri_path / 'wm.mgz'
    if mgdm_seg:
        print('\n[Stage 4c] Merging MGDM WM mask into wm.mgz ...')
        if check_skip(
            {'wm_mgdm': mri_path / 'wm_mgdm.mgz'},
            ow['inject_wm'],
            'Stage 4c: inject WM mask',
        ):
            wm_mgz = mri_path / 'wm.mgz'
        else:
            wm_mgz = inject_wm_mask(
                mgdm_seg=mgdm_seg,
                subjects_dir=subjects_dir,
                subject=subject,
                wm_label=mgdm_wm_label,
            )
            print('[Stage 4c] WM mask injected: {}'.format(wm_mgz))

        # -------------------------------------------------------------- #
        # QC checkpoint 2 — surfaces + WM                                #
        # -------------------------------------------------------------- #
        qc_prompt_surfaces(
            subjects_dir=subjects_dir,
            subject=subject,
            skip=skip_qc_2,
        )

        # -------------------------------------------------------------- #
        # Stage 4d – autorecon2-wm                                       #
        # -------------------------------------------------------------- #
        print('\n[Stage 4d] autorecon2-wm ...')
        print(surf_dir / 'lh.white')
        if check_skip(
            {'lh_white_preaparc': surf_dir / 'lh.white.preaparc',
             'rh_white_preaparc': surf_dir / 'rh.white.preaparc'},
            ow['autorecon2_wm'],
            'Stage 4d: autorecon2-wm',
        ):
            pass
        else:
            bloop
            run_autorecon2_wm(
                subjects_dir=subjects_dir,
                subject=subject,
                extra_flags=extra_flags,
            )
            print('[Stage 4d] autorecon2-wm complete.')

    else:
        print('\n[Stage 4c] No --mgdm-seg supplied — '
              'skipping WM injection and autorecon2-wm.')

    # ------------------------------------------------------------------ #
    # Collect outputs                                                     #
    # ------------------------------------------------------------------ #
    results = {
        'subject_dir':   str(subj_dir),
        'brainmask_mgz': str(brainmask_mgz),
        'wm_mgz':        str(wm_mgz),
        # 'lh_white':      str(surf_dir / 'lh.white'),
        # 'rh_white':      str(surf_dir / 'rh.white'),
        # 'lh_pial':       str(surf_dir / 'lh.pial'),
        # 'rh_pial':       str(surf_dir / 'rh.pial'),
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
                    'autorecon1 → brainmask inject → autorecon2 '
                    '→ WM inject → autorecon2-wm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--uni-mpragised-brain', required=True,
                   help='Skull-stripped MPRAGEised UNI (.nii.gz)')
    p.add_argument('--brain-mask', required=True,
                   help='nighres brain mask (.nii.gz)')
    p.add_argument('--subjects-dir', required=True,
                   help='FreeSurfer SUBJECTS_DIR')
    p.add_argument('--subject', required=True,
                   help='FreeSurfer subject label (e.g. sub-01_ses-01)')

    p.add_argument('--brain-mask-edited', default=None,
                   help='Manually edited brain mask (.nii.gz) — '
                        'overrides --brain-mask at injection step')
    p.add_argument('--mgdm-seg', default=None,
                   help='MGDM segmentation (.nii.gz) for WM injection; '
                        'if omitted, stages 4c and 4d are skipped')
    p.add_argument('--mgdm-wm-label', type=int, default=MGDM_WM_LABEL,
                   help='WM label integer in the MGDM segmentation')

    p.add_argument('--skip-autorecon1', action='store_true',
                   help='Force-skip autorecon1 regardless of overwrite setting')
    p.add_argument('--skip-autorecon2', action='store_true',
                   help='Force-skip autorecon2 regardless of overwrite setting')
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
        uni_mpragised_brain=args.uni_mpragised_brain,
        brain_mask=args.brain_mask,
        subjects_dir=args.subjects_dir,
        subject=args.subject,
        brain_mask_edited=args.brain_mask_edited,
        mgdm_seg=args.mgdm_seg,
        mgdm_wm_label=args.mgdm_wm_label,
        skip_autorecon1=args.skip_autorecon1,
        skip_autorecon2=args.skip_autorecon2,
        skip_qc_1=args.skip_qc_1,
        skip_qc_2=args.skip_qc_2,
        extra_flags=args.extra_flags,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()