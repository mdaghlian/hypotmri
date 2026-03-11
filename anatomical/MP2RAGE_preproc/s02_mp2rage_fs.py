#!/usr/bin/env python
"""
run_freesurfer_recon.py
=======================
Run FreeSurfer recon-all stages 3 and 4 of the MP2RAGE pipeline:

    Stage 3  – recon-all -autorecon1 -noskullstrip
                 Conforms the input image, runs Talairach registration,
                 but skips FreeSurfer's own skull stripping.

    Stage 4  – Inject brain mask (nighres or manually edited) into the
                 FreeSurfer subject directory as brainmask.mgz, then
                 optionally inject an MGDM-derived WM mask as wm.mgz.

    Stage 4b – recon-all -autorecon2
                 White matter segmentation, surface tessellation and
                 placement (white + pial surfaces).

Inputs are the outputs of run_mp2rage_preproc.py.  The script pauses at
the QC checkpoint between mask injection and autorecon2 so the user can
inspect and/or edit brainmask.mgz before surface generation begins.

Usage examples
--------------
# Minimal — no T1map, no MGDM WM injection, no manual mask override
python run_freesurfer_recon.py \\
    --uni-mpragised-brain /out/sub-01_ses-01_UNI-mpragised-brain.nii.gz \\
    --brain-mask          /out/sub-01_ses-01_brain-mask.nii.gz \\
    --subjects-dir        /out/freesurfer \\
    --subject             sub-01_ses-01

# Full — with edited mask and MGDM WM injection
python run_freesurfer_recon.py \\
    --uni-mpragised-brain /out/sub-01_ses-01_UNI-mpragised-brain.nii.gz \\
    --brain-mask          /out/sub-01_ses-01_brain-mask.nii.gz \\
    --brain-mask-edited   /out/sub-01_ses-01_brain-mask-edited.nii.gz \\
    --mgdm-seg            /out/sub-01_ses-01_mgdm-seg.nii.gz \\
    --subjects-dir        /out/freesurfer \\
    --subject             sub-01_ses-01 \\
    --skip-qc-prompt
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers (mirrors preproc_utils.py style)
# ---------------------------------------------------------------------------

def check_result(result, tool_name: str) -> None:
    """Raise RuntimeError with full stdout/stderr if a subprocess failed."""
    if result.returncode != 0:
        raise RuntimeError(
            '{} failed (exit {}).\n'
            '--- stdout ---\n{}\n'
            '--- stderr ---\n{}'.format(
                tool_name, result.returncode, result.stdout, result.stderr)
        )


def run_cmd(cmd: list, tool_name: str, env: dict = None, timeout: int = None) -> None:
    """
    Run a subprocess, stream its output, and raise on failure.

    Parameters
    ----------
    cmd       : Command and arguments as a list of strings
    tool_name : Label used in error messages
    env       : Optional environment dict (merged with os.environ)
    timeout   : Optional timeout in seconds
    """
    merged_env = {**os.environ, **(env or {})}
    print('[{}] Running: {}'.format(tool_name, ' '.join(cmd)))

    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge so ordering is preserved
        text=True,
        env=merged_env,
        timeout=timeout,
    )

    # Stream captured output so the user can follow progress
    if result.stdout:
        for line in result.stdout.splitlines():
            print('[{}] {}'.format(tool_name, line))

    check_result(result, tool_name)


def mri_dir(subjects_dir: str, subject: str) -> Path:
    """Return the mri/ subdirectory for this subject."""
    return Path(subjects_dir) / subject / 'mri'


# ---------------------------------------------------------------------------
# Stage 3 – autorecon1 (no skull strip)
# ---------------------------------------------------------------------------

def run_autorecon1(
    uni_mpragised_brain: str,
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon1 -noskullstrip.

    Uses the MPRAGEised skull-stripped UNI as the input image.  Skipping
    FreeSurfer's skull stripping here is intentional — the nighres mask
    (injected in Stage 4) is superior for 7T data.

    Parameters
    ----------
    uni_mpragised_brain : Path to skull-stripped MPRAGEised UNI (.nii.gz)
    subjects_dir        : FreeSurfer SUBJECTS_DIR
    subject             : FreeSurfer subject label
    extra_flags         : Any additional recon-all flags (e.g. ['-parallel'])
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
    Place the nighres brain mask (or a manually edited version) into the
    FreeSurfer subject directory as brainmask.mgz.

    The mask is:
      1. Converted to MGZ via mri_convert
      2. Applied to T1.mgz via mri_mask to produce the final brainmask.mgz

    Parameters
    ----------
    brain_mask        : nighres brain mask (.nii.gz)
    subjects_dir      : FreeSurfer SUBJECTS_DIR
    subject           : FreeSurfer subject label
    brain_mask_edited : Manually edited mask (overrides brain_mask if supplied)

    Returns
    -------
    Path to the written brainmask.mgz
    """
    mri_path = mri_dir(subjects_dir, subject)
    t1_mgz   = mri_path / 'T1.mgz'

    if not t1_mgz.exists():
        raise FileNotFoundError(
            'T1.mgz not found — has autorecon1 been run?\n'
            '  Expected: {}'.format(t1_mgz)
        )

    # Decide which mask to use
    mask_to_use = brain_mask_edited if brain_mask_edited else brain_mask
    mask_path   = Path(mask_to_use).resolve()
    print('\n[inject_brain_mask] Using mask: {}'.format(mask_path))

    # -- Convert mask to MGZ in FreeSurfer conformed space --
    mask_mgz = mri_path / 'brainmask_nighres.mgz'
    run_cmd(
        ['mri_convert', str(mask_path), str(mask_mgz)],
        tool_name='mri_convert (mask)',
    )

    # -- Apply mask to T1 → brainmask.mgz --
    brainmask_mgz = mri_path / 'brainmask.mgz'
    run_cmd(
        ['mri_mask', str(t1_mgz), str(mask_mgz), str(brainmask_mgz)],
        tool_name='mri_mask',
    )

    print('[inject_brain_mask] Written: {}'.format(brainmask_mgz))
    return brainmask_mgz


# ---------------------------------------------------------------------------
# Stage 4b – optional MGDM WM mask injection
# ---------------------------------------------------------------------------

# Label value used by nighres MGDM for cerebral white matter.
# Adjust if your atlas uses a different convention.
MGDM_WM_LABEL = 32   # WM label in the nighres MGDM segmentation


def inject_wm_mask(
    mgdm_seg: str,
    subjects_dir: str,
    subject: str,
    wm_label: int = MGDM_WM_LABEL,
) -> Path:
    mri_path = mri_dir(subjects_dir, subject)
    t1_mgz   = mri_path / 'T1.mgz'          # <-- was norm.mgz

    if not t1_mgz.exists():
        raise FileNotFoundError(
            'T1.mgz not found — has autorecon1 completed?\n'
            '  Expected: {}'.format(t1_mgz)
        )

    seg_img  = nib.load(str(Path(mgdm_seg).resolve()))
    seg_data = np.round(seg_img.get_fdata()).astype(np.int32)

    wm_mask = (seg_data == wm_label).astype(np.uint8)
    if wm_mask.sum() == 0:
        raise ValueError(
            'No voxels found with WM label {} in MGDM segmentation.\n'
            '  Check --mgdm-wm-label and verify the segmentation: {}'.format(
                wm_label, mgdm_seg)
        )

    wm_fs  = (wm_mask * 110).astype(np.uint8)
    wm_nii = mri_path / 'wm_nighres.nii.gz'
    wm_mgz = mri_path / 'wm.mgz'

    nib.save(nib.Nifti1Image(wm_fs, seg_img.affine, seg_img.header), str(wm_nii))

    run_cmd(
        ['mri_convert', '--resample_type', 'nearest',
         '--reslice_like', str(t1_mgz),           # <-- was norm_mgz
         str(wm_nii), str(wm_mgz)],
        tool_name='mri_convert (wm)',
    )

    print('[inject_wm_mask] Written: {}'.format(wm_mgz))
    return wm_mgz

# ---------------------------------------------------------------------------
# QC prompt
# ---------------------------------------------------------------------------

def qc_prompt(subjects_dir: str, subject: str, skip: bool = False) -> None:
    """
    Pause pipeline and print QC instructions.

    Opens freeview with the T1 and brainmask overlaid if freeview is on PATH.
    The user must press Enter to continue (or pass --skip-qc-prompt to bypass).

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    skip         : If True, print instructions but do not wait for input
    """
    mri_path      = mri_dir(subjects_dir, subject)
    t1_mgz        = mri_path / 'T1.mgz'
    brainmask_mgz = mri_path / 'brainmask.mgz'

    print('\n' + '=' * 70)
    print('QC CHECKPOINT — inspect brain mask before surface generation')
    print('=' * 70)
    print('T1       : {}'.format(t1_mgz))
    print('Brainmask: {}'.format(brainmask_mgz))
    print()
    print('Suggested freeview command:')
    print('  freeview {} {}:colormap=heat:opacity=0.4'.format(
        t1_mgz, brainmask_mgz))
    print()
    print('If edits are needed:')
    print('  1. Edit the mask in freeview or ITK-SNAP')
    print('  2. Save as brainmask-edited.nii.gz')
    print('  3. Re-run this script with --brain-mask-edited <path>')
    print('     (autorecon1 will be skipped if the subject dir already exists)')
    print('=' * 70)

    # Attempt to launch freeview non-blocking
    if shutil.which('freeview'):
        try:
            subprocess.Popen([
                'freeview',
                str(t1_mgz),
                '{}:colormap=heat:opacity=0.4'.format(brainmask_mgz),
            ])
            print('[QC] freeview launched in background.')
        except Exception as exc:
            print('[QC] Could not launch freeview automatically: {}'.format(exc))
    else:
        print('[QC] freeview not found on PATH — open the files manually.')

    if not skip:
        input('\nPress Enter when you are satisfied with the brain mask '
              'and ready to continue to autorecon2 ...')
    else:
        print('[QC] --skip-qc-prompt set — continuing without waiting.')


# ---------------------------------------------------------------------------
# Stage 4c – autorecon2
# ---------------------------------------------------------------------------

def run_autorecon2(
    subjects_dir: str,
    subject: str,
    extra_flags: list = None,
) -> None:
    """
    Run recon-all -autorecon2.

    Performs intensity normalisation, WM segmentation, surface tessellation,
    and white/pial surface placement.

    Parameters
    ----------
    subjects_dir : FreeSurfer SUBJECTS_DIR
    subject      : FreeSurfer subject label
    extra_flags  : Any additional recon-all flags (e.g. ['-parallel'])
    """
    cmd = [
        'recon-all',
        '-s',          subject,
        '-sd',         subjects_dir,
        '-autorecon2',
    ] + (extra_flags or [])

    run_cmd(cmd, tool_name='recon-all autorecon2', timeout=21600)


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
    skip_qc_prompt: bool = False,
    extra_flags: list = None,
) -> dict:
    """
    Run Stage 3 (autorecon1) → mask injection → QC → Stage 4 (autorecon2).

    Parameters
    ----------
    uni_mpragised_brain : Skull-stripped MPRAGEised UNI (.nii.gz)
    brain_mask          : nighres brain mask (.nii.gz)
    subjects_dir        : FreeSurfer SUBJECTS_DIR
    subject             : FreeSurfer subject label
    brain_mask_edited   : Manually edited brain mask (overrides brain_mask)
    mgdm_seg            : MGDM segmentation for WM injection (optional)
    mgdm_wm_label       : WM label integer in the MGDM segmentation
    skip_autorecon1     : Skip autorecon1 if subject dir already exists
    skip_qc_prompt      : Do not pause at the QC checkpoint
    extra_flags         : Extra flags forwarded to recon-all

    Returns
    -------
    dict with paths to key FreeSurfer outputs
    """
    os.makedirs(subjects_dir, exist_ok=True)

    subj_dir = Path(subjects_dir) / subject
    mri_path = mri_dir(subjects_dir, subject)

    # ------------------------------------------------------------------ #
    # Stage 3 – autorecon1                                                #
    # ------------------------------------------------------------------ #
    if skip_autorecon1 and (mri_path / 'T1.mgz').exists():
        print('\n[Stage 3] Skipping autorecon1 — T1.mgz already exists.')
    else:
        print('\n[Stage 3] recon-all -autorecon1 -noskullstrip ...')
        run_autorecon1(
            uni_mpragised_brain=uni_mpragised_brain,
            subjects_dir=subjects_dir,
            subject=subject,
            extra_flags=extra_flags,
        )
        print('[Stage 3] autorecon1 complete.')

    # ------------------------------------------------------------------ #
    # Stage 4a – inject brain mask                                        #
    # ------------------------------------------------------------------ #
    print('\n[Stage 4a] Injecting brain mask ...')
    brainmask_mgz = inject_brain_mask(
        brain_mask=brain_mask,
        subjects_dir=subjects_dir,
        subject=subject,
        brain_mask_edited=brain_mask_edited,
    )
    print('[Stage 4a] Brain mask injected: {}'.format(brainmask_mgz))

    # ------------------------------------------------------------------ #
    # Stage 4b – optional MGDM WM injection                              #
    # ------------------------------------------------------------------ #
    wm_mgz = None
    if mgdm_seg:
        print('\n[Stage 4b] Injecting MGDM WM mask ...')
        wm_mgz = inject_wm_mask(
            mgdm_seg=mgdm_seg,
            subjects_dir=subjects_dir,
            subject=subject,
            wm_label=mgdm_wm_label,
        )
        print('[Stage 4b] WM mask injected: {}'.format(wm_mgz))
    else:
        print('\n[Stage 4b] No MGDM segmentation provided — '
              'FreeSurfer will compute its own WM segmentation.')

    # ------------------------------------------------------------------ #
    # QC checkpoint                                                       #
    # ------------------------------------------------------------------ #
    qc_prompt(subjects_dir=subjects_dir, subject=subject, skip=skip_qc_prompt)

    # ------------------------------------------------------------------ #
    # Stage 4c – autorecon2                                               #
    # ------------------------------------------------------------------ #
    print('\n[Stage 4c] recon-all -autorecon2 ...')
    run_autorecon2(
        subjects_dir=subjects_dir,
        subject=subject,
        extra_flags=extra_flags,
    )
    print('[Stage 4c] autorecon2 complete.')

    # ------------------------------------------------------------------ #
    # Collect key outputs                                                 #
    # ------------------------------------------------------------------ #
    surf_dir = subj_dir / 'surf'
    results = {
        'subjects_dir':   str(subj_dir),
        'brainmask_mgz':  str(brainmask_mgz),
        'wm_mgz':         str(wm_mgz) if wm_mgz else None,
        'lh_white':       str(surf_dir / 'lh.white'),
        'rh_white':       str(surf_dir / 'rh.white'),
        'lh_pial':        str(surf_dir / 'lh.pial'),
        'rh_pial':        str(surf_dir / 'rh.pial'),
    }

    print('\n[Done] Outputs:')
    for k, v in results.items():
        if v:
            print('  {:20s} {}'.format(k, v))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='FreeSurfer autorecon1 + mask injection + autorecon2 '
                    'for MP2RAGE 7T data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs
    p.add_argument(
        '--uni-mpragised-brain', required=True,
        help='Skull-stripped MPRAGEised UNI image (.nii.gz) — '
             'output of run_mp2rage_preproc.py',
    )
    p.add_argument(
        '--brain-mask', required=True,
        help='nighres brain mask (.nii.gz) — '
             'output of run_mp2rage_preproc.py',
    )

    # FreeSurfer subject
    p.add_argument('--subjects-dir', required=True,
                   help='FreeSurfer SUBJECTS_DIR')
    p.add_argument('--subject', required=True,
                   help='FreeSurfer subject label (e.g. sub-01_ses-01)')

    # Optional overrides / injections
    p.add_argument(
        '--brain-mask-edited', default=None,
        help='Manually edited brain mask (.nii.gz) — overrides --brain-mask '
             'at the injection step if supplied',
    )
    p.add_argument(
        '--mgdm-seg', default=None,
        help='MGDM segmentation (.nii.gz) for WM mask injection (optional)',
    )
    p.add_argument(
        '--mgdm-wm-label', type=int, default=MGDM_WM_LABEL,
        help='Integer WM label in the MGDM segmentation',
    )

    # Behaviour flags
    p.add_argument(
        '--skip-autorecon1', action='store_true',
        help='Skip autorecon1 if T1.mgz already exists in the subject dir',
    )
    p.add_argument(
        '--skip-qc-prompt', action='store_true',
        help='Do not pause at the QC checkpoint — run autorecon2 immediately',
    )
    p.add_argument(
        '--extra-flags', nargs=argparse.REMAINDER, default=[],
        help='Extra flags passed verbatim to recon-all (e.g. -- -parallel)',
    )

    return p


def main():
    args = _build_parser().parse_args()
    run_freesurfer_stages(
        uni_mpragised_brain=args.uni_mpragised_brain,
        brain_mask=args.brain_mask,
        subjects_dir=args.subjects_dir,
        subject=args.subject,
        brain_mask_edited=args.brain_mask_edited,
        mgdm_seg=args.mgdm_seg,
        mgdm_wm_label=args.mgdm_wm_label,
        skip_autorecon1=args.skip_autorecon1,
        skip_qc_prompt=args.skip_qc_prompt,
        extra_flags=args.extra_flags,
    )


if __name__ == '__main__':
    main()