#!/usr/bin/env python
"""
s02_coreg.py
===============
Motion correction, coregistration to FreeSurfer T1, and surface projection
for BOLD runs that have already been SDC-corrected.

Coregistration strategy
-----------------------
(1) Build BREF_MAIN (one per subject, stored above session level)
(2) Coregister BREF_MAIN to anatomy with bbregister
(3) Coregister each sbref_i to BREF_MAIN   (FLIRT, normcorr, DOF 6)
(4) MCFLIRT per run, referencing the corresponding sbref_i
(5) Concatenate transforms:  VOL -> sbref_i -> BREF_MAIN -> FS_T1

Run selection
-------------
Specify which functional runs to process with one of:

  --include-files  FILE [FILE ...]   exact basenames to find under <input>/sub/
  --include-patterns PAT [PAT ...]   glob patterns relative to <input>/sub/
  --exclude-patterns PAT [PAT ...]   basename patterns (fnmatch) to exclude

If none are given, all *bold*.nii* in <input>/sub/ses/ are used
(the --ses label controls which session directory is searched).
Session labels are always extracted from each bold filename automatically
so outputs are placed in the correct session sub-folder.

BREF_MAIN selection
---------------------
One BREF_MAIN is built per subject and stored at <output>/sub/
(above the session level) so it is shared across sessions.

  --bref-main NAME_OR_PATH
      NAME  — basename found recursively under <input>/sub/
      PATH  — absolute or relative path to an existing file
      (omit) — auto: first sbref under <input>/sub/, or vol-0 of first bold

A provenance note is written to <output>/sub/bref_main_notes.txt.

Overwrite behaviour
-------------------
Existence is checked against final BIDS-named files in *output_dir*.
Skipped steps restore outputs to *work_dir* so downstream steps can proceed.

Usage example
-------------
python s02_coreg.py \\
    --bids-dir      /data \\
    --input-file    s1_sdc_AFNI \\
    --output-file   s2_coreg \\
    --sub           sub-01 \\
    --subjects-dir  /data/freesurfer \\
    --docker        freesurfer/freesurfer:7.4.1 \\
    --include-patterns 'ses-01/*_task-pRF*bold*.nii.gz' \\
    --bref-main   sub-01_ses-01_task-pRF_run-01_sbref.nii.gz
"""

import argparse
import fnmatch
import glob
import os
opj = os.path.join
import re
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from cvl_utils.preproc_func import (
    build_output_name,
    check_skip,
    run_cmd,
    run_local,
    get_labels,
    fsl_val,
    make_safe_workdir,
    _stage,
    _container_path,
    _strip_extensions,
    _bold_base
)

# ---------------------------------------------------------------------------
# Step keys
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'bref_main',
    'bbregister',
    'sbref_to_main',   # per-run sbref_i -> BREF_MAIN
    'mcflirt',
    'concat_xfm',
    'applywarp',
    'surf_project',
]

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _extract_session(filepath: str):
    """Extract ses-XX from a filepath basename, or return None."""
    m = re.search(r'(ses-[^_]+)', os.path.basename(filepath))
    return m.group(1) if m else None


def _find_bold_files(
    subject_input_dir: str,
    subject: str,
    session: str,
    include_files: list,
    include_patterns: list,
    exclude_patterns: list,
) -> list:
    """
    Return sorted list of BOLD file paths to process.

    Priority:
      1. include_files     — exact basenames, found recursively under subject_input_dir
      2. include_patterns  — glob patterns relative to subject_input_dir
      3. fallback          — all *bold*.nii* in subject_input_dir/session/
    Exclusions via exclude_patterns are applied as fnmatch against basenames.
    """
    if include_files:
        found = []
        for fname in include_files:
            matches = glob.glob(opj(subject_input_dir, '**', fname), recursive=True)
            if not matches:
                raise FileNotFoundError(
                    'include-files: cannot find "{}" under {}'.format(
                        fname, subject_input_dir))
            found.extend(matches)
        bold_files = sorted(set(found))

    elif include_patterns:
        found = []
        print(include_patterns)
        print(os.listdir(subject_input_dir))
        for pat in include_patterns:
            found.extend(glob.glob(opj(subject_input_dir, pat)))
        bold_files = sorted(set(found))

    else:
        bold_pattern = opj(
            subject_input_dir, session,
            '{}_{}*bold*.nii*'.format(subject, session))
        bold_files = sorted(glob.glob(bold_pattern))

    if not bold_files:
        raise FileNotFoundError(
            'No BOLD files found. Check --include-files / --include-patterns, '
            'or that <input>/{}/{} exists and contains bold files.'.format(
                subject, session))

    if exclude_patterns:
        def _excluded(f):
            bn = os.path.basename(f)
            return any(fnmatch.fnmatch(bn, pat) for pat in exclude_patterns)
        bold_files = [f for f in bold_files if not _excluded(f)]
        if not bold_files:
            raise FileNotFoundError(
                'All BOLD files were excluded by --exclude-patterns.')

    return bold_files


def _find_sbref_for_bold(
    bold_file: str,
    subject: str,
    session: str,
    subject_output_dir: str,
) -> tuple:
    """
    Find the run-matched sbref in the same directory as bold_file.
    Falls back to a synthetic sbref (vol-0 of the BOLD run) if none found.

    Returns (sbref_path, source_description).
    """
    run_label, task_label = get_labels(bold_file)
    bold_dir = os.path.dirname(bold_file)

    if run_label:
        sbref_pat = opj(
            bold_dir,
            '{}_{}_{}_{}*sbref*.nii*'.format(subject, session, task_label, run_label))
        sbref_matches = sorted(glob.glob(sbref_pat))
    else:
        all_sbrefs = glob.glob(opj(
            bold_dir,
            '{}_{}_{}_*sbref*.nii*'.format(subject, session, task_label)))
        sbref_matches = [f for f in all_sbrefs
                         if not re.search(r'run-\d+', os.path.basename(f))]

    if sbref_matches:
        return sbref_matches[0], 'input'

    # Synthetic fallback: vol-0 of the BOLD run
    parts = [p for p in [subject, session, task_label, run_label] if p]
    parts.append('desc-vol0bold_sbref')
    synthetic_path = opj(subject_output_dir, '_'.join(parts) + '.nii.gz')

    if not Path(synthetic_path).exists():
        print('  No SBREF found — extracting vol 0 as synthetic sbref...')
        img  = nib.load(bold_file)
        data = img.get_fdata(dtype=np.float32)
        vol  = data[..., 0] if data.ndim == 4 else data
        out  = nib.Nifti1Image(vol, img.affine, img.header)
        out.set_data_dtype(np.float32)
        nib.save(out, synthetic_path)
    else:
        print('  No SBREF found — reusing existing synthetic sbref.')

    return synthetic_path, 'vol-0 (synthetic)'


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def make_bref_main(
    bref_spec,
    search_dir: str,
    subject: str,
    subject_main_dir: str,
    note_file: str,
    work_dir: str,
    docker_image: str,
) -> str:
    """
    Build BREF_MAIN and store it at subject_main_dir (above session level).

    bref_spec:
      None                 — auto: first sbref under search_dir; else vol-0 of first bold
      'some_name.nii.gz'   — find this basename recursively under search_dir
      '/absolute/path/...' — use this exact file

    Writes a provenance note to note_file.
    Returns the host path to BREF_MAIN.nii.gz.
    """
    out_path = build_output_name(subject_main_dir, subject, None, 'BREF_MAIN')
    src = None

    if os.path.exists(out_path) & (bref_spec is not None):
        print('!!!!! Warning !!!!!')
        print('YOU HAVE SPECIFIED A BREF SPEC & THERE ALREADY EXISTS A BREF')

    if bref_spec is None:
        sbrefs = sorted(glob.glob(
            opj(search_dir, '**', '*sbref*.nii*'), recursive=True))
        if sbrefs:
            src  = sbrefs[0]
            note = 'AUTO-DETECT (first sbref): {}'.format(src)
        else:
            bolds = sorted(glob.glob(
                opj(search_dir, '**', '*bold*.nii*'), recursive=True))
            if not bolds:
                raise FileNotFoundError(
                    'No sbref or bold files found under {}'.format(search_dir))
            img  = nib.load(bolds[0])
            data = img.get_fdata(dtype=np.float32)
            vol  = data[..., 0] if data.ndim == 4 else data
            out  = nib.Nifti1Image(vol, img.affine, img.header)
            out.set_data_dtype(np.float32)
            nib.save(out, out_path)
            note = 'AUTO-DETECT (vol-0 of {}): no sbref found'.format(bolds[0])
            src  = None  # already written to out_path

    elif os.path.isabs(bref_spec) or (os.sep in bref_spec):
        if not Path(bref_spec).exists():
            raise FileNotFoundError(
                '--bref-main path not found: {}'.format(bref_spec))
        src  = bref_spec
        note = 'EXPLICIT PATH: {}'.format(src)

    else:
        # Treat as a filename to find under search_dir
        matches = sorted(glob.glob(
            opj(search_dir, '**', bref_spec), recursive=True))
        if not matches:
            raise FileNotFoundError(
                '--bref-main "{}": not found under {}'.format(
                    bref_spec, search_dir))
        src  = matches[0]
        note = 'NAMED FILE ({}): found at {}'.format(bref_spec, src)

    if src is not None:
        if src.endswith('.nii'):
            nii_dst = out_path.replace('.gz', '')
            shutil.copy(src, nii_dst)
            subprocess.run(['gzip', nii_dst], check=True)
        else:
            shutil.copy(src, out_path)

    with open(note_file, 'a') as fh:
        fh.write('BREF_MAIN source : {}\n'.format(note))
        fh.write('BREF_MAIN output : {}\n'.format(out_path))
        fh.write('{}\n'.format('-' * 60))

    print('  Source: {}'.format(note))

    _stage(out_path, work_dir)
    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=['fslreorient2std',
             _container_path(work_dir, os.path.basename(out_path), docker_image)],
    )
    shutil.copy(opj(work_dir, os.path.basename(out_path)), out_path)

    return out_path


def convert_fs_t1(
    subjects_dir: str,
    subject: str,
    subject_main_dir: str,
    work_dir: str,
    docker_image: str,
) -> str:
    """
    Convert FreeSurfer brain.mgz -> NIfTI and reorient to standard.
    Stored at subject_main_dir (above session level).

    Returns the host path to the converted NIfTI.
    """
    mgz = opj(subjects_dir, subject, 'mri', 'brain.mgz')
    if not Path(mgz).exists():
        raise FileNotFoundError(
            'FreeSurfer brain.mgz not found: {}'.format(mgz))

    mgz_staged = _stage(mgz, work_dir)
    fs_t1_work = opj(work_dir, 'desc-fsbrain.nii.gz')
    mgz_c   = _container_path(work_dir, os.path.basename(mgz_staged), docker_image)
    fs_t1_c = _container_path(work_dir, 'desc-fsbrain.nii.gz',        docker_image)

    run_cmd(work_dir=work_dir, docker_image=docker_image,
            cmd=['mri_convert', mgz_c, fs_t1_c])
    run_cmd(work_dir=work_dir, docker_image=docker_image,
            cmd=['fslreorient2std', fs_t1_c])

    fs_t1_nii = build_output_name(subject_main_dir, subject, None, 'desc-fsbrain')
    shutil.copy(fs_t1_work, fs_t1_nii)
    return fs_t1_nii


def run_bbregister(
    bref_main: str,
    fs_t1_nii: str,
    subject: str,
    subject_main_dir: str,
    subjects_dir: str,
    work_dir: str,
    docker_image: str,
) -> tuple:
    """
    FLIRT initialisation followed by bbregister (BREF_MAIN -> FS T1).
    Outputs stored at subject_main_dir (above session level).

    Returns (bbreg_dat, sbref2fs_fslmat) as host paths.
    """
    _stage(bref_main, work_dir)
    _stage(fs_t1_nii,   work_dir)

    bref_c  = _container_path(work_dir, os.path.basename(bref_main), docker_image)
    fs_t1_c = _container_path(work_dir, os.path.basename(fs_t1_nii),   docker_image)

    # FLIRT initialisation
    init_mat_c = _container_path(work_dir, 'sbref_initial_reg.mat', docker_image)
    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'flirt',
            '-in',  bref_c,
            '-ref', fs_t1_c,
            '-dof', '6',
            '-cost', 'mutualinfo',
            '-omat', init_mat_c,
        ],
    )

    # Stage FreeSurfer subject tree
    subj_fs_dst = opj(work_dir, 'subjects', subject)
    if not Path(subj_fs_dst).exists():
        shutil.copytree(opj(subjects_dir, subject), subj_fs_dst)

    init_dat_c     = _container_path(work_dir, 'sbref_initial_reg.dat', docker_image)
    subjects_dir_c = _container_path(work_dir, 'subjects',              docker_image)

    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'tkregister2',
            '--s',    subject,
            '--mov',  bref_c,
            '--targ', fs_t1_c,
            '--fsl',  init_mat_c,
            '--reg',  init_dat_c,
            '--noedit',
        ],
        env_vars={'SUBJECTS_DIR': subjects_dir_c},
    )

    bbreg_dat_c    = _container_path(work_dir, 'sbref_bbreg.dat',     docker_image)
    sbref2fs_mat_c = _container_path(work_dir, 'sbref_bbreg_fsl.mat', docker_image)

    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'bbregister',
            '--s',        subject,
            '--mov',      bref_c,
            '--init-reg', init_dat_c,
            '--reg',      bbreg_dat_c,
            '--fslmat',   sbref2fs_mat_c,
            '--bold',
        ],
        env_vars={'SUBJECTS_DIR': subjects_dir_c},
    )

    # QC: apply registration
    aligned_c = _container_path(work_dir, 'BREF_MAIN_aligned.nii.gz', docker_image)
    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'flirt',
            '-in',       bref_c,
            '-ref',      fs_t1_c,
            '-applyxfm', '-init', sbref2fs_mat_c,
            '-out',      aligned_c,
        ],
    )

    bbreg_dat = build_output_name(
        subject_main_dir, subject, None, 'desc-sbref2fs_bbr', extension='.dat')
    sbref2fs_fslmat = build_output_name(
        subject_main_dir, subject, None, 'desc-sbref2fs_bbr_fsl', extension='.mat')
    aligned_final = build_output_name(
        subject_main_dir, subject, None, 'BREF_MAIN_aligned')

    shutil.copy(opj(work_dir, 'sbref_bbreg.dat'),            bbreg_dat)
    shutil.copy(opj(work_dir, 'sbref_bbreg_fsl.mat'),        sbref2fs_fslmat)
    shutil.copy(opj(work_dir, 'BREF_MAIN_aligned.nii.gz'), aligned_final)

    return bbreg_dat, sbref2fs_fslmat


def register_sbref_to_main(
    sbref_file: str,
    bref_main: str,
    task_label: str,
    run_label: str,
    subject_output_dir: str,
    work_dir: str,
    docker_image: str,
) -> str:
    """
    Register sbref_i to BREF_MAIN with FLIRT (normcorr, DOF 6).

    Returns the host path to the .mat file.
    """
    _stage(sbref_file,  work_dir)
    _stage(bref_main, work_dir)

    sbref_c  = _container_path(work_dir, os.path.basename(sbref_file),  docker_image)
    main_c = _container_path(work_dir, os.path.basename(bref_main), docker_image)
    mat_name = '{}_{}_sbref_to_bref_main.mat'.format(task_label, run_label)
    vol_name = '{}_{}_sbref_to_bref_main.nii.gz'.format(task_label, run_label)
    mat_c    = _container_path(work_dir, mat_name, docker_image)
    vol_c    = _container_path(work_dir, vol_name, docker_image)

    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'flirt',
            '-in',   sbref_c,
            '-ref',  main_c,
            '-dof',  '6',
            '-cost', 'normcorr',
            '-omat', mat_c,
            '-out',  vol_c,
        ],
    )

    mat_final = opj(subject_output_dir, mat_name)
    shutil.copy(opj(work_dir, mat_name), mat_final)
    return mat_final


def run_mcflirt(
    bold_file: str,
    sbref_i: str,
    work_dir: str,
    docker_image: str,
) -> tuple:
    """
    Run MCFLIRT on *bold_file*, referencing *sbref_i* (the run-matched sbref).

    Returns (mcf_nii, mcf_par, mcf_mats_dir) as host paths inside work_dir.
    """
    _stage(bold_file, work_dir)
    _stage(sbref_i,   work_dir)

    bold_c        = _container_path(work_dir, os.path.basename(bold_file), docker_image)
    sbref_i_c     = _container_path(work_dir, os.path.basename(sbref_i),   docker_image)
    mcf_prefix_c  = _container_path(work_dir, 'bold_mcf',                  docker_image)

    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'mcflirt',
            '-in',      bold_c,
            '-reffile', sbref_i_c,
            '-out',     mcf_prefix_c,
            '-mats',
            '-plots',
            '-report',
        ],
    )

    mcf_prefix = opj(work_dir, 'bold_mcf')
    return mcf_prefix + '.nii.gz', mcf_prefix + '.par', mcf_prefix + '.mat'


def concat_transforms(
    mcf_mats_dir: str,
    sbref_i_to_main_mat: str,
    sbref2fs_fslmat: str,
    combined_mats_dir: str,
    work_dir: str,
    docker_image: str,
) -> None:
    os.makedirs(combined_mats_dir, exist_ok=True)
    mat_files = sorted(glob.glob(opj(mcf_mats_dir, 'MAT_*')))
    if not mat_files:
        raise FileNotFoundError('No MAT_* files found in {}'.format(mcf_mats_dir))

    for mat in mat_files:
        _stage(mat, work_dir)
    _stage(sbref_i_to_main_mat, work_dir)
    _stage(sbref2fs_fslmat, work_dir)

    mats_c     = _container_path(work_dir, os.path.basename(mcf_mats_dir),          docker_image)
    combined_c = _container_path(work_dir, os.path.basename(combined_mats_dir),     docker_image)
    m1_c       = _container_path(work_dir, os.path.basename(sbref_i_to_main_mat), docker_image)
    m2_c       = _container_path(work_dir, os.path.basename(sbref2fs_fslmat),       docker_image)

    shell_script = (
        f'for mat in {mats_c}/MAT_*; do '
        f'  bn=$(basename "$mat"); '
        f'  convert_xfm -omat {combined_c}/tmp_$bn -concat {m1_c} $mat && '
        f'  convert_xfm -omat {combined_c}/$bn     -concat {m2_c} {combined_c}/tmp_$bn && '
        f'  rm {combined_c}/tmp_$bn; '
        f'done'
    )

    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=['bash', '-c', shell_script],
    )


def apply_xfm4d(
    bold_file: str,
    fs_t1_nii: str,
    combined_mats_dir: str,
    work_dir: str,
    bold_fs_out: str,
    docker_image: str,
) -> None:
    """
    Resample *bold_file* into FS-T1 space with a single interpolation step,
    preserving the native BOLD voxel size.
    """
    _stage(bold_file, work_dir)
    _stage(fs_t1_nii, work_dir)

    bold_c  = _container_path(work_dir, os.path.basename(bold_file),  docker_image)
    fs_t1_c = _container_path(work_dir, os.path.basename(fs_t1_nii),  docker_image)

    # Extract first volume to read voxel size
    res_ref = opj(work_dir, 'res_ref.nii.gz')
    run_local(['fslroi', bold_file, res_ref, '0', '1'])
    vox = fsl_val(res_ref, 'pixdim1')

    # Resample FS T1 to BOLD voxel size (still in FS space)
    res_ref_hd_c = _container_path(work_dir, 'res_ref_correct_header.nii.gz', docker_image)
    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'flirt',
            '-in',          fs_t1_c,
            '-ref',         fs_t1_c,
            '-applyisoxfm', vox,
            '-out',         res_ref_hd_c,
        ],
    )

    bold_fs_out_c   = _container_path(work_dir, os.path.basename(bold_fs_out),       docker_image)
    combined_mats_c = _container_path(work_dir, os.path.basename(combined_mats_dir), docker_image)

    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'applyxfm4D',
            bold_c,
            res_ref_hd_c,
            bold_fs_out_c,
            combined_mats_c,
            '-fourdigit',
            '-interp', 'trilinear',
        ],
    )


def project_to_surface(
    bold_fs_out: str,
    subject: str,
    subjects_dir: str,
    bold_base: str,
    work_dir: str,
    docker_image: str,
) -> dict:
    """
    Project *bold_fs_out* to lh and rh cortical surfaces via mri_vol2surf.

    Returns a dict mapping hemisphere ('lh', 'rh') -> output GIFTI path.
    """
    subj_fs_dst = opj(work_dir, 'subjects', subject)
    if not Path(subj_fs_dst).exists():
        shutil.copytree(opj(subjects_dir, subject), subj_fs_dst)

    bold_staged    = _stage(bold_fs_out, work_dir)
    bold_c         = _container_path(work_dir, os.path.basename(bold_staged), docker_image)
    subjects_dir_c = _container_path(work_dir, 'subjects',                    docker_image)

    hemi_map = {'lh': 'L', 'rh': 'R'}
    outputs  = {}

    for hemi, hemi_gifti in hemi_map.items():
        surf_name = '{}_space-fsnative_hemi-{}_bold.func.gii'.format(
            bold_base, hemi_gifti)
        surf_work = opj(work_dir, surf_name)
        surf_c    = _container_path(work_dir, surf_name, docker_image)

        run_cmd(
            work_dir=work_dir,
            docker_image=docker_image,
            cmd=[
                'mri_vol2surf',
                '--mov',          bold_c,
                '--hemi',         hemi,
                '--projfrac-avg', '0.2', '0.8', '0.1',
                '--o',            surf_c,
                '--trgsubject',   subject,
                '--cortex',
                '--regheader',    subject,
            ],
            env_vars={'SUBJECTS_DIR': subjects_dir_c},
        )

        outputs[hemi] = surf_work
        print('  Created surface timeseries: {}'.format(surf_work))

    return outputs


# ---------------------------------------------------------------------------
# Per-run pipeline
# ---------------------------------------------------------------------------

def process_run(
    bold_file: str,
    sbref_file: str,
    bref_main: str,
    sbref2fs_fslmat: str,
    fs_t1_nii: str,
    subject: str,
    session: str,
    subjects_dir: str,
    subject_output_dir: str,
    docker_image: str,
    overwrite: dict,
) -> dict:
    """
    Execute all per-run steps:
        2b: sbref_i -> BREF_MAIN  |  3: MCFLIRT  |  4: concat xfm
        5: applyxfm4D               |  6: surface project

    Returns a dict of final output paths for this run.
    """
    ow = {k: False for k in STEP_KEYS}
    ow.update(overwrite)

    run_label, task_label = get_labels(bold_file)

    run_suffix = '_'.join(t for t in [task_label, run_label] if t)

    work_dir = opj(subject_output_dir, run_suffix)
    os.makedirs(work_dir, exist_ok=True)
    safe_work_dir = make_safe_workdir(work_dir)

    base = f'{task_label}_{run_label}'

    def _final(suffix, ext='.nii.gz'):
        return build_output_name(
            subject_output_dir, subject, session, suffix, extension=ext)

    def _work(filename):
        return opj(safe_work_dir, filename)

    # ------------------------------------------------------------------
    # Step 2b - Register sbref_i to BREF_MAIN
    # ------------------------------------------------------------------
    print('\n  [Step 2b] Registering sbref_i to BREF_MAIN...')

    mat_name = '{}_sbref_to_bref_main.mat'.format(base)
    sbref_to_main_final = opj(subject_output_dir, mat_name)
    sbref_to_main_work  = _work(mat_name)

    if not check_skip(
        {'sbref_to_main': sbref_to_main_final},
        ow['sbref_to_main'],
        'Step 2b: sbref_i -> BREF_MAIN',
        workdir_paths={'sbref_to_main': sbref_to_main_work},
    ):
        sbref_to_main_final = register_sbref_to_main(
            sbref_file=sbref_file,
            bref_main=bref_main,
            task_label=task_label,
            run_label=run_label,
            subject_output_dir=subject_output_dir,
            work_dir=safe_work_dir,
            docker_image=docker_image,
        )
        shutil.copy(sbref_to_main_final, sbref_to_main_work)

    print('  sbref_i -> BREF_MAIN mat: {}'.format(sbref_to_main_final))

    # ------------------------------------------------------------------
    # Step 3 - MCFLIRT (reference = sbref_i)
    # ------------------------------------------------------------------
    print('\n  [Step 3] MCFLIRT motion correction (ref = sbref_i)...')

    mcf_mats_dir_final = opj(
        subject_output_dir, '{}_desc-mcflirt.mat'.format(base))
    mcf_par_final      = _final('{}_desc-mcflirt_motion'.format(base), ext='.par')
    mcf_mats_dir_work  = _work('bold_mcf.mat')
    mcf_par_work       = _work('bold_mcf.par')

    if not check_skip(
        {'mcf_mats': mcf_mats_dir_final, 'mcf_par': mcf_par_final},
        ow['mcflirt'],
        'Step 3: MCFLIRT',
        workdir_paths={'mcf_mats': mcf_mats_dir_work, 'mcf_par': mcf_par_work},
    ):
        _, mcf_par_work, mcf_mats_dir_work = run_mcflirt(
            bold_file=bold_file,
            sbref_i=sbref_file,
            work_dir=safe_work_dir,
            docker_image=docker_image,
        )
        if Path(mcf_mats_dir_final).exists():
            shutil.rmtree(mcf_mats_dir_final)
        shutil.copytree(mcf_mats_dir_work, mcf_mats_dir_final)
        shutil.copy(mcf_par_work, mcf_par_final)

    print('  Motion parameters : {}'.format(mcf_par_final))
    print('  Per-volume mats   : {}'.format(mcf_mats_dir_final))

    # ------------------------------------------------------------------
    # Step 4 - Concatenate transforms: VOL -> sbref_i -> BREF_MAIN -> FS_T1
    # ------------------------------------------------------------------
    print('\n  [Step 4] Concatenating transforms '
          '(VOL -> sbref_i -> BREF_MAIN -> FS_T1)...')

    combined_mats_dir_final = opj(
        subject_output_dir,
        '{}_desc-mcflirt+bbreg_transforms'.format(base),
    )
    combined_mats_dir_work = _work('bold2fs.mat')

    if not check_skip(
        {'combined_mats': combined_mats_dir_final},
        ow['concat_xfm'],
        'Step 4: concat transforms',
        workdir_paths={'combined_mats': combined_mats_dir_work},
    ):
        concat_transforms(
            mcf_mats_dir=mcf_mats_dir_work,
            sbref_i_to_main_mat=sbref_to_main_work,
            sbref2fs_fslmat=sbref2fs_fslmat,
            combined_mats_dir=combined_mats_dir_work,
            work_dir=safe_work_dir,
            docker_image='local',
        )
        if Path(combined_mats_dir_final).exists():
            shutil.rmtree(combined_mats_dir_final)
        shutil.copytree(combined_mats_dir_work, combined_mats_dir_final)

    print('  Combined mats -> {}'.format(combined_mats_dir_final))

    # ------------------------------------------------------------------
    # Step 5 - applyxfm4D
    # ------------------------------------------------------------------
    print('\n  [Step 5] Applying combined transforms (single interpolation)...')

    bold_fs_out_final = _final('{}_space-fsT1_desc-moco_bbreg_bold'.format(base))
    bold_fs_out_work  = _work('bold_space-fsT1_desc-moco_bbreg.nii.gz')

    if not check_skip(
        {'bold_fs': bold_fs_out_final},
        ow['applywarp'],
        'Step 5: applyxfm4D',
        workdir_paths={'bold_fs': bold_fs_out_work},
    ):
        apply_xfm4d(
            bold_file=bold_file,
            fs_t1_nii=fs_t1_nii,
            combined_mats_dir=combined_mats_dir_work,
            work_dir=safe_work_dir,
            bold_fs_out=bold_fs_out_work,
            docker_image=docker_image,
        )
        shutil.copy(bold_fs_out_work, bold_fs_out_final)

    print('  Single-step output -> {}'.format(bold_fs_out_final))

    # ------------------------------------------------------------------
    # Step 6 - Surface projection
    # ------------------------------------------------------------------
    print('\n  [Step 6] Projecting to cortical surface...')
    surf_lh_final = opj(
        subject_output_dir,
        '{}_{}_{}_space-fsnative_hemi-L_bold.func.gii'.format(subject, session, base))
    surf_rh_final = opj(
        subject_output_dir,
        '{}_{}_{}_space-fsnative_hemi-R_bold.func.gii'.format(subject, session, base))

    if not check_skip(
        {'surf_lh': surf_lh_final, 'surf_rh': surf_rh_final},
        ow['surf_project'],
        'Step 6: surface projection',
    ):
        outputs = project_to_surface(
            bold_fs_out=bold_fs_out_work,
            subject=subject,
            subjects_dir=subjects_dir,
            bold_base=base,
            work_dir=safe_work_dir,
            docker_image=docker_image,
        )
        shutil.copy(outputs['lh'], surf_lh_final)
        shutil.copy(outputs['rh'], surf_rh_final)

    shutil.rmtree(work_dir)
    return {
        'sbref_to_main_mat': sbref_to_main_final,
        'mcf_motion_params':   mcf_par_final,
        'mcf_mats':            mcf_mats_dir_final,
        'combined_mats':       combined_mats_dir_final,
        'bold_space_fsT1':     bold_fs_out_final,
        'surf_lh':             surf_lh_final,
        'surf_rh':             surf_rh_final,
    }


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    bids_dir: str,
    input_file: str,
    output_file: str,
    subject: str,
    session: str = 'ses-01',
    subjects_dir: str = None,
    docker_image: str = 'local',
    overwrite: dict = None,
    include_files: list = None,
    include_patterns: list = None,
    exclude_patterns: list = None,
    bref_main_spec: str = None,
) -> dict:
    """
    Run the full motion correction + registration + surface projection pipeline.

    BREF_MAIN and bbregister are subject-level (one each, stored above session).
    Per-run steps (2b–6) are repeated for each discovered BOLD file, with outputs
    placed in the session sub-folder extracted from each filename.

    Parameters
    ----------
    include_files : list of str, optional
        Exact basenames to find recursively under <input>/subject/.
    include_patterns : list of str, optional
        Glob patterns relative to <input>/subject/.
    exclude_patterns : list of str, optional
        fnmatch patterns applied to basenames after include selection.
    bref_main_spec : str, optional
        None = auto-detect; a basename = find under <input>/subject/;
        an absolute/relative path = use directly.

    Returns a dict mapping run keys -> per-run output dicts.
    """
    ow = {k: False for k in STEP_KEYS}
    if overwrite:
        unknown = set(overwrite) - set(STEP_KEYS)
        if unknown:
            raise ValueError(
                'Unknown overwrite key(s): {}.  Valid keys: {}'.format(
                    sorted(unknown), STEP_KEYS)
            )
        ow.update(overwrite)

    input_dir  = str(Path(opj(bids_dir, 'derivatives', input_file)).resolve())
    output_dir = str(Path(opj(bids_dir, 'derivatives', output_file)).resolve())

    subject_input_dir  = opj(input_dir,  subject)
    subject_main_dir = opj(output_dir, subject)   # above session level

    os.makedirs(subject_main_dir, exist_ok=True)

    if subjects_dir is None:
        subjects_dir = os.environ.get('SUBJECTS_DIR', '')
    if not subjects_dir:
        raise ValueError('--subjects-dir not set and $SUBJECTS_DIR is empty.')

    main_work_dir = opj(subject_main_dir, '_session_work')
    os.makedirs(main_work_dir, exist_ok=True)
    safe_main_work = make_safe_workdir(main_work_dir)

    print('-' * 55)
    print('Processing: Motion Correction + Registration')
    print('-' * 55)
    print(' Input       : {}'.format(input_dir))
    print(' Output      : {}'.format(output_dir))
    print(' Subject     : {}'.format(subject))
    print(' SUBJECTS_DIR: {}'.format(subjects_dir))
    print(' Docker      : {}'.format(docker_image))
    print('-' * 55)

    # ------------------------------------------------------------------
    # Discover BOLD runs
    # ------------------------------------------------------------------
    bold_files = _find_bold_files(
        subject_input_dir=subject_input_dir,
        subject=subject,
        session=session,
        include_files=include_files,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    print('\nFound {} BOLD run(s).'.format(len(bold_files)))
    for idx, bf in enumerate(bold_files, start=1):
        print('  {}. {}'.format(idx, os.path.basename(bf)))

    # ------------------------------------------------------------------
    # Step 1 - BREF_MAIN  (subject-level, above session)
    # ------------------------------------------------------------------
    print('\n[Step 1] Creating BREF_MAIN (subject-level)...')

    note_file         = opj(subject_main_dir, 'bref_main_notes.txt')
    bref_main_final = build_output_name(subject_main_dir, subject, None, 'BREF_MAIN')

    if not check_skip(
        {'bref_main': bref_main_final},
        ow['bref_main'],
        'Step 1: BREF_MAIN',
    ):
        bref_main_final = make_bref_main(
            bref_spec=bref_main_spec,
            search_dir=subject_input_dir,
            subject=subject,
            subject_main_dir=subject_main_dir,
            note_file=note_file,
            work_dir=safe_main_work,
            docker_image=docker_image,
        )

    print('  -> {}'.format(bref_main_final))

    # ------------------------------------------------------------------
    # Step 2a - FreeSurfer T1 conversion + bbregister (subject-level)
    # ------------------------------------------------------------------
    print('\n[Step 2a] bbregister (BREF_MAIN -> FreeSurfer T1)...')

    fs_t1_nii_final       = build_output_name(
        subject_main_dir, subject, None, 'desc-fsbrain')
    bbreg_dat_final       = build_output_name(
        subject_main_dir, subject, None, 'desc-sbref2fs_bbr', extension='.dat')
    sbref2fs_fslmat_final = build_output_name(
        subject_main_dir, subject, None, 'desc-sbref2fs_bbr_fsl', extension='.mat')

    if not check_skip(
        {
            'fs_t1_nii':       fs_t1_nii_final,
            'bbreg_dat':       bbreg_dat_final,
            'sbref2fs_fslmat': sbref2fs_fslmat_final,
        },
        ow['bbregister'],
        'Step 2a: bbregister',
    ):
        if not Path(fs_t1_nii_final).exists():
            fs_t1_nii_final = convert_fs_t1(
                subjects_dir=subjects_dir,
                subject=subject,
                subject_main_dir=subject_main_dir,
                work_dir=safe_main_work,
                docker_image=docker_image,
            )

        bbreg_dat_final, sbref2fs_fslmat_final = run_bbregister(
            bref_main=bref_main_final,
            fs_t1_nii=fs_t1_nii_final,
            subject=subject,
            subject_main_dir=subject_main_dir,
            subjects_dir=subjects_dir,
            work_dir=safe_main_work,
            docker_image=docker_image,
        )

    print('  bbregister .dat : {}'.format(bbreg_dat_final))
    print('  FSL .mat        : {}'.format(sbref2fs_fslmat_final))
    print('  FS T1 NIfTI     : {}'.format(fs_t1_nii_final))

    # ------------------------------------------------------------------
    # Per-run processing
    # ------------------------------------------------------------------
    all_results = {}

    for run_idx, bold_file in enumerate(bold_files, start=1):
        print('\n' + '=' * 55)
        print('Processing run {}/{}: {}'.format(
            run_idx, len(bold_files), os.path.basename(bold_file)))
        print('=' * 55)

        # Extract session from filename; fall back to --ses default
        ses = _extract_session(bold_file) or session
        subject_session_dir = opj(output_dir, subject, ses)
        os.makedirs(subject_session_dir, exist_ok=True)

        sbref_file, sbref_source = _find_sbref_for_bold(
            bold_file=bold_file,
            subject=subject,
            session=ses,
            subject_output_dir=subject_session_dir,
        )

        print('  BOLD  : {}'.format(bold_file))
        print('  SBREF : {} [{}]'.format(sbref_file, sbref_source))

        run_results = process_run(
            bold_file=bold_file,
            sbref_file=sbref_file,
            bref_main=bref_main_final,
            sbref2fs_fslmat=sbref2fs_fslmat_final,
            fs_t1_nii=fs_t1_nii_final,
            subject=subject,
            session=ses,
            subjects_dir=subjects_dir,
            subject_output_dir=subject_session_dir,
            docker_image=docker_image,
            overwrite=ow,
        )

        run_label, _ = get_labels(bold_file)
        key = run_label if run_label else 'run-{:02d}'.format(run_idx)
        all_results[key] = run_results
        print('\n  Run {} completed.'.format(run_idx))

    print('\n' + '=' * 55)
    print('All {} run(s) completed successfully.'.format(len(bold_files)))
    print('Output directory: {}'.format(subject_main_dir))
    print('=' * 55)

    subj_fs_dst = opj(main_work_dir, 'subjects')
    if Path(subj_fs_dst).exists():
        shutil.rmtree(subj_fs_dst)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Motion correction + bbregister + surface projection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',    required=True,
                     help='BIDS directory')
    req.add_argument('--input-file',  required=True,
                     help='Input derivatives label (under bids-dir/derivatives/)')
    req.add_argument('--output-file', required=True,
                     help='Output derivatives label')
    req.add_argument('--sub',         required=True,
                     help='Subject label (e.g. sub-01)')

    p.add_argument('--ses',          default='ses-01',
                   help='Fallback session label used when no include args are given, '
                        'or when a bold filename contains no ses- entity')
    p.add_argument('--subjects-dir', default=None,
                   help='FreeSurfer SUBJECTS_DIR (default: $SUBJECTS_DIR)')
    p.add_argument('--docker',
                   default=os.environ.get('FSL_FREESURFER_IMAGE', 'local'),
                   help='Docker image for FSL/FreeSurfer tools, or "local"')

    sel = p.add_argument_group(
        'run selection',
        'Choose which BOLD files to process. '
        'At most one of --include-files / --include-patterns may be used. '
        'If neither is given, all *bold*.nii* in <input>/sub/ses/ are used.'
    )
    sel.add_argument(
        '--include-files',
        nargs='+',
        metavar='FILE',
        default=None,
        help='Exact basenames to find recursively under <input>/sub/.',
    )
    sel.add_argument(
        '--include-patterns',
        nargs='+',
        metavar='PATTERN',
        default=None,
        help='Glob patterns relative to <input>/sub/ '
             '(e.g. "ses-01/*task-pRF*bold*.nii.gz").',
    )
    sel.add_argument(
        '--exclude-patterns',
        nargs='+',
        metavar='PATTERN',
        default=None,
        help='fnmatch patterns applied to basenames to exclude after selection.',
    )

    bref = p.add_argument_group('BREF_MAIN selection')
    bref.add_argument(
        '--bref-main',
        default=None,
        metavar='NAME_OR_PATH',
        help='Basename to find under <input>/sub/, or an absolute/relative path '
             'to an existing file.  Omit for automatic selection (first sbref, '
             'or vol-0 of first bold).',
    )

    ow_group = p.add_argument_group(
        'overwrite options',
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

    if args.include_files and args.include_patterns:
        raise SystemExit(
            'error: --include-files and --include-patterns are mutually exclusive.')

    if args.overwrite_all:
        overwrite = {k: True for k in STEP_KEYS}
    else:
        overwrite = {k: (k in args.overwrite) for k in STEP_KEYS}

    args.sub = 'sub-' + args.sub.removeprefix('sub-')
    args.ses = 'ses-' + args.ses.removeprefix('ses-')

    run_pipeline(
        bids_dir=args.bids_dir,
        input_file=args.input_file,
        output_file=args.output_file,
        subject=args.sub,
        session=args.ses,
        subjects_dir=args.subjects_dir,
        docker_image=args.docker,
        overwrite=overwrite,
        include_files=args.include_files,
        include_patterns=args.include_patterns,
        exclude_patterns=args.exclude_patterns,
        bref_main_spec=args.bref_main,
    )


if __name__ == '__main__':
    main()
