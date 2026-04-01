#!/usr/bin/env python
"""
s02_coreg.py
===============
Motion correction, coregistration to FreeSurfer T1, and surface projection
for BOLD runs that have already been SDC-corrected.

Coregistration strategy
-----------------------
(1) Select first sbref as "master" - coregister to anatomy with bbregister
(2) Coregister each sbref_i to sbref_master   (FLIRT, normcorr, DOF 6)
(3) MCFLIRT per run, referencing the corresponding sbref_i
(4) Concatenate transforms:  VOL -> sbref_i -> sbref_master -> FS_T1

Overwrite behaviour
-------------------
Existence is checked against final BIDS-named files in *output_dir*.
Skipped steps restore outputs to *work_dir* so downstream steps can proceed.

Usage example
-------------
python s02_coreg.py \\
    --input-file   s1_sdc_AFNI \\
    --output-dir   s2_coreg \\
    --sub          sub-01 \\
    --ses          ses-01 \\
    --subjects-dir /data/freesurfer \\
    --docker       freesurfer/freesurfer:7.4.1
"""

import argparse
import glob
import os
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
    'bref_master',
    'bbregister',
    'sbref_to_master',   # per-run sbref_i -> sbref_master
    'mcflirt',
    'concat_xfm',
    'applywarp',
    'surf_project',
]

# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def make_bref_master(
    subject_input_dir: str,
    subject: str,
    session: str,
    subject_output_dir: str,
    note_file: str,
    work_dir: str,
    docker_image: str,
) -> str:
    """
    Build BREF_MASTER from the first SBREF found in *subject_input_dir*.

    Returns the host path to BREF_MASTER.nii.gz.
    """
    pattern = os.path.join(
        subject_input_dir, '{}_{}*sbref*.nii*'.format(subject, session))
    sbrefs = sorted(glob.glob(pattern))

    bref_master = build_output_name(
        subject_output_dir, subject, session, 'BREF_MASTER')

    if sbrefs:
        sbref_src = sbrefs[0]
        print('  Using SBREF as BREF_MASTER: {}'.format(sbref_src))

        if sbref_src.endswith('.nii'):
            nii_dst = bref_master.replace('.gz', '')
            shutil.copy(sbref_src, nii_dst)
            subprocess.run(['gzip', nii_dst], check=True)
        else:
            shutil.copy(sbref_src, bref_master)

        note = 'sbref - {} was used as BREF_MASTER\n'.format(sbref_src)

    else:
        raise FileNotFoundError(
            'No SBREF files found in {}'.format(subject_input_dir))

    with open(note_file, 'a') as fh:
        fh.write(note)

    # Reorient to standard (in-place via work_dir)
    _stage(bref_master, work_dir)
    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=['fslreorient2std',
             _container_path(work_dir, os.path.basename(bref_master), docker_image)],
    )
    shutil.copy(os.path.join(work_dir, os.path.basename(bref_master)), bref_master)

    return bref_master


def convert_fs_t1(
    subjects_dir: str,
    subject: str,
    subject_output_dir: str,
    work_dir: str,
    docker_image: str,
) -> str:
    """
    Convert FreeSurfer brain.mgz -> NIfTI and reorient to standard.

    Returns the host path to the converted NIfTI.
    """
    mgz = os.path.join(subjects_dir, subject, 'mri', 'brain.mgz')
    if not Path(mgz).exists():
        raise FileNotFoundError(
            'FreeSurfer brain.mgz not found: {}'.format(mgz))

    mgz_staged = _stage(mgz, work_dir)
    fs_t1_work = os.path.join(work_dir, 'desc-fsbrain.nii.gz')
    mgz_c      = _container_path(work_dir, os.path.basename(mgz_staged), docker_image)
    fs_t1_c    = _container_path(work_dir, 'desc-fsbrain.nii.gz',        docker_image)

    run_cmd(work_dir=work_dir, docker_image=docker_image,
            cmd=['mri_convert', mgz_c, fs_t1_c])
    run_cmd(work_dir=work_dir, docker_image=docker_image,
            cmd=['fslreorient2std', fs_t1_c])

    fs_t1_nii = build_output_name(
        subject_output_dir, subject, None, 'desc-fsbrain')
    shutil.copy(fs_t1_work, fs_t1_nii)
    return fs_t1_nii


def run_bbregister(
    bref_master: str,
    fs_t1_nii: str,
    subject: str,
    session: str,
    subject_output_dir: str,
    subjects_dir: str,
    work_dir: str,
    docker_image: str,
) -> tuple:
    """
    FLIRT initialisation followed by bbregister (BREF_MASTER -> FS T1).

    Returns (bbreg_dat, sbref2fs_fslmat) as host paths in subject_output_dir.
    """
    _stage(bref_master, work_dir)
    _stage(fs_t1_nii,   work_dir)

    bref_c  = _container_path(work_dir, os.path.basename(bref_master), docker_image)
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
    subj_fs_dst = os.path.join(work_dir, 'subjects', subject)
    if not Path(subj_fs_dst).exists():
        shutil.copytree(os.path.join(subjects_dir, subject), subj_fs_dst)

    init_dat_c      = _container_path(work_dir, 'sbref_initial_reg.dat', docker_image)
    subjects_dir_c  = _container_path(work_dir, 'subjects',              docker_image)

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
    aligned_c = _container_path(work_dir, 'BREF_MASTER_aligned.nii.gz', docker_image)
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
        subject_output_dir, subject, session, 'desc-sbref2fs_bbr', extension='.dat')
    sbref2fs_fslmat = build_output_name(
        subject_output_dir, subject, session, 'desc-sbref2fs_bbr_fsl', extension='.mat')

    shutil.copy(os.path.join(work_dir, 'sbref_bbreg.dat'),     bbreg_dat)
    shutil.copy(os.path.join(work_dir, 'sbref_bbreg_fsl.mat'), sbref2fs_fslmat)

    aligned_final = build_output_name(
        subject_output_dir, subject, session, 'BREF_MASTER_aligned')
    shutil.copy(os.path.join(work_dir, 'BREF_MASTER_aligned.nii.gz'), aligned_final)

    return bbreg_dat, sbref2fs_fslmat


def register_sbref_to_master(
    sbref_file: str,
    bref_master: str,
    task_label: str,
    run_label: str,
    subject_output_dir: str,
    work_dir: str,
    docker_image: str,
) -> str:
    """
    Register sbref_i to BREF_MASTER with FLIRT (normcorr, DOF 6).

    Returns the host path to the .mat file.
    """
    _stage(sbref_file,  work_dir)
    _stage(bref_master, work_dir)

    sbref_c  = _container_path(work_dir, os.path.basename(sbref_file),  docker_image)
    master_c = _container_path(work_dir, os.path.basename(bref_master), docker_image)

    mat_name = '{}_{}_brefi_to_bref_master.mat'.format(task_label, run_label)
    vol_name = '{}_{}_brefi_to_bref_master.nii.gz'.format(task_label, run_label)
    mat_c    = _container_path(work_dir, mat_name, docker_image)

    run_cmd(
        work_dir=work_dir,
        docker_image=docker_image,
        cmd=[
            'flirt',
            '-in',   sbref_c,
            '-ref',  master_c,
            '-dof',  '6',
            '-cost', 'normcorr',
            '-omat', mat_c,
            '-out', os.path.join(work_dir, vol_name),
        ],
    )

    mat_final = os.path.join(subject_output_dir, mat_name)
    shutil.copy(os.path.join(work_dir, mat_name), mat_final)
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

    mcf_prefix = os.path.join(work_dir, 'bold_mcf')
    return mcf_prefix + '.nii.gz', mcf_prefix + '.par', mcf_prefix + '.mat'


def concat_transforms(
    mcf_mats_dir: str,
    sbref_i_to_master_mat: str,
    sbref2fs_fslmat: str,
    combined_mats_dir: str,
) -> None:
    """
    Concatenate per-volume transforms: VOL -> sbref_i -> sbref_master -> FS_T1.
    """
    os.makedirs(combined_mats_dir, exist_ok=True)
    mat_files = sorted(glob.glob(os.path.join(mcf_mats_dir, 'MAT_*')))
    if not mat_files:
        raise FileNotFoundError(
            'No MAT_* files found in {}'.format(mcf_mats_dir))

    for mat in mat_files:
        bn      = os.path.basename(mat)
        tmp_mat = os.path.join(combined_mats_dir, 'tmp_{}'.format(bn))
        out_mat = os.path.join(combined_mats_dir, bn)

        # Step 1: VOL -> sbref_master
        run_local([
            'convert_xfm',
            '-omat',   tmp_mat,
            '-concat', sbref_i_to_master_mat,
            mat,
        ], verbose=False)

        # Step 2: VOL -> FS_T1
        run_local([
            'convert_xfm',
            '-omat',   out_mat,
            '-concat', sbref2fs_fslmat,
            tmp_mat,
        ], verbose=False)

        os.remove(tmp_mat)


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
    _stage(bold_file,  work_dir)
    _stage(fs_t1_nii,  work_dir)

    bold_c  = _container_path(work_dir, os.path.basename(bold_file),  docker_image)
    fs_t1_c = _container_path(work_dir, os.path.basename(fs_t1_nii),  docker_image)

    # Extract first volume to read voxel size
    res_ref = os.path.join(work_dir, 'res_ref.nii.gz')
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
    subject_output_dir: str,
    bold_base: str,
    work_dir: str,
    docker_image: str,
) -> dict:
    """
    Project *bold_fs_out* to lh and rh cortical surfaces via mri_vol2surf.

    Returns a dict mapping hemisphere ('lh', 'rh') -> output GIFTI path.
    """
    subj_fs_dst = os.path.join(work_dir, 'subjects', subject)
    if not Path(subj_fs_dst).exists():
        shutil.copytree(os.path.join(subjects_dir, subject), subj_fs_dst)

    bold_staged    = _stage(bold_fs_out, work_dir)
    bold_c         = _container_path(work_dir, os.path.basename(bold_staged), docker_image)
    subjects_dir_c = _container_path(work_dir, 'subjects',                    docker_image)

    hemi_map = {'lh': 'L', 'rh': 'R'}
    outputs  = {}

    for hemi, hemi_gifti in hemi_map.items():
        surf_name = '{}_space-fsnative_hemi-{}_bold.func.gii'.format(
            bold_base, hemi_gifti)
        surf_work = os.path.join(work_dir, surf_name)
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

        surf_final = os.path.join(subject_output_dir, surf_name)
        shutil.copy(surf_work, surf_final)
        outputs[hemi] = surf_final
        print('  Created surface timeseries: {}'.format(surf_final))

    return outputs


# ---------------------------------------------------------------------------
# Per-run pipeline
# ---------------------------------------------------------------------------

def process_run(
    bold_file: str,
    sbref_file: str,
    bref_master: str,
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
        sbref_i->master  |  MCFLIRT  |  concat  |  applyxfm4D  |  surface project

    Returns a dict of final output paths for this run.
    """
    ow = {k: False for k in STEP_KEYS}
    ow.update(overwrite)

    run_label, task_label = get_labels(bold_file)
    run_suffix = '_'.join(t for t in [task_label, run_label] if t)

    work_dir = os.path.join(subject_output_dir, run_suffix)
    os.makedirs(work_dir, exist_ok=True)
    safe_work_dir = make_safe_workdir(work_dir)

    # ------------------------------------------------------------------
    # Derive bold_base: BIDS entities from task onward, no sub/ses prefix,
    # no file extensions.  Used as suffix arg to build_output_name.
    # ------------------------------------------------------------------
    base = _bold_base(bold_file, subject, session)

    def _final(suffix, ext='.nii.gz'):
        return build_output_name(
            subject_output_dir, subject, session, suffix, extension=ext)

    def _work(filename):
        return os.path.join(safe_work_dir, filename)

    # ------------------------------------------------------------------
    # Step 2b - Register sbref_i to BREF_MASTER
    # ------------------------------------------------------------------
    print('\n  [Step 2b] Registering sbref_i to BREF_MASTER...')

    mat_name = '{}_{}_brefi_to_bref_master.mat'.format(task_label, run_label)
    sbref_to_master_final = os.path.join(subject_output_dir, mat_name)
    sbref_to_master_work  = _work(mat_name)

    if not check_skip(
        {'sbref_to_master': sbref_to_master_final},
        ow['sbref_to_master'],
        'Step 2b: sbref_i -> BREF_MASTER',
        workdir_paths={'sbref_to_master': sbref_to_master_work},
    ):
        sbref_to_master_final = register_sbref_to_master(
            sbref_file=sbref_file,
            bref_master=bref_master,
            task_label=task_label,
            run_label=run_label,
            subject_output_dir=subject_output_dir,
            work_dir=safe_work_dir,
            docker_image=docker_image,
        )
        shutil.copy(sbref_to_master_final, sbref_to_master_work)

    print('  sbref_i -> master mat: {}'.format(sbref_to_master_final))

    # ------------------------------------------------------------------
    # Step 3 - MCFLIRT (reference = sbref_i)
    # ------------------------------------------------------------------
    print('\n  [Step 3] MCFLIRT motion correction (ref = sbref_i)...')

    mcf_mats_dir_final = os.path.join(
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
    # Step 4 - Concatenate transforms: VOL -> sbref_i -> master -> FS_T1
    # ------------------------------------------------------------------
    print('\n  [Step 4] Concatenating transforms '
          '(VOL -> sbref_i -> BREF_MASTER -> FS_T1)...')

    combined_mats_dir_final = os.path.join(
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
            sbref_i_to_master_mat=sbref_to_master_work,
            sbref2fs_fslmat=sbref2fs_fslmat,
            combined_mats_dir=combined_mats_dir_work,
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

    surf_lh_final = os.path.join(
        subject_output_dir,
        '{}_space-fsnative_hemi-L_bold.func.gii'.format(base))
    surf_rh_final = os.path.join(
        subject_output_dir,
        '{}_space-fsnative_hemi-R_bold.func.gii'.format(base))

    if not check_skip(
        {'surf_lh': surf_lh_final, 'surf_rh': surf_rh_final},
        ow['surf_project'],
        'Step 6: surface projection',
    ):
        project_to_surface(
            bold_fs_out=bold_fs_out_work,
            subject=subject,
            subjects_dir=subjects_dir,
            subject_output_dir=subject_output_dir,
            bold_base=base,
            work_dir=safe_work_dir,
            docker_image=docker_image,
        )
    shutil.rmtree(work_dir,)
    return {
        'sbref_to_master_mat': sbref_to_master_final,
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
    bids_dir : str,
    input_file: str,
    output_file: str,
    subject: str,
    session: str = 'ses-01',
    subjects_dir: str = None,
    docker_image: str = 'local',
    overwrite: dict = None,
) -> dict:
    """
    Run the full motion correction + registration + surface projection pipeline.

    Steps 1 & bbregister are session-level (one BREF_MASTER, one bbregister).
    Steps 2b–6 are repeated per BOLD run.

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

    input_dir  = str(Path(
        os.path.join(bids_dir, 'derivatives',input_file)
        ).resolve())
    output_dir = str(Path(
        os.path.join(bids_dir, 'derivatives',output_file)
        ).resolve())

    subject_input_dir  = os.path.join(input_dir,  subject, session)
    subject_output_dir = os.path.join(output_dir, subject, session)

    os.makedirs(subject_output_dir, exist_ok=True)

    if subjects_dir is None:
        subjects_dir = os.environ.get('SUBJECTS_DIR', '')
    if not subjects_dir:
        raise ValueError('--subjects-dir not set and $SUBJECTS_DIR is empty.')

    session_work_dir = os.path.join(subject_output_dir, '_session_work')
    os.makedirs(session_work_dir, exist_ok=True)
    safe_session_work = make_safe_workdir(session_work_dir)

    print('-' * 55)
    print('Processing: Motion Correction + Registration')
    print('-' * 55)
    print(' Input       : {}'.format(input_dir))
    print(' Output      : {}'.format(output_dir))
    print(' Subject     : {}'.format(subject))
    print(' Session     : {}'.format(session))
    print(' SUBJECTS_DIR: {}'.format(subjects_dir))
    print(' Docker      : {}'.format(docker_image))
    print('-' * 55)

    # ------------------------------------------------------------------
    # Step 1 - BREF_MASTER
    # ------------------------------------------------------------------
    print('\n[Step 1] Creating BREF_MASTER...')

    bref_master_final = build_output_name(
        subject_output_dir, subject, session, 'BREF_MASTER')
    note_file = os.path.join(
        subject_output_dir, 'reference_method_notes.txt')

    if not check_skip(
        {'bref_master': bref_master_final},
        ow['bref_master'],
        'Step 1: BREF_MASTER',
    ):
        bref_master_final = make_bref_master(
            subject_input_dir=subject_input_dir,
            subject=subject,
            session=session,
            subject_output_dir=subject_output_dir,
            note_file=note_file,
            work_dir=safe_session_work,
            docker_image=docker_image,
        )

    print('  -> {}'.format(bref_master_final))

    # ------------------------------------------------------------------
    # Step 2a - FreeSurfer T1 conversion + bbregister (session-level)
    # ------------------------------------------------------------------
    print('\n[Step 2a] bbregister (BREF_MASTER -> FreeSurfer T1)...')

    fs_t1_nii_final       = build_output_name(
        subject_output_dir, subject, None, 'desc-fsbrain')
    bbreg_dat_final       = build_output_name(
        subject_output_dir, subject, session,
        'desc-sbref2fs_bbr', extension='.dat')
    sbref2fs_fslmat_final = build_output_name(
        subject_output_dir, subject, session,
        'desc-sbref2fs_bbr_fsl', extension='.mat')

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
                subject_output_dir=subject_output_dir,
                work_dir=safe_session_work,
                docker_image=docker_image,
            )

        bbreg_dat_final, sbref2fs_fslmat_final = run_bbregister(
            bref_master=bref_master_final,
            fs_t1_nii=fs_t1_nii_final,
            subject=subject,
            session=session,
            subject_output_dir=subject_output_dir,
            subjects_dir=subjects_dir,
            work_dir=safe_session_work,
            docker_image=docker_image,
        )

    print('  bbregister .dat : {}'.format(bbreg_dat_final))
    print('  FSL .mat        : {}'.format(sbref2fs_fslmat_final))
    print('  FS T1 NIfTI     : {}'.format(fs_t1_nii_final))

    # ------------------------------------------------------------------
    # Discover BOLD runs
    # ------------------------------------------------------------------
    bold_pattern = os.path.join(
        subject_input_dir, '{}_{}*bold*.nii*'.format(subject, session))
    bold_files = sorted(glob.glob(bold_pattern))
    if not bold_files:
        raise FileNotFoundError(
            'No BOLD files found for {}_{}.  Searched: {}'.format(
                subject, session, bold_pattern)
        )

    print('\nFound {} BOLD run(s).'.format(len(bold_files)))

    all_results = {}

    for run_idx, bold_file in enumerate(bold_files, start=1):
        print('\n' + '=' * 55)
        print('Processing run {}/{}: {}'.format(
            run_idx, len(bold_files), os.path.basename(bold_file)))
        print('=' * 55)

        run_label, task_label = get_labels(bold_file)

        # ------------------------------------------------------------------
        # Locate per-run sbref, or synthesise one from vol 0 of this BOLD run
        # ------------------------------------------------------------------
        if run_label:
            sbref_pat = os.path.join(
                subject_input_dir,
                '{}_{}_{}_{}*sbref*.nii*'.format(
                    subject, session, task_label, run_label))
            sbref_matches = sorted(glob.glob(sbref_pat))
        else:
            all_sbrefs = glob.glob(os.path.join(
                subject_input_dir,
                '{}_{}_{}_*sbref*.nii*'.format(subject, session, task_label)))
            sbref_matches = [f for f in all_sbrefs
                             if not re.search(r'run-\d+', os.path.basename(f))]

        if sbref_matches:
            sbref_file   = sbref_matches[0]
            sbref_source = 'input'
        else:
            # Build the expected output name and skip recompute if it exists
            parts = [subject]
            if session:
                parts.append(session)
            if task_label:
                parts.append(task_label)
            if run_label:
                parts.append(run_label)
            parts.append('desc-vol0bold_sbref')
            synthetic_sbref = os.path.join(
                subject_output_dir, '_'.join(parts) + '.nii.gz')

            if not Path(synthetic_sbref).exists():
                print('  No SBREF found — extracting vol 0 as synthetic sbref...')
                synthetic_sbref = make_first_vol_sbref(
                    bold_file=bold_file,
                    subject=subject,
                    session=session,
                    run_label=run_label,
                    task_label=task_label,
                    subject_output_dir=subject_output_dir,
                )
            else:
                print('  No SBREF found — reusing existing synthetic sbref.')

            sbref_file   = synthetic_sbref
            sbref_source = 'vol-0 (synthetic)'

        print('  BOLD  : {}'.format(bold_file))
        print('  SBREF : {} [{}]'.format(sbref_file, sbref_source))

        run_results = process_run(
            bold_file=bold_file,
            sbref_file=sbref_file,
            bref_master=bref_master_final,
            sbref2fs_fslmat=sbref2fs_fslmat_final,
            fs_t1_nii=fs_t1_nii_final,
            subject=subject,
            session=session,
            subjects_dir=subjects_dir,
            subject_output_dir=subject_output_dir,
            docker_image=docker_image,
            overwrite=ow,
        )

        key = run_label if run_label else 'run-{:02d}'.format(run_idx)
        all_results[key] = run_results
        print('\n  Run {} completed.'.format(run_idx))

    print('\n' + '=' * 55)
    print('All {} run(s) completed successfully.'.format(len(bold_files)))
    print('Output directory: {}'.format(subject_output_dir))
    print('=' * 55)
    subj_fs_dst = os.path.join(session_work_dir, 'subjects')
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
    req.add_argument('--bids-dir', required=True,
                     help='BIDS directory')
    req.add_argument('--input-file',  required=True,
                     help='Input directory containing SDC-corrected BOLD + SBREF files')
    req.add_argument('--output-file', required=True,
                     help='Output derivatives file')
    req.add_argument('--sub',        required=True,
                     help='Subject label (e.g. sub-01)')

    p.add_argument('--ses',          default='ses-01',
                   help='Session label')

    def str2bool(v):
        if v.lower() in ('true', '1', 'yes'):
            return True
        elif v.lower() in ('false', '0', 'no'):
            return False
        raise argparse.ArgumentTypeError(
            "Boolean value expected, got '{}'".format(v))

    p.add_argument('--subjects-dir', default=None,
                   help='FreeSurfer SUBJECTS_DIR (default: $SUBJECTS_DIR)')
    p.add_argument('--docker',
                   default=os.environ.get('NEURO_IMAGE', 'local'),
                   help='Docker image for FSL/FreeSurfer tools, or "local"')

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

    if args.overwrite_all:
        overwrite = {k: True for k in STEP_KEYS}
    else:
        overwrite = {k: (k in args.overwrite) for k in STEP_KEYS}
    args.sub = "sub-" + args.sub.removeprefix("sub-")
    args.ses = "ses-" + args.ses.removeprefix("ses-")
    run_pipeline(
        bids_dir=args.bids_dir,
        input_file=args.input_file,
        output_file=args.output_file,
        subject=args.sub,
        session=args.ses,
        subjects_dir=args.subjects_dir,
        docker_image=args.docker,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()