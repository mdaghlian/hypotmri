"""
coreg.py
========
Reusable registration / motion-correction step functions shared by the
functional coregistration pipeline (functional/s02_coreg.py).

Each function is a single pipeline step: it takes explicit host paths plus
a work_dir/docker_image, runs one or more FSL/FreeSurfer commands via
run_cmd, and returns the resulting host path(s). No argparse, output-name
bookkeeping, or overwrite/skip logic lives here — that stays in the
orchestrating pipeline script.
"""

import glob
import os
opj = os.path.join
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from cvl_utils.preproc_func import (
    build_output_name,
    run_cmd,
    run_local,
    fsl_val,
    _stage,
    _container_path,
)

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
