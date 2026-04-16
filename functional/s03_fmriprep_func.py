#!/usr/bin/env python
#$ -V
#$ -cwd
"""
run_fmriprep_confounds.py
=========================
Run fMRIPrep on preprocessed BOLD data to extract confounds.

Pipeline overview
-----------------
    Step 1  - Copy preprocessed BOLD files into FPREP_BIDS with standardised naming
    Step 2  - Write per-run JSON sidecar (RepetitionTime, TaskName)
    Step 3  - Run fMRIPrep via Docker or Apptainer/Singularity

Usage example
-------------
python run_fmriprep_confounds.py \\
    --bids-dir   /data/bids \\
    --sub        sub-01 \\
    --ses        ses-01 \\
    --input-file my_preprocessed_deriv
"""

import argparse
import json
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
    _bold_base, 
    _get_tr
)



# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def stage_bold_files(
    bold_files: list[str],
    fprep_func_dir: str,
    subject: str,
    session: str,
) -> list[dict]:
    """
    Copy each BOLD run into *fprep_func_dir* with a clean BIDS name and
    write the accompanying JSON sidecar.  Returns a list of metadata dicts.
    """
    staged = []
    for bold_file in bold_files:
        basename = os.path.basename(bold_file)
        print('  Processing: {}'.format(basename))

        
        run_label, task_label = get_labels(bold_file)
        stem      = f'{subject}_{session}_{task_label}_{run_label}_bold'
        dest_nii  = os.path.join(fprep_func_dir, stem + '.nii.gz')
        dest_json = os.path.join(fprep_func_dir, stem + '.json')
        print('    -> {}'.format(stem + '.nii.gz'))

        tr = _get_tr(bold_file)
        sidecar = {
            'RepetitionTime': tr,
            'TaskName': task_label.split('-')[-1],
        }
        with open(dest_json, 'w') as fh:
            json.dump(sidecar, fh, indent=2)

        shutil.copy(bold_file, dest_nii)

        staged.append({
            'source':    bold_file,
            'dest_nii':  dest_nii,
            'dest_json': dest_json,
            'task':      task_label,
            'run':       run_label,
            'tr':        tr,
        })

    return staged


def run_fmriprep(
    fprep_bids_dir: str,
    bids_dir: str,
    fprep_bids_dir_wf: str,
    subject: str,
    container_type: str,
    fprep_image: str = '',
    subjects_dir: str = '',
    pipeline_dir: str = '',
    sif_dir: str = '',
    fprep_sif: str = '',
) -> None:
    """Launch fMRIPrep inside Docker or Apptainer/Singularity."""
    fmriprep_out = os.path.join(bids_dir, 'derivatives', 'fmriprep')
    Path(fmriprep_out).mkdir(parents=True, exist_ok=True)

    license_file = os.path.join(pipeline_dir, 'config', 'license.txt')

    # Shared fMRIPrep arguments (everything after the positional args)
    fmriprep_args = [
        '/data', '/out', 'participant',
        '--participant-label', subject,
        '--skip_bids_validation',
        '--fs-subjects-dir', '/fsdir',
        '--fs-license-file', '/license.txt',
        '--work-dir', '/work',
        '--omp-nthreads', '4',
        '--nprocs', '4',
        '--ignore', 'fieldmaps', 'slicetiming',
        '--output-spaces', 'func',
    ]

    if container_type == 'docker':
        cmd = [
            'docker', 'run', '--rm',
            '-v', '{}:/data:ro'.format(fprep_bids_dir),
            '-v', '{}:/out'.format(fmriprep_out),
            '-v', '{}:/work'.format(fprep_bids_dir_wf),
            '-v', '{}:/fsdir'.format(subjects_dir),
            '-v', '{}:/license.txt'.format(license_file),
            fprep_image,
        ] + fmriprep_args

    elif container_type in ('apptainer', 'singularity'):
        sif_path = os.path.join(sif_dir, fprep_sif)
        cmd = [
            container_type, 'run',
            '--cleanenv',
            '-B', '{}:/data'.format(fprep_bids_dir),
            '-B', '{}:/out'.format(fmriprep_out),
            '-B', '{}:/work'.format(fprep_bids_dir_wf),
            '-B', '{}:/fsdir'.format(subjects_dir),
            '-B', '{}:/license.txt'.format(license_file),
            sif_path,
        ] + fmriprep_args

    else:
        raise ValueError('Invalid CONTAINER_TYPE: {}'.format(container_type))

    print('Running: {}'.format(' '.join(cmd)))
    subprocess.run(cmd, check=True)


def run_pipeline(
    bids_dir: str,
    input_file: str,
    subject: str,
    session: str,
) -> None:
    """
    Full pipeline: stage BOLD files then launch fMRIPrep.
    Container configuration is read from environment variables.
    """
    bids_dir = str(Path(bids_dir).resolve())

    input_dir         = os.path.join(bids_dir, 'derivatives', input_file)
    subject_input_dir = os.path.join(input_dir, subject, session)

    if not Path(subject_input_dir).is_dir():
        raise FileNotFoundError(
            'Input directory not found: {}'.format(subject_input_dir))

    print('-' * 55)
    print('Running fmriprep - to get the confounds')
    print('-' * 55)
    print(' Input   : {}'.format(input_dir))
    print(' Output  : {}'.format(bids_dir))
    print(' Subject : {}'.format(subject))
    print(' Session : {}'.format(session))
    print('-' * 55)

    # ------------------------------------------------------------------
    # Step 1 — Prepare staging directories
    # ------------------------------------------------------------------
    fprep_bids_dir = os.path.join(bids_dir, 'derivatives', 'FPREP_BIDS')
    Path(fprep_bids_dir).mkdir(parents=True, exist_ok=True)
    fprep_bids_json = os.path.join(fprep_bids_dir, "dataset_description.json")
    if not os.path.exists(fprep_bids_json):
        data = {
            "Name": "Example dataset",
            "BIDSVersion": "1.0.2"
        }    
        with open(fprep_bids_json, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Created {fprep_bids_json}")

    fprep_bids_dir_wf = os.path.join(bids_dir, 'derivatives', 'FPREP_BIDS_WF')
    Path(fprep_bids_dir_wf).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2 — Discover BOLD files
    # ------------------------------------------------------------------
    pattern = '{}_{}*space-fsT1_*.nii*'.format(subject, session)
    matches = sorted(Path(input_dir).rglob(pattern))
    bold_files = [str(p) for p in matches]

    if not bold_files:
        raise FileNotFoundError(
            'No BOLD files found for {}_{} in {}'.format(subject, session, input_dir))

    print('Found {} run(s) to process'.format(len(bold_files)))
    print('BOLD files:')
    for f in bold_files:
        print('  - {}'.format(os.path.basename(f)))

    # ------------------------------------------------------------------
    # Step 3 — Stage BOLD files into FPREP_BIDS
    # ------------------------------------------------------------------
    fprep_func_dir = os.path.join(fprep_bids_dir, subject, session, 'func')
    Path(fprep_func_dir).mkdir(parents=True, exist_ok=True)

    print('\nCopying BOLD files into FPREP_BIDS...')
    stage_bold_files(bold_files, fprep_func_dir, subject, session)

    # ------------------------------------------------------------------
    # Step 4 — Run fMRIPrep
    # ------------------------------------------------------------------
    print('\n' + '=' * 42)
    print('Copied it all over - now for fmriprep')
    print('=' * 42)

    container_type = os.environ.get('CONTAINER_TYPE', '')
    run_fmriprep(
        fprep_bids_dir=fprep_bids_dir,
        bids_dir=bids_dir,
        fprep_bids_dir_wf=fprep_bids_dir_wf,
        subject=subject,
        container_type=container_type,
        fprep_image=os.environ.get('FPREP_IMAGE', ''),
        subjects_dir=os.environ.get('SUBJECTS_DIR', ''),
        pipeline_dir=os.environ.get('PIPELINE_DIR', ''),
        sif_dir=os.environ.get('SIF_DIR', ''),
        fprep_sif=os.environ.get('FPREP_SIF', ''),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Run fMRIPrep on preprocessed BOLD data to extract confounds.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',    required=True, help='Path to BIDS root directory')
    req.add_argument('--input-file',  required=True,
                     help='Name of derivatives folder containing preprocessed BOLD')
    req.add_argument('--sub',         required=True, help='Subject label (e.g. sub-01)')
    req.add_argument('--ses',         required=True, help='Session label (e.g. ses-01)')
    p.add_argument('--help-env', action='store_true',
                   help='Print expected environment variables and exit')
    return p


def print_env_help() -> None:
    print("""
Expected environment variables
-------------------------------
CONTAINER_TYPE   docker | apptainer | singularity
FPREP_IMAGE      Docker image tag        (docker only)
FPREP_SIF        Singularity .sif file   (apptainer/singularity only)
SIF_DIR          Directory containing the .sif file
SUBJECTS_DIR     FreeSurfer subjects directory
PIPELINE_DIR     Pipeline root (must contain config/license.txt)
""")


def main() -> None:
    args = _build_parser().parse_args()

    if args.help_env:
        print_env_help()
        return

    subject = 'sub-' + args.sub.removeprefix('sub-')
    session = 'ses-' + args.ses.removeprefix('ses-')

    run_pipeline(
        bids_dir=args.bids_dir,
        input_file=args.input_file,
        subject=subject,
        session=session,
    )


if __name__ == '__main__':
    main()