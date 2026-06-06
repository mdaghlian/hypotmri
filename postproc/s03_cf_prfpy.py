#!/usr/bin/env python
"""
s03_cf_prfpy.py
===============
Run connective field model on surface data using prfpy.
-> Per hemisphere
1) Convert time series to % signal change and concatenate them
2) Calculate the pairwise geodesic distance matrices for the source region
3) Run the grid fit for different connective field sizes & calculate the betas + baselines
4) Save outputs as a .csv file


Overwrite behaviour
-------------------
Existence is checked against final BIDS-named files in *output_dir*.
Skipped steps restore outputs to *work_dir* so downstream steps can proceed.

Skip behaviour
--------------
Steps can be force-skipped with --skip regardless of whether their outputs
exist. No files are touched and no outputs are restored to work_dir.
Downstream steps continue regardless — caller is responsible for dependencies.

Arguments:
    --bids-dir      BIDS directory containing input and output derivatives
    --input-file    derivatives/<dir> with surface data 
    --output-file   Name of derivatives directory to write prf outputs
    --sub           Subject label (e.g. sub-01)
    --ses           Session label (e.g. ses-01)
    --task          task label (e.g., pRFLE)
    --project       used to find *.yml & dm.npy inside the postproc dir
    --roi-src       fs label to use as source for CF model (default: b14_V1.)
    --roi-target    fs label to use as target for CF model (default: all)

Usage example
-------------
s03_cf_prfpy.py \\
    --bids-dir /path/to/bids_dir \\
    --input-file s04_conf_denoised \\
    --output-file s03_cf_prfpy \\
    --sub 01 \\
    --ses 01 \\
    --task pRFLE \\
    --project hypot
    --roi-src b14_V1 \\
    --roi-target b14_ALL 

"""

import argparse
import glob
import os
opj = os.path.join
import re
import shutil
from pathlib import Path
import yaml

import nibabel as nib
import numpy as np
import pandas as pd


from dpu_mini.fs_tools import dpu_load_roi
from dpu_mini.mesh_maker import GenMeshMaker
from dpu_mini.mesh_format import dpu_pairwise_geodesic_distance
from dpu_mini.stats import dpu_coord_convert


from prfpy.stimulus import CFStimulus 
from prfpy.model import CFGaussianModel
from prfpy.fit import CFFitter

from cvl_utils.preproc_func import (
    check_skip,
)

from cvl_utils.prfpy_utils import (
    raw_ts_to_average_psc, 
    filter_for_nans,
    prfpy_params_dict, 
    get_dm_and_settings
)
# ---------------------------------------------------------------------------
# Step keys — one per docstring step (Steps 1-3 are one atomic unit)
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'psc_concat',     # psc transform all the runs & concatenate
    'gdist',          # Calculate the pairwise geodesic distance for vertices in the src region
    'grid_fit',        # Grid fit
]
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _load_gii_run(task, run_folder):
    hemi_files = {}
    # find *fsnative.gii file in denoise_dir
    hemi_files = glob.glob(os.path.join(run_folder, f'*{task}*fsnative*hemi-L*.gii'))
    hemi_files.sort()
    print(run_folder)
    # load as .np arrays
    run_data = []
    for iR,hL in enumerate(hemi_files):
        hR = hL.replace('hemi-L', 'hemi-R')
        arr_L = np.vstack([i.data for i in nib.load(hL).darrays])
        arr_R = np.vstack([i.data for i in nib.load(hR).darrays])
        arr_LR = np.hstack([arr_L, arr_R]).T
        run_data.append(arr_LR)
    return run_data

# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def psc_concat(
    subject_input_dir: str,
    psc_file: str,
    task: str,
    prf_settings : dict, 
    ):
    run_data = _load_gii_run(task,subject_input_dir)
    psc_data = []
    for run in run_data:
        psc_chop = raw_ts_to_average_psc(
            run,
            baseline=prf_settings.get('psc_baseline', None)
            )[:,prf_settings["vols_to_chop"]:]
        psc_data.append(psc_chop)
    psc_data = np.concatenate(psc_data, axis=-1)
    np.save(psc_file, psc_data)
    return psc_data
# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    bids_dir: str,
    fs_dir: str,
    input_file: str,
    output_file: str,
    subject: str,
    session: str,
    task: str,
    project: str,
    roi_src : str,
    roi_target : str,
    overwrite: dict = None,
    skip: dict = None,
) -> dict:
    """
    Run conncetive field modelling with prfpy
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

    sk = {k: False for k in STEP_KEYS}
    if skip:
        unknown = set(skip) - set(STEP_KEYS)
        if unknown:
            raise ValueError(
                'Unknown skip key(s): {}.  Valid keys: {}'.format(
                    sorted(unknown), STEP_KEYS)
            )
        sk.update(skip)
    
    prf_settings, dm = get_dm_and_settings(task,project)
    input_dir     = str(Path(
        os.path.join(bids_dir, 'derivatives', input_file)
    ))
    output_dir   = str(Path(
        os.path.join(bids_dir, 'derivatives', output_file)
    ))

    subject_input_dir    = os.path.join(input_dir,     subject, session)
    subject_output_dir   = os.path.join(output_dir,   subject, session)

    os.makedirs(subject_output_dir, exist_ok=True)
    # ------------------------------------
    # ------------------------------------
    # HEMISPHERE LOOP 
    # ------------------------------------
    # ------------------------------------
    hemi_pd = []
    for hemi in ['lh', 'rh']:
        print('-' * 55)
        print(f' Surface data input  : {input_dir}')
        print(f' Output      : {output_dir}')
        print(f' Subject     : {subject}')
        print(f' Session     : {session}')
        print(f' roi target  : {roi_target}')
        print(f' roi src     : {roi_src}')
        print(f' hemi        : {hemi}')
        print(f' ')
        print(f'Project {project}, task {task}')
        print(f' Settings    : {prf_settings}')
        print('-' * 55)

        # ------------------------------------------------------------------
        # Get runs (concatendated + psc changed) 
        # ------------------------------------------------------------------
        psc_file = os.path.join(
            subject_input_dir, f'{subject}_{session}_task-{task}_cat-psc.npy')
        if not check_skip(
            {'psc_concat': psc_file},
            ow['psc_concat'],
            'psc_concat',
            force_skip=sk['psc_concat'],
        ):
            psc_data = psc_concat(
                subject_input_dir   = subject_input_dir,
                psc_file            = psc_file,
                task                = f'task-{task}',
                prf_settings        = prf_settings,
                )
        else:
            psc_data = np.load(psc_file)

        # ------------------------------------------------------------------
        # Get roi masks
        # ------------------------------------------------------------------
        roi_src_mask = dpu_load_roi(subject, f'{hemi}.{roi_src}', fs_dir)
        roi_target_mask = dpu_load_roi(subject, f'{hemi}.{roi_target}', fs_dir)
        # remove src vertices from target
        roi_target_mask[roi_src_mask] = False
        # Create mesh object, useful for performing distance calculations
        gm = GenMeshMaker(subject, fs_dir)

        # ------------------------------------------------------------------
        # Get geodesic distance & appropriate masks
        # ------------------------------------------------------------------
        gdist_file = os.path.join(
            subject_input_dir, f'{subject}_gdist_hemi-{hemi}_roi_src-{roi_src}.npy')
        if not check_skip(
            {'gdist': gdist_file},
            ow['gdist'],
            'gdist',
            force_skip=sk['gdist'],
        ):
            gdist = dpu_pairwise_geodesic_distance(
                gm.mesh_info['pial'][hemi], 
                roi_src_mask[gm.hemi_masks[hemi]], 
                m=100.0
                )
            np.save(gdist_file, gdist)
        else:
            gdist = np.load(gdist_file)

        # ------------------------------------------------------------------
        # Prep prfpy objects 
        # ------------------------------------------------------------------
        ts_target = psc_data[roi_target_mask, :]
        cf_stim = CFStimulus(
            data=psc_data, 
            vertinds=np.where(roi_src_mask)[0],
            distances=gdist, 
        )    
        cf_model = CFGaussianModel(cf_stim)
        cf_fitter = CFFitter(model=cf_model, data=ts_target)
        sigma_grid = np.array(prf_settings['sigma_grid'])
        print(f'Running CF fitting with sigmas = {sigma_grid}')
        print(f'Number of vertices in source = {roi_src_mask.sum()}')
        print(f'Number of vertices in target = {roi_target_mask.sum()}')

        # ------------------------------------------------------------------
        # Grid stage
        # ------------------------------------------------------------------
        cf_keys = prfpy_params_dict()['cf']
        grid_csv = opj(
            subject_output_dir, 
            f'{subject}_{session}_roisrc-{roi_src}_roitarget-{roi_target}_task-{task}_model-cf_hemi-{hemi}.csv')
        if not check_skip(
            {'grid_fit': grid_csv},
            ow['grid_fit'],
            'grid_fit',
            force_skip=sk['grid_fit'],
        ):

            cf_fitter.grid_fit(
                sigma_grid=sigma_grid,
                pos_prfs_only=True, 
                )
            # Sometimes the fits are bad and will return NaN values. We do not want this so will remove them here:
            cf_fitter.gridsearch_params = filter_for_nans(cf_fitter.gridsearch_params)

            grid_dict = {}
            grid_dict['index'] = np.where(roi_target_mask)[0]
            for key in cf_keys.keys():
                grid_dict[key] = cf_fitter.gridsearch_params[:,cf_keys[key]]
            grid_pd = pd.DataFrame(grid_dict)
            grid_pd.to_csv(grid_csv)
        else:
            grid_pd = pd.read_csv(grid_csv)

        print(f'Mean r2 = {grid_pd["rsq"].mean():.3f}')
        hemi_pd.append(grid_pd)

    # Sta
    grid_csv = opj(
        subject_output_dir, 
        f'{subject}_{session}_roisrc-{roi_src}_roitarget-{roi_target}_task-{task}_model-cf.csv')
    
    both_hemi_pd = pd.concat(hemi_pd)
    if os.path.exists(grid_csv):
        os.unlink(grid_csv)
    both_hemi_pd.to_csv(grid_csv)
    print('Saved both hemispheres together')
    


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Connective field fitting for surface data (using prfpy)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',    required=True,
                     help='BIDS directory')
    req.add_argument('--input-file',   required=True,
                     help='Surface time series data')
    req.add_argument('--output-file', required=True,
                     help='Where to put the fits')
    req.add_argument('--sub',         required=True,
                     help='Subject label (e.g. sub-01 or 01)')
    p.add_argument('--task', type=str,
                   help='task label', required=True)
    p.add_argument('--project', type=str, 
                   help='project for selecting the settings file', required=True)
    p.add_argument('--ses',
                   help='Session label (e.g. ses-01)', required=True)
    p.add_argument('--roi-src', default='b14_V1.',
                   help='ROI source region (default benson 14 V1)', required=True)
    p.add_argument('--roi-target', default='b14_ALL', 
                   help='ROI of target region: default benson 14 ALL', required=True)

    ow_group = p.add_argument_group(
        'overwrite / skip options',
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
    ow_group.add_argument(
        '--skip',
        nargs='+',
        metavar='STEP',
        default=[],
        choices=STEP_KEYS,
        help=(
            'Hard-skip one or more named steps — no file checks, no work_dir '
            'restore. Useful for omitting optional steps (e.g. denoise_surf). '
            'Downstream steps continue regardless; caller is responsible for '
            'any missing dependencies.'
        ),
    )

    return p


def main():
    args = _build_parser().parse_args()

    if args.overwrite_all:
        overwrite = {k: True for k in STEP_KEYS}
    else:
        overwrite = {k: (k in args.overwrite) for k in STEP_KEYS}

    skip = {k: (k in args.skip) for k in STEP_KEYS}

    args.sub = 'sub-' + args.sub.removeprefix('sub-')
    args.ses = 'ses-' + args.ses.removeprefix('ses-')
    
    run_pipeline(
        bids_dir        = args.bids_dir,
        fs_dir          = opj(args.bids_dir, 'derivatives', 'freesurfer'),
        input_file      = args.input_file,
        output_file     = args.output_file,
        subject         = args.sub,
        session         = args.ses,
        task            = args.task,
        project         = args.project,
        roi_src         = args.roi_src,
        roi_target      = args.roi_target,
        overwrite       = overwrite,
        skip            = skip,
    )


if __name__ == '__main__':
    main()