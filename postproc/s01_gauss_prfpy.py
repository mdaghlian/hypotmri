#!/usr/bin/env python
"""
s01_gauss_prfpy.py
===============
Run gaussian fitting on surface data
1) Grid fit
2) Iterative fit
3) Save outputs as .csv & .pkl

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

Usage example
-------------
s01_gauss_prfpy.py \\
    --bids-dir /path/to/bids_dir \\
    --input-file s04_conf_denoised \\
    --output-file s01_gauss_prfpy \\
    --sub 01 \\
    --ses 01 \\
    --task pRFLE \\
    --project hypot

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

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter
from prfpy.rf import gauss2D_iso_cart
from dpu_mini.fs_tools import dpu_load_roi
from dpu_mini.stats import dpu_coord_convert
from cvl_utils.preproc_func import (
    build_output_name,
    check_skip,
    get_labels,
    make_safe_workdir,
    _bold_base,
    _container_path,
    _stage,
    _get_tr,
)

from cvl_utils.prfpy_utils import (
    raw_ts_to_average_psc, 
    filter_for_nans,
    prfpy_params_dict
)
# ---------------------------------------------------------------------------
# Step keys — one per docstring step (Steps 1-3 are one atomic unit)
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'psc_average',     # Average & psc all the runs & make a .npy 
    'grid_fit',        # Grid fit
    'iter_fit',        # Iterative fit
]
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _load_gii_run(task, run_folder):
    hemi_files = {}
    # find *fsnative.gii file in denoise_dir
    hemi_files = glob.glob(os.path.join(run_folder, f'*{task}*fsnative*hemi-L*.gii'))
    hemi_files.sort()
    # load as .np arrays
    run_data = []
    for iR,hL in enumerate(hemi_files):
        hR = hL.replace('hemi-L', 'hemi-R')
        arr_L = np.vstack([i.data for i in nib.load(hL).darrays])
        arr_R = np.vstack([i.data for i in nib.load(hR).darrays])
        arr_LR = np.hstack([arr_L, arr_R]).T
        run_data.append(arr_LR)
    return run_data

def _get_dm_and_settings(task,project):
    postproc_dir = opj(os.environ['PIPELINE_DIR'], 'postproc')
    
    settings_file = glob.glob(opj(postproc_dir, f'project_*{project}*{task}*.yml'))
    if not settings_file:
        settings_file = glob.glob(opj(postproc_dir, f'project_*{project}*.yml'))
        if not settings_file:
            raise FileNotFoundError(
                'No settings files found for {}.  Searched: {}'.format(
                    project, postproc_dir)
            )
    with open(settings_file[0]) as f:
        prf_settings = yaml.safe_load(f)    
    
    dm_file = glob.glob(opj(postproc_dir, f'project_*{project}*{task}*_dm.npy'))
    if not dm_file:
        dm_file = glob.glob(opj(postproc_dir, f'project_*{project}*_dm.npy'))
        if not dm_file:
            raise FileNotFoundError(
                'No dm files found for {}.  Searched: {}'.format(
                    project, postproc_dir)
            )
    dm = np.load(dm_file[0])
    return prf_settings, dm
# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def psc_average(
    subject_input_dir: str,
    psc_file: str,
    task: str,
    prf_settings : dict, 
    ):
    run_data = _load_gii_run(task,subject_input_dir)
    psc_data = raw_ts_to_average_psc(
        run_data,
        baseline=prf_settings.get('psc_baseline', None)
        )
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
    roi : str = 'all',
    overwrite: dict = None,
    skip: dict = None,
) -> dict:
    """
    Run prfpy fitting of gaussian model 

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
    prf_settings, dm = _get_dm_and_settings(task,project)
    
    input_dir     = str(Path(
        os.path.join(bids_dir, 'derivatives', input_file)
    ).resolve())
    output_dir   = str(Path(
        os.path.join(bids_dir, 'derivatives', output_file)
    ).resolve())

    subject_input_dir    = os.path.join(input_dir,     subject, session)
    subject_output_dir   = os.path.join(output_dir,   subject, session)

    os.makedirs(subject_output_dir, exist_ok=True)

    print('-' * 55)
    print(' Surface data input  : {}'.format(input_dir))
    print(' Output      : {}'.format(output_dir))
    print(' Subject     : {}'.format(subject))
    print(' Session     : {}'.format(session))
    print(f'Project {project}, task {task}')
    print(f' Settings    : {prf_settings}')
    print('-' * 55)

    # ------------------------------------------------------------------
    # Get average psc for runs
    # ------------------------------------------------------------------
    psc_file = os.path.join(
        subject_input_dir, f'{subject}_{session}_task-{task}_avg-psc.npy')
    if not check_skip(
        {'psc_average': psc_file},
        ow['psc_average'],
        'psc_average',
        force_skip=sk['psc_average'],
    ):
        psc_data = psc_average(
            subject_input_dir   = subject_input_dir,
            psc_file            = psc_file,
            task                = f'task-{task}',
            prf_settings        = prf_settings,
            )
    else:
        psc_data = np.load(psc_file)
    print(f'Chopping data, removing first {prf_settings["vols_to_chop"]}')
    psc_data=psc_data[:,prf_settings["vols_to_chop"]:]
    # Check times series are correct
    assert dm.shape[-1] == psc_data.shape[-1]
    # ------------------------------------------------------------------
    # Prep prfpy objects + roi mask
    # ------------------------------------------------------------------
    roi_mask = dpu_load_roi(subject, roi, fs_dir)
    roi_idx = np.where(roi_mask)[0]
    print(f'Loading roi {roi}, fitting {roi_mask.sum()} vertices')
    print(f'(which is {roi_mask.mean()*100:.3f}% of all vertices)')
    prf_stim = PRFStimulus2D(
        screen_size_cm=prf_settings['screen_size_cm'],          # Distance of screen to eye
        screen_distance_cm=prf_settings['screen_distance_cm'],  # height of the screen (i.e., the diameter of the stimulated region)
        design_matrix=dm,                                   # dm (npix x npix x time_points)
        TR=prf_settings['TR'],                                  # TR
        )
    print(f'Screen size in degrees of visual angle = {prf_stim.screen_size_degrees}')
    gmodel = Iso2DGaussianModel(
        stimulus=prf_stim,                                  # The stimulus we made earlier
        hrf=prf_settings['hrf']['pars'],                        # These are the parameters for the HRF that we normally use at Spinoza (with 7T data). (we can fit it, this will be done later...)
        filter_predictions = prf_settings['filter_predictions'],# Do you want to filter the predictions? (depends what you did to the data, try and match it... default is not to do anything)
        normalize_RFs= prf_settings['normalize_RFs'],           # Normalize the volume of the RF (so that RFs w/ different sizes have the same volume. Generally not needed, as this can be solved using the beta values i.e.,amplitude)
        # FILTER SETTINGS TO DO
        )
    gfit = Iso2DGaussianFitter(
        data=psc_data[roi_mask,:],  # time series
        model=gmodel,                   # model (see above)
        n_jobs=prf_settings['n_jobs'],  # number of jobs to use in parallelization 
        )
    # keys for gauss pars
    gaussp_keys = prfpy_params_dict()['gauss']
    max_eccentricity = prf_stim.screen_size_degrees/2 # It doesn't make sense to look for PRFs which are outside the stimulated region
    grid_nr = prf_settings['grid_nr'] # Size of the grid (i.e., number of possible PRF models). Higher number means that the grid fit will be more exact, but take longer...
    eccs    = max_eccentricity * np.linspace(
        0.25, 1, grid_nr['ecc'])**2 # Squared because of cortical magnification, more efficiently tiles the visual field...
    sizes   = max_eccentricity * np.linspace(
        0.1, 1, grid_nr['size'])**2  # Possible size values (i.e., sigma in gaussian model) 
    polars  = np.linspace(
        0, 2*np.pi, grid_nr['pol'])              # Possible polar angle coordinates

    # We can also fit the hrf in the same way (specifically the derivative)
    # -> make a grid between 0-10 (see settings file)
    if grid_nr['hrf_1'] == 1:
        # Stick to default hrf
        hrf_1_grid = np.array(prf_settings['hrf']['pars'][1])
    else:
        hrf_1_grid = np.linspace(
            prf_settings['hrf']['deriv_bound'][0], 
            prf_settings['hrf']['deriv_bound'][1], 
            grid_nr['hrf_1'])
    # We generally recommend to fix the dispersion value to 0
    hrf_2_grid = np.array([0.0])
    # Amplitude bounds for gauss grid fit - set [min, max]
    gauss_grid_bounds = [[prf_settings['prf_ampl'][0],prf_settings['prf_ampl'][1]]] 
    gauss_bounds = [
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # x bound
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # y bound
        (1e-1, max_eccentricity*3),                             # prf size bounds
        (prf_settings['prf_ampl'][0],prf_settings['prf_ampl'][1]),      # prf amplitude
        (prf_settings['bold_bsl'][0],prf_settings['bold_bsl'][1]),      # bold baseline (fixed)
        (prf_settings['hrf']['deriv_bound'][0], prf_settings['hrf']['deriv_bound'][1]), # hrf_1 bound
        (prf_settings['hrf']['disp_bound'][0], prf_settings['hrf']['disp_bound'][1]), # hrf_2 bound
    ]
    

    # *** Save your fitting parameters ***
    # We may run our analysis several times. If so we want to save the important information all together
    # We will use a pickle file to do this.
    prf_settings['max_eccentricity']  = max_eccentricity
    prf_settings['eccs'] = eccs
    prf_settings['sizes'] = sizes
    prf_settings['polars'] = polars
    prf_settings['hrf_1_grid'] = hrf_1_grid
    prf_settings['hrf_2_grid'] = hrf_2_grid
    prf_settings['gauss_bounds'] = gauss_bounds
    prf_settings['gauss_grid_bounds'] = gauss_grid_bounds

    # ------------------------------------------------------------------
    # Grid stage
    # ------------------------------------------------------------------
    grid_csv = opj(
        subject_output_dir, 
        f'{subject}_{session}_roi-{roi}_task-{task}_model-gauss_stage-grid.csv')
    if not check_skip(
        {'grid_fit': grid_csv},
        ow['grid_fit'],
        'grid_fit',
        force_skip=sk['grid_fit'],
    ):

        gfit.grid_fit(
            ecc_grid=eccs,
            polar_grid=polars,
            size_grid=sizes,
            hrf_1_grid=hrf_1_grid,
            hrf_2_grid=hrf_2_grid,
            verbose=True,
            n_batches=prf_settings['n_batches'],               # The grid fit is performed in parallel over n_batches of units.Batch parallelization is faster than single-unit parallelization and of sequential computing.
            fixed_grid_baseline=prf_settings['fixed_grid_baseline'], # Fix the baseline? This makes sense if we have fixed the baseline in preprocessing
            grid_bounds=gauss_grid_bounds
            )
        # Sometimes the fits are bad and will return NaN values. We do not want this so will remove them here:
        gfit.gridsearch_params = filter_for_nans(gfit.gridsearch_params)
        gfit.gridsearch_r2 = filter_for_nans(gfit.gridsearch_r2)

        grid_dict = {}
        grid_dict['index'] = roi_idx
        for key in gaussp_keys.keys():
            grid_dict[key] = gfit.gridsearch_params[:,gaussp_keys[key]]
        grid_pd = pd.DataFrame(grid_dict)
        grid_pd.to_csv(grid_csv)
        grid_pars_np = gfit.gridsearch_params
    else:
        grid_pd = pd.read_csv(grid_csv)
        grid_pars_np = np.zeros((roi_mask.sum(), 8))
        for key in gaussp_keys.keys():
            grid_pars_np[:,gaussp_keys[key]] = grid_pd[key].to_numpy()

    print(f'Mean r2 = {grid_pd["rsq"].mean():.3f}')

    # ------------------------------------------------------------------
    # iter stage
    # ------------------------------------------------------------------
    iter_csv = opj(
        subject_output_dir, 
        f'{subject}_{session}_roi-{roi}_task-{task}_model-gauss_stage-iter.csv')
    if not check_skip(
        {'iter_fit': iter_csv},
        ow['iter_fit'],
        'iter_fit',
        force_skip=sk['iter_fit'],
    ):
        gfit.iterative_fit(
            rsq_threshold=prf_settings['rsq_threshold'],    # Minimum variance explained. Puts a lower bound on the quality of PRF fits. Any fits worse than this are thrown away...     
            verbose=True,
            bounds=gauss_bounds,       # Bounds (on parameters)
            )               
        # Sometimes the fits are bad and will return NaN values. We do not want this so will remove them here:
        gfit.iterative_search_params = filter_for_nans(gfit.iterative_search_params)

        iter_dict = {}
        iter_dict['index'] = roi_idx
        for key in gaussp_keys.keys():
            iter_dict[key] = gfit.iterative_search_params[:,gaussp_keys[key]]
        iter_pd = pd.DataFrame(iter_dict)
        iter_pd.to_csv(iter_csv)
        print(f'Mean r2 = {iter_pd["rsq"].mean():.3f}')
    

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Gaussian PRF fitting for surface data (using prfpy)',
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
                   help='task label for prf', required=True)
    p.add_argument('--project', type=str, 
                   help='project for selecting the settings file', required=True)
    p.add_argument('--ses',
                   help='Session label (e.g. ses-01)', required=True)
    p.add_argument('--roi', default='all',
                   help='ROI label (what to filter with fs labels)', required=True)

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
        roi             = args.roi,
        overwrite       = overwrite,
        skip            = skip,
    )


if __name__ == '__main__':
    main()