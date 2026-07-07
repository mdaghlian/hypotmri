#!/usr/bin/env python
"""
s01_gauss_braincoder.py
===============
Run gaussian pRF fitting on surface data using braincoder (tensorflow backend)
1) Grid fit
2) Iterative (gradient descent) fit
3) Save outputs as .csv
4) Save the final (iterative) fit parameters as FreeSurfer .mgh surface files

Braincoder-specific notes
--------------------------
Uses braincoder's GaussianPRF2DWithHRF + ParameterFitter (see
postproc/sZ_bcoder.ipynb for the exploratory notebook this was adapted from).
Output columns are renamed to match the prfpy convention used elsewhere in
the pipeline (mu_x, mu_y, size, beta, baseline, rsq — see
cvl_utils.prfpy_utils.prfpy_params_dict), so downstream CSV consumers
(s02_prf_visualize.ipynb, s04_collate_analyses.py) work unchanged.

Requires the "bcode_mac" conda environment (tensorflow-macos/tensorflow-metal,
braincoder_bprf) — see config/envs/bcode_mac.yml. Apple Silicon Mac only.

Overwrite behaviour
-------------------
Existence is checked against final BIDS-named files in *output_dir*.

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
s01_gauss_braincoder.py \\
    --bids-dir /path/to/bids_dir \\
    --input-file s04_conf_denoised \\
    --output-file s01_gauss_braincoder \\
    --sub 01 \\
    --ses 01 \\
    --task pRFLE \\
    --project hypot \\
    --roi all

"""

import argparse
import glob
import os
opj = os.path.join
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter

from dpu_mini.fs_tools import dpu_load_roi, dpu_load_nverts
from dpu_mini.stats import dpu_coord_convert
from cvl_utils.preproc_func import check_skip
from cvl_utils.prf_utils import (
    raw_ts_to_average_psc,
    get_dm_and_settings,
    cm_to_dva,
)

# ---------------------------------------------------------------------------
# Step keys — one per docstring step
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'psc_average',     # Average & psc all the runs & make a .npy
    'grid_fit',        # Grid fit
    'iter_fit',        # Iterative fit
    'save_mgh',        # Save the iterative fit parameters as .mgh surface files
]

# Parameters saved out to csv / mgh for the gaussian model (prfpy-style naming)
GAUSS_PARAMS = ['mu_x', 'mu_y', 'size', 'beta', 'baseline', 'ecc', 'pol', 'rsq']

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


def _bc_params_to_dict(bc_pars: pd.DataFrame, rsq: np.ndarray, roi_idx: np.ndarray) -> dict:
    """
    Rename braincoder's gaussian parameter columns (x, y, sd, amplitude, baseline)
    to the prfpy naming convention (mu_x, mu_y, size, beta, baseline) used across
    the rest of the pipeline.
    """
    return {
        'index': roi_idx,
        'mu_x': bc_pars['x'].to_numpy(),
        'mu_y': bc_pars['y'].to_numpy(),
        'size': bc_pars['sd'].to_numpy(),
        'beta': bc_pars['amplitude'].to_numpy(),
        'baseline': bc_pars['baseline'].to_numpy(),
        'rsq': np.asarray(rsq),
    }


def _save_surf_mgh(full_data: np.ndarray, n_verts: list, out_path_no_hemi: str) -> None:
    """
    Save a full-brain (LH+RH concatenated) vertex array as a pair of
    FreeSurfer-style per-hemisphere .mgh files (<out_path_no_hemi>_hemi-L.mgh /
    _hemi-R.mgh).
    """
    n_lh = n_verts[0]
    hemi_data = {'L': full_data[:n_lh], 'R': full_data[n_lh:]}
    for hemi, data in hemi_data.items():
        img = nib.freesurfer.mghformat.MGHImage(
            data.astype(np.float32).reshape(-1, 1, 1),
            affine=np.eye(4, dtype=np.float32),
        )
        nib.save(img, f'{out_path_no_hemi}_hemi-{hemi}.mgh')


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
    Run braincoder fitting of the gaussian pRF model

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
    bc_settings = prf_settings.get('bcoder', {})

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
    # Prep braincoder objects + roi mask
    # ------------------------------------------------------------------
    roi_mask = dpu_load_roi(subject, roi, fs_dir)
    roi_idx = np.where(roi_mask)[0]
    print(f'Loading roi {roi}, fitting {roi_mask.sum()} vertices')
    print(f'(which is {roi_mask.mean()*100:.3f}% of all vertices)')

    radius_deg = cm_to_dva(
        size_cm=prf_settings['screen_size_cm'],
        distance_cm=prf_settings['screen_distance_cm'],
    ) / 2
    print(f'Screen radius = {radius_deg:.3f} dva')

    paradigm = np.rollaxis(np.flipud(dm), 2, 0)  # time, y, x
    x_grid, y_grid = np.meshgrid(
        np.linspace(-radius_deg, radius_deg, dm.shape[1]),
        np.linspace(-radius_deg, radius_deg, dm.shape[0]),
    )
    grid_coordinates = np.stack(
        (x_grid.ravel().astype(np.float32), y_grid.ravel().astype(np.float32)), 1
    )

    gmodel = GaussianPRF2DWithHRF(
        grid_coordinates,
        paradigm=paradigm,
        hrf_model=SPMHRFModel(tr=prf_settings['TR']),
        flexible_hrf_parameters=bc_settings.get('flexible_hrf_parameters', False),
    )
    gfitter = ParameterFitter(
        gmodel, psc_data[roi_mask, :].T, gmodel.paradigm,
    )

    # ------------------------------------------------------------------
    # Grid stage
    # ------------------------------------------------------------------
    grid_points = bc_settings.get('grid_points', 20)
    grid_csv = opj(
        subject_output_dir,
        f'{subject}_{session}_roi-{roi}_task-{task}_model-gauss-bc_stage-grid.csv')
    if not check_skip(
        {'grid_fit': grid_csv},
        ow['grid_fit'],
        'grid_fit',
        force_skip=sk['grid_fit'],
    ):
        ggp = gfitter.fit_grid(
            x=np.linspace(-radius_deg, radius_deg, grid_points),
            y=np.linspace(-radius_deg, radius_deg, grid_points),
            sd=np.linspace(0.1, radius_deg, grid_points),
            amplitude=[1.0],
            baseline=[0.0],
            use_correlation_cost=True,
        )
        ggp = gfitter.refine_baseline_and_amplitude(ggp)
        grid_rsq = np.asarray(gfitter.get_rsq(ggp))

        grid_pd = pd.DataFrame(_bc_params_to_dict(ggp, grid_rsq, roi_idx))
        grid_pd['ecc'], grid_pd['pol'] = dpu_coord_convert(
                grid_pd['mu_x'], grid_pd['mu_y'], 'cart2pol')
        grid_pd.to_csv(grid_csv)
        grid_init_pars = ggp
    else:
        grid_pd = pd.read_csv(grid_csv)
        grid_init_pars = pd.DataFrame({
            'x': grid_pd['mu_x'].to_numpy(),
            'y': grid_pd['mu_y'].to_numpy(),
            'sd': grid_pd['size'].to_numpy(),
            'baseline': grid_pd['baseline'].to_numpy(),
            'amplitude': grid_pd['beta'].to_numpy(),
        })
    print(f'Mean r2 = {grid_pd["rsq"].mean():.3f}')

    # ------------------------------------------------------------------
    # iter stage
    # ------------------------------------------------------------------
    iter_csv = opj(
        subject_output_dir,
        f'{subject}_{session}_roi-{roi}_task-{task}_model-gauss-bc_stage-iter.csv')
    if not check_skip(
        {'iter_fit': iter_csv},
        ow['iter_fit'],
        'iter_fit',
        force_skip=sk['iter_fit'],
    ):
        gpars = gfitter.fit(
            init_pars=grid_init_pars,
            max_n_iterations=bc_settings.get('max_iterations', 1000),
            learning_rate=bc_settings.get('learning_rate', 0.1),
            **bc_settings.get('fitter_args', {}),
        )
        iter_rsq = np.asarray(gfitter.r2)

        iter_pd = pd.DataFrame(_bc_params_to_dict(gpars, iter_rsq, roi_idx))
        iter_pd['ecc'], iter_pd['pol'] = dpu_coord_convert(
                iter_pd['mu_x'], iter_pd['mu_y'], 'cart2pol')
        iter_pd.to_csv(iter_csv)
        print(f'Mean r2 = {iter_pd["rsq"].mean():.3f}')
    else:
        iter_pd = pd.read_csv(iter_csv)

    # ------------------------------------------------------------------
    # save iterative fit as .mgh surface files
    # ------------------------------------------------------------------
    mgh_dir = opj(subject_output_dir, 'mgh')
    mgh_base = opj(
        mgh_dir,
        f'{subject}_{session}_roi-{roi}_task-{task}_model-gauss-bc_stage-iter')
    mgh_paths = {
        f'{p}-{hemi}': f'{mgh_base}_param-{p}_hemi-{hemi}.mgh'
        for p in GAUSS_PARAMS for hemi in ('L', 'R')
    }
    if not check_skip(
        mgh_paths,
        ow['save_mgh'],
        'save_mgh',
        force_skip=sk['save_mgh'],
    ):
        os.makedirs(mgh_dir, exist_ok=True)
        n_verts = dpu_load_nverts(subject, fs_dir)
        total_n_vx = int(np.sum(n_verts))
        for p in GAUSS_PARAMS:
            full = np.zeros(total_n_vx, dtype=np.float32)
            full[roi_idx] = iter_pd[p].to_numpy()
            _save_surf_mgh(full, n_verts, f'{mgh_base}_param-{p}')
        print(f'Saved .mgh files to {mgh_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Gaussian PRF fitting for surface data (using braincoder)',
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
            'restore. Useful for omitting optional steps. '
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
