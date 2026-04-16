#!/usr/bin/env python
"""
s04_confounds.py
===============
Denoise fMRI data using confounds
1) Extract the noise components output from fMRIprep (e.g. *confounds.tsv file)
- remove all of the motion related ones
2) Extract motion confounds from mcflirt output:
- Motion parameters (6 dof)
- FD (using nipype + .par files from mcflirt)
- DVARS (using nipype + .par files from mcflirt)
- Use brainmask, also from fmriprep
- ? Spikes ? Derivatives ? Quadratic terms ? - not included for now...
3) Save them all together in a single .tsv file per run, with standardized naming:
sub-XX_ses-XX_task-XX_run-XX_desc-confounds.tsv
4) Run PCA on these confounds, selecting the top N components:
- High pass filter these PCA components to remove slow drifts (using a SG filter)
- Save these in a separate .tsv file: sub-XX_ses-XX_task-XX_run-XX_desc-pca_confounds.tsv
5) Do the denoising regression:
- High pass filter the data with the same filter
- Regress out these PCA components from the data, saving the result as _desc-denoised_bold.nii.gz
- The data to be denoised are the sdc+moco volume data & the surface projected data

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
    --moco-file     derivatives/<dir> with moco-files (and the .par files)
    --output-file   Name of derivatives directory to write motion confounds to (e.g. s4_motion_confounds)
    --sub           Subject label (e.g. sub-01)
    --ses           Session label (e.g. ses-01)
    --n-pca         Number of PCA components to retain (default: 6)
    --sg-window     Savitzky-Golay filter window length in SECONDS (default: 120)
    --sg-order      Savitzky-Golay filter polynomial order (default: 3)
    --filter-only   Skip confound extraction/PCA; apply SG high-pass filter only.
                    Outputs saved as _desc-filtered_bold (vs _desc-denoised_bold).
                    Useful for isolating the contribution of filtering vs denoising.

Usage example
-------------
python s04_confounds.py \\
    --bids-dir /path/to/bids_dir \\
    --moco-file s03_motion_correction \\
    --output-file s04_motion_confounds \\
    --sub 01 \\
    --ses 01

# Filter-only mode (no confound regression):
python s04_confounds.py \\
    --bids-dir /path/to/bids_dir \\
    --moco-file s03_motion_correction \\
    --output-file s04_motion_confounds \\
    --sub 01 \\
    --ses 01 \\
    --filter-only

# Skip specific steps (hard skip — no file checks, no work_dir restore):
python s04_confounds.py \\
    --bids-dir /path/to/bids_dir \\
    --moco-file s03_motion_correction \\
    --output-file s04_motion_confounds \\
    --sub 01 \\
    --ses 01 \\
    --skip denoise_surf filter_surf
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
import pandas as pd
from nipype.algorithms.confounds import FramewiseDisplacement, ComputeDVARS

from scipy.signal import savgol_filter
from scipy import stats
from sklearn.decomposition import PCA

from cvl_utils.preproc_func import (
    build_output_name,
    check_skip,
    get_labels,
    make_safe_workdir,
    _bold_base,
    _container_path,
    _stage,
    run_cmd,
    _get_tr,
)
from cvl_utils.denoise import (
    PCA_denoiser, 
    SGFilter,
)

# ---------------------------------------------------------------------------
# Step keys — one per docstring step (Steps 1-3 are one atomic unit)
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'confounds',       # Steps 1-3: fMRIprep noise regressors + mcflirt motion → merged .tsv
    'pca_confounds',   # Step 4:    PCA + SG high-pass filter → pca_confounds.tsv
    'denoise_vol',     # Step 5a:   SG high-pass + OLS regression on volumetric BOLD
    'denoise_surf',    # Step 5b:   same for surface projections
    'filter_vol',      # Filter-only 5a: SG high-pass only on volumetric BOLD (no regression)
    'filter_surf',     # Filter-only 5b: same for surface projections
]

# ---------------------------------------------------------------------------
# Motion-related column name patterns to REMOVE from fMRIprep confounds
# (Step 1: keep only non-motion noise regressors)
# ---------------------------------------------------------------------------

MOTION_PATTERNS = [
    r'^trans_[xyz]',
    r'^rot_[xyz]',
    r'^framewise_displacement',
    r'^dvars',
    r'^std_dvars',
    r'^rmsd',
    r'^motion_outlier',
]

# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def extract_fmriprep_confounds(
    fmriprep_confounds_tsv: str,
) -> pd.DataFrame:
    """
    Step 1: Load the fMRIprep *_desc-confounds_timeseries.tsv and remove all
    motion-related columns, retaining noise regressors such as aCompCor,
    tCompCor, CSF, white matter, etc.

    Parameters
    ----------
    fmriprep_confounds_tsv : path to the fMRIprep confounds .tsv

    Returns
    -------
    pd.DataFrame with motion columns removed.
    """
    df = pd.read_csv(fmriprep_confounds_tsv, sep='\t', na_values='n/a')

    motion_cols = [
        col for col in df.columns
        if any(re.match(pat, col) for pat in MOTION_PATTERNS)
    ]
    noise_df = df.drop(columns=motion_cols)

    print('  fMRIprep confounds: kept {} / {} columns '
          '(dropped {} motion columns)'.format(
              len(noise_df.columns), len(df.columns), len(motion_cols)))
    print('dropping')
    for c in motion_cols:
        print(c)
    return noise_df


def extract_mcflirt_confounds(
    mcf_nii: str,
    mcf_par_file: str,
    brainmask: str,
    work_dir: str,
) -> pd.DataFrame:
    """
    Step 2: Extract motion confounds from MCFLIRT output.

    Columns returned
    ----------------
    rot_x, rot_y, rot_z, trans_x, trans_y, trans_z — from .par file (FSL order)
    framewise_displacement                          — FD (mm) via nipype
    dvars                                           — standardised DVARS via nipype,
                                                      using the fmriprep brainmask

    Parameters
    ----------
    mcf_nii      : motion-corrected 4-D BOLD NIfTI (used for DVARS)
    mcf_par_file : MCFLIRT .par file (6 DOF, FSL convention)
    brainmask    : brain mask NIfTI from fMRIprep (used for DVARS)
    work_dir     : scratch directory for nipype intermediates
    """

    # ------------------------------------------------------------------
    # [1] Load the .par file (6 columns: 3 rotations, 3 translations)
    # ------------------------------------------------------------------
    mc_confounds = pd.read_csv(mcf_par_file, sep=r'\s+', header=None)
    mc_confounds.columns = [
        'rot_x', 'rot_y', 'rot_z',
        'trans_x', 'trans_y', 'trans_z',
    ]

    # ------------------------------------------------------------------
    # [2] Framewise Displacement via nipype
    # ------------------------------------------------------------------
    fd_node = FramewiseDisplacement(
        in_file=mcf_par_file,
        parameter_source='FSL',
        save_plot=False,
    )
    fd_result = fd_node.run(cwd=work_dir)
    fd_file   = fd_result.outputs.out_file
    fd_vals   = np.loadtxt(fd_file, skiprows=1)  # header: "FramewiseDisplacement"
    # nipype omits volume 0 — prepend NaN to align with motion params
    fd_vals   = np.concatenate([[np.nan], fd_vals])

    # ------------------------------------------------------------------
    # [3] DVARS via nipype, using the fmriprep brainmask
    # ------------------------------------------------------------------
    mcf_nii_stage = _stage(mcf_nii, work_dir)

    dvars_node = ComputeDVARS(
        in_file=mcf_nii_stage,
        in_mask=brainmask,
        save_plot=False,
        save_all=True,
    )
    dvars_result = dvars_node.run(cwd=work_dir)
    dvars_file   = dvars_result.outputs.out_all
    dvars_vals   = np.loadtxt(dvars_file, skiprows=1)[:, 0]  # col 0 = std DVARS
    dvars_vals   = np.concatenate([[np.nan], dvars_vals])

    # ------------------------------------------------------------------
    # [4] Assemble
    # ------------------------------------------------------------------
    mc_confounds['framewise_displacement'] = fd_vals
    mc_confounds['dvars']                  = dvars_vals

    return mc_confounds


def build_confounds_tsv(
    fmriprep_confounds_tsv: str,
    mcf_nii: str,
    mcf_par_file: str,
    brainmask: str,
    output_tsv: str,
    work_dir: str,
) -> None:
    """
    Steps 1-3: Extract fMRIprep noise confounds (motion columns removed) and
    mcflirt motion confounds (6 DOF + FD + DVARS), then merge and save with
    the standardised BIDS-style name:

        sub-XX_ses-XX_task-XX_run-XX_desc-confounds.tsv

    Parameters
    ----------
    fmriprep_confounds_tsv : fMRIprep *_desc-confounds_timeseries.tsv
    mcf_nii                : motion-corrected 4-D BOLD NIfTI
    mcf_par_file           : MCFLIRT .par file
    brainmask              : brain mask NIfTI from fMRIprep
    output_tsv             : destination path (the standardised name above)
    work_dir               : scratch directory
    """

    # Step 1 — fMRIprep noise regressors (motion columns dropped)
    noise_df = extract_fmriprep_confounds(fmriprep_confounds_tsv)

    # Step 2 — mcflirt motion confounds (6 DOF + FD + DVARS, fmriprep mask)
    motion_df = extract_mcflirt_confounds(
        mcf_nii=mcf_nii,
        mcf_par_file=mcf_par_file,
        brainmask=brainmask,
        work_dir=work_dir,
    )

    # Step 3 — merge (motion first, then fMRIprep noise) and save
    merged = pd.concat(
        [motion_df.reset_index(drop=True), noise_df.reset_index(drop=True)],
        axis=1,
    )
    merged.to_csv(output_tsv, sep='\t', index=False, na_rep='n/a')
    print('  Confounds ({} columns) -> {}'.format(len(merged.columns), output_tsv))
    return merged


def compute_pca_confounds(
    confounds_tsv: str,
    output_tsv: str,
    lf_filter,
    ncomps: int = 6,
    nuissance_vars: list = [
        'csf', 'white_matter',
        'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04',
        't_comp_cor_00', 't_comp_cor_01', 't_comp_cor_02', 't_comp_cor_03', 't_comp_cor_04',
        'std_dvars', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
        'framewise_displacement', 'dvars',
    ],
) -> None:
    """
    Step 4: Run PCA on the confound matrix, retain the top *ncomps*,
    high-pass filter each component with a Savitzky-Golay filter to remove
    slow drifts, and save.

    Output file name:
        sub-XX_ses-XX_task-XX_run-XX_desc-pca_confounds.tsv

    PCA logic:
      - Subsets to nuissance_vars that exist in the file
      - NaNs are replaced with per-column medians (matching prepare_frame())
      - Data is z-scored (scipy.stats.zscore)
      - PCA is fit and transformed
      - SG filter is applied to PCA components (high-pass: subtract trend,
        add back mean)
      - sg_window is in SECONDS; converted to volumes via TR internally

    Parameters
    ----------
    confounds_tsv : merged confound file (from build_confounds_tsv)
    output_tsv    : destination path (pca_confounds.tsv)
    ncomps  : number of PCA components to retain (default: 6) 
    nuissance_vars: columns to select from the confounds tsv
    """
    df = pd.read_csv(confounds_tsv, sep='\t', na_values='n/a')
    if isinstance(nuissance_vars, str):
        if nuissance_vars == 'all':
            nuissance_vars = list(df.columns)
    # Subset to nuissance variables that are actually present
    available = [v for v in nuissance_vars if v in df.columns]
    if not available:
        raise ValueError(
            'None of the requested nuissance_vars were found in {}'.format(
                confounds_tsv))
    df = df[available]
    print(df)
    print(f'N components available = {len(available)}')
    pca_denoiser = PCA_denoiser(
        confounds=df, 
        lf_filter=lf_filter, 
        ncomps=ncomps,
    )
    pca_comps = pca_denoiser.run_pca()
    col_names = ['pca_comp_{:02d}'.format(i + 1) for i in range(ncomps)]
    out_df = pd.DataFrame(pca_comps, columns=col_names)
    out_df.to_csv(output_tsv, sep='\t', index=False, na_rep='n/a')
    return pca_denoiser


def denoise_volume(
    bold_file: str,
    pca_denoiser, 
    output_nii: str,
) -> None:
    """
    (volume): Denoise a 4-D NIfTI

    Output saved as _desc-denoised_bold.nii.gz.

    Parameters
    ----------
    bold_file         : motion-corrected BOLD NIfTI (sdc+moco)
    pca_denoiser      : PCADenoiser object 
    output_nii        : destination path (_desc-denoised_bold.nii.gz)
    """
    print('  Loading BOLD: {}'.format(bold_file))
    img    = nib.load(bold_file)
    data   = img.get_fdata(dtype=np.float32)    # (X, Y, Z, T)
    shape  = data.shape
    n_vols = shape[-1]
    flat   = data.reshape(-1, n_vols)           # (voxels, T)

    # Load PCA confounds
    flat_denoised = pca_denoiser.PCA_regression(flat)
    # Save
    denoised = flat_denoised.reshape(shape).astype(np.float32)
    out_img  = nib.Nifti1Image(denoised, img.affine, img.header)
    nib.save(out_img, output_nii)
    print('  Denoised volume  -> {}'.format(output_nii))


def denoise_surface(
    gifti_file: str,
    pca_denoiser,
    output_gifti: str,
) -> None:
    """
    (surface): Denoise a surface GIFTI 
    
    Output saved as _desc-denoised_bold.func.gii.

    Parameters
    ----------
    gifti_file        : .func.gii surface timeseries
    pca_denoiser      : PCADenoiser object
    output_gifti      : destination path for denoised GIFTI
    """

    print('  Loading surface: {}'.format(gifti_file))
    gii    = nib.load(gifti_file)
    arrays = [da.data for da in gii.darrays]
    data   = np.vstack(arrays).T.astype(float)   # (vertices, T)
    n_vols = data.shape[-1]

    data_denoised = pca_denoiser.PCA_regression(data)

    # Rebuild GIFTI
    data_denoised_T = data_denoised.T                                # (T, vertices)
    new_darrays = []
    for t, da in enumerate(gii.darrays):
        new_da = nib.gifti.GiftiDataArray(
            data=data_denoised_T[t, :].astype(np.float32),
            intent=da.intent,
            datatype=da.datatype,
            meta=da.meta,
        )
        new_darrays.append(new_da)

    out_gii = nib.gifti.GiftiImage(darrays=new_darrays, meta=gii.meta)
    nib.save(out_gii, output_gifti)
    print('  Denoised surface -> {}'.format(output_gifti))


def filter_volume(
    bold_file: str,
    output_nii: str,
    lf_filter,
) -> None:
    """
    Filter-only (volume): Apply SG high-pass filter to a 4-D NIfTI volume
    without any confound regression.

    Identical filter parameters to denoise_volume() so outputs are directly
    comparable — the difference between _desc-filtered_bold and
    _desc-denoised_bold isolates the contribution of confound regression.

    Output saved as _desc-filtered_bold.nii.gz.

    Parameters
    ----------
    bold_file  : motion-corrected BOLD NIfTI (sdc+moco)
    output_nii : destination path (_desc-filtered_bold.nii.gz)
    lf_filter  : SGFilter
    """
    

    print('  Loading BOLD: {}'.format(bold_file))
    img    = nib.load(bold_file)
    data   = img.get_fdata(dtype=np.float32)    # (X, Y, Z, T)
    shape  = data.shape
    n_vols = shape[-1]
    flat   = data.reshape(-1, n_vols)           # (voxels, T)

    flat_hp = lf_filter.filter_data(data=flat)
    filtered = flat_hp.reshape(shape).astype(np.float32)
    out_img  = nib.Nifti1Image(filtered, img.affine, img.header)
    nib.save(out_img, output_nii)
    print('  Filtered volume  -> {}'.format(output_nii))

def filter_surface(
    gifti_file: str,
    output_gifti: str,
    lf_filter,
) -> None:
    """
    Filter-only (surface): Apply SG high-pass filter to a surface GIFTI
    timeseries without any confound regression.

    Output saved as _desc-filtered_bold.func.gii.

    Parameters
    ----------
    gifti_file   : .func.gii surface timeseries
    output_gifti : destination path for filtered GIFTI
    lf_filter    : SGFIlter object
    """

    print('  Loading surface: {}'.format(gifti_file))
    gii    = nib.load(gifti_file)
    arrays = [da.data for da in gii.darrays]
    data   = np.vstack(arrays).T.astype(float)   # (vertices, T)

    data_hp = lf_filter.filter_data(data=data)
    # Rebuild GIFTI
    data_hp_T = data_hp.T                               # (T, vertices)
    new_darrays = []
    for t, da in enumerate(gii.darrays):
        new_da = nib.gifti.GiftiDataArray(
            data=data_hp_T[t, :].astype(np.float32),
            intent=da.intent,
            datatype=da.datatype,
            meta=da.meta,
        )
        new_darrays.append(new_da)

    out_gii = nib.gifti.GiftiImage(darrays=new_darrays, meta=gii.meta)
    nib.save(out_gii, output_gifti)
    print('  Filtered surface -> {}'.format(output_gifti))


# ---------------------------------------------------------------------------
# Per-run pipeline
# ---------------------------------------------------------------------------

def process_run(
    bold_file: str,
    mcf_par_file: str,
    fmriprep_confounds_tsv: str,
    brainmask: str,
    surf_lh_file: str | None,
    surf_rh_file: str | None,
    subject: str,
    session: str,
    subject_output_dir: str,
    n_pca: int,
    sg_window: int,
    sg_polyorder: int,
    sg_deriv: int,
    overwrite: dict,
    skip: dict,
    filter_only: bool = False,
) -> dict:
    """
    Execute all per-run confound extraction and denoising steps.

    Steps 1-3 (confounds)  : extract fMRIprep noise regressors + mcflirt
                              motion confounds, merge into a single .tsv
    Step 4 (pca_confounds) : PCA + SG high-pass filter → pca_confounds.tsv
    Step 5 (denoise_vol)   : SG high-pass + OLS regression → denoised volume
         (denoise_surf)    : same for each surface hemisphere

    When filter_only=True, Steps 1-4 are skipped entirely and Step 5 applies
    only the SG high-pass filter (no regression), saving outputs as
    _desc-filtered_bold instead of _desc-denoised_bold.

    Parameters
    ----------
    overwrite : dict mapping step key → bool; True forces re-run even if
                outputs exist.
    skip      : dict mapping step key → bool; True hard-skips the step with
                no file checks and no work_dir restore. Downstream steps
                continue regardless — caller is responsible for dependencies.

    Returns a dict of final output paths for this run.
    """
    ow = {k: False for k in STEP_KEYS}
    ow.update(overwrite)

    sk = {k: False for k in STEP_KEYS}
    sk.update(skip)

    run_label, task_label = get_labels(bold_file)
    run_suffix = '_'.join(t for t in [task_label, run_label] if t)

    work_dir = os.path.join(subject_output_dir, run_suffix)
    os.makedirs(work_dir, exist_ok=True)
    safe_work_dir = make_safe_workdir(work_dir)

    base = '_'.join([subject, session, run_suffix])

    def _final(suffix, ext='.tsv'):
        return build_output_name(
            subject_output_dir, subject, session, suffix, extension=ext)

    def _work(filename):
        return os.path.join(safe_work_dir, filename)

    results = {}

    tr = _get_tr(bold_file)
    # create filter object
    lf_filter = SGFilter(
        polyorder=sg_polyorder,
        deriv=sg_deriv,
        window_length=sg_window,
        tr=tr,
    )
    if filter_only:
        # ------------------------------------------------------------------
        # Filter-only mode: SG high-pass only, no confound regression
        # ------------------------------------------------------------------

        # --- Volumetric ---
        print('\n  [Filter-only] High-pass filtering volumetric BOLD...')

        filtered_vol_final = _final(
            '{}_desc-filtered_bold'.format(base), ext='.nii.gz')
        filtered_vol_work  = _work('bold_filtered.nii.gz')

        if not check_skip(
            {'filtered_vol': filtered_vol_final},
            ow['filter_vol'],
            'Filter-only: filter volume',
            workdir_paths={'filtered_vol': filtered_vol_work},
            force_skip=sk['filter_vol'],
        ):
            filter_volume(
                bold_file=bold_file,
                output_nii=filtered_vol_work,
                lf_filter=lf_filter,
            )
            shutil.copy(filtered_vol_work, filtered_vol_final)

        print('  Filtered volume  : {}'.format(filtered_vol_final))
        results['filtered_bold'] = filtered_vol_final

        # --- Surface ---
        print('\n  [Filter-only] High-pass filtering surface BOLD...')

        surf_outputs = {}
        for hemi_key, surf_file, hemi_gifti in [
            ('lh', surf_lh_file, 'L'),
            ('rh', surf_rh_file, 'R'),
        ]:
            if surf_file is None or not Path(surf_file).exists():
                print('  No surface file for {} — skipping.'.format(hemi_key))
                surf_outputs[hemi_key] = None
                continue

            surf_name = '{}_space-fsnative_hemi-{}_desc-filtered_bold.func.gii'.format(
                base, hemi_gifti)
            filtered_surf_final = os.path.join(subject_output_dir, surf_name)
            filtered_surf_work  = _work(
                'bold_hemi-{}_filtered.func.gii'.format(hemi_gifti))

            if not check_skip(
                {'filtered_surf_{}'.format(hemi_key): filtered_surf_final},
                ow['filter_surf'],
                'Filter-only: filter surface ({})'.format(hemi_key),
                workdir_paths={
                    'filtered_surf_{}'.format(hemi_key): filtered_surf_work},
                force_skip=sk['filter_surf'],
            ):
                filter_surface(
                    gifti_file=surf_file,
                    output_gifti=filtered_surf_work,
                    lf_filter=lf_filter,
                )
                shutil.copy(filtered_surf_work, filtered_surf_final)

            surf_outputs[hemi_key] = filtered_surf_final
            print('  Filtered surface ({}) : {}'.format(
                hemi_key, filtered_surf_final))

        results['filtered_surf_lh'] = surf_outputs.get('lh')
        results['filtered_surf_rh'] = surf_outputs.get('rh')

    else:
        # ------------------------------------------------------------------
        # Full denoising mode (original pipeline)
        # ------------------------------------------------------------------

        # Steps 1-3: Extract fMRIprep noise + mcflirt motion → merged .tsv
        print('\n  [Steps 1-3] Extracting and merging confounds...')

        confounds_final = _final('{}_desc-confounds'.format(base))
        confounds_work  = _work('confounds.tsv')

        if not check_skip(
            {'confounds_tsv': confounds_final},
            ow['confounds'],
            'Steps 1-3: build confounds',
            workdir_paths={'confounds_tsv': confounds_work},
            force_skip=sk['confounds'],
        ):
            build_confounds_tsv(
                fmriprep_confounds_tsv=fmriprep_confounds_tsv,
                mcf_nii=bold_file,
                mcf_par_file=mcf_par_file,
                brainmask=brainmask,
                output_tsv=confounds_work,
                work_dir=safe_work_dir,
            )
            shutil.copy(confounds_work, confounds_final)

        print('  Confounds        : {}'.format(confounds_final))

        # Step 4: PCA + SG high-pass filter
        print('\n  [Step 4] Computing PCA confounds...')

        pca_tsv_final = _final('{}_desc-pca_confounds'.format(base))
        pca_tsv_work  = _work('pca_confounds.tsv')

        if not check_skip(
            {'pca_tsv': pca_tsv_final},
            ow['pca_confounds'],
            'Step 4: PCA confounds',
            workdir_paths={'pca_tsv': pca_tsv_work},
            force_skip=sk['pca_confounds'],
        ):
            pca_denoiser = compute_pca_confounds(
                confounds_tsv=confounds_work,
                output_tsv=pca_tsv_work,
                lf_filter=lf_filter,
                ncomps=n_pca,
            )
            shutil.copy(pca_tsv_work, pca_tsv_final)
        else:
            full_conf = pd.read_csv(confounds_final, sep='\t', na_values='n/a')
            pca_conf = pd.read_csv(pca_tsv_final, sep='\t', na_values='n/a')
            pca_denoiser = PCA_denoiser(
                confounds=full_conf,
                lf_filter=lf_filter,
                ncomps=n_pca,
            )
            pca_denoiser.pca_comps = pca_conf.to_numpy()
            
        print('  PCA confounds    : {}'.format(pca_tsv_final))

        # Step 5a: Denoise volumetric BOLD
        print('\n  [Step 5a] Denoising volumetric BOLD...')

        denoised_vol_final = _final(
            '{}_desc-denoised_bold'.format(base), ext='.nii.gz')
        denoised_vol_work  = _work('bold_denoised.nii.gz')

        if not check_skip(
            {'denoised_vol': denoised_vol_final},
            ow['denoise_vol'],
            'Step 5a: denoise volume',
            workdir_paths={'denoised_vol': denoised_vol_work},
            force_skip=sk['denoise_vol'],
        ):
            denoise_volume(
                bold_file=bold_file,
                pca_denoiser=pca_denoiser,
                output_nii=denoised_vol_work,
            )
            shutil.copy(denoised_vol_work, denoised_vol_final)

        print('  Denoised volume  : {}'.format(denoised_vol_final))

        # Step 5b: Denoise surface projections (lh + rh)
        print('\n  [Step 5b] Denoising surface BOLD...')

        surf_outputs = {}

        for hemi_key, surf_file, hemi_gifti in [
            ('lh', surf_lh_file, 'L'),
            ('rh', surf_rh_file, 'R'),
        ]:
            if surf_file is None or not Path(surf_file).exists():
                print('  No surface file for {} — skipping.'.format(hemi_key))
                surf_outputs[hemi_key] = None
                continue

            surf_name = '{}_space-fsnative_hemi-{}_desc-denoised_bold.func.gii'.format(
                base, hemi_gifti)
            denoised_surf_final = os.path.join(subject_output_dir, surf_name)
            denoised_surf_work  = _work(
                'bold_hemi-{}_denoised.func.gii'.format(hemi_gifti))

            if not check_skip(
                {'denoised_surf_{}'.format(hemi_key): denoised_surf_final},
                ow['denoise_surf'],
                'Step 5b: denoise surface ({})'.format(hemi_key),
                workdir_paths={
                    'denoised_surf_{}'.format(hemi_key): denoised_surf_work},
                force_skip=sk['denoise_surf'],
            ):
                denoise_surface(
                    gifti_file=surf_file,
                    pca_denoiser=pca_denoiser,
                    output_gifti=denoised_surf_work,
                )
                shutil.copy(denoised_surf_work, denoised_surf_final)

            surf_outputs[hemi_key] = denoised_surf_final
            print('  Denoised surface ({}) : {}'.format(
                hemi_key, denoised_surf_final))

        results = {
            'confounds':        confounds_final,
            'pca_confounds':    pca_tsv_final,
            'denoised_bold':    denoised_vol_final,
            'denoised_surf_lh': surf_outputs.get('lh'),
            'denoised_surf_rh': surf_outputs.get('rh'),
        }

    shutil.rmtree(work_dir)
    return results


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    bids_dir: str,
    moco_file: str,
    output_file: str,
    subject: str,
    session: str = 'ses-01',
    n_pca: int = 6,
    sg_window: int = 120,
    sg_polyorder: int = 3,
    sg_deriv: int = 0, 
    overwrite: dict = None,
    skip: dict = None,
    filter_only: bool = False,
) -> dict:
    """
    Run the full confound extraction and denoising pipeline.

    For each BOLD run the pipeline:
      - Locates fMRIprep confounds .tsv and brainmask
        (from bids_dir/derivatives/fmriprep/<subject>/<session>/func/)
      - Locates the MCFLIRT .par file (from moco_file derivatives directory)
      - Steps 1-3: builds the merged confounds .tsv
      - Step 4: computes PCA confounds with SG high-pass filter
      - Step 5: denoises volumetric and surface BOLD

    When filter_only=True:
      - Steps 1-4 (confound extraction and PCA) are skipped entirely.
      - Only the SG high-pass filter is applied (no confound regression).
      - Outputs are saved as _desc-filtered_bold to distinguish them from
        _desc-denoised_bold, allowing direct comparison of filter-only vs
        full denoising.
      - fMRIprep confounds .tsv and brainmask are NOT required in this mode.

    The moco_file directory is expected to contain:
      - *bold*.nii.gz           – motion-corrected BOLD volumes
      - *mcflirt_motion*.par    – MCFLIRT .par files (not needed in filter_only mode)
      - *hemi-L*.func.gii       – left-hemisphere surface timeseries
      - *hemi-R*.func.gii       – right-hemisphere surface timeseries

    fMRIprep derivatives (auto-located) must contain per run (full mode only):
      - *desc-confounds_timeseries.tsv  – fMRIprep confounds
      - *desc-brain_mask.nii.gz         – brain mask (used for DVARS)

    Parameters
    ----------
    overwrite : dict mapping step key → bool; True forces re-run of that step
                even if outputs already exist. Valid keys: STEP_KEYS.
    skip      : dict mapping step key → bool; True hard-skips that step with
                no file checks and no work_dir restore. Downstream steps
                continue regardless — caller is responsible for dependencies.
                Valid keys: STEP_KEYS.

    Returns a dict mapping run keys → per-run output dicts.
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

    moco_dir     = str(Path(
        os.path.join(bids_dir, 'derivatives', moco_file)
    ).resolve())
    output_dir   = str(Path(
        os.path.join(bids_dir, 'derivatives', output_file)
    ).resolve())
    fmriprep_dir = str(Path(
        os.path.join(bids_dir, 'derivatives', 'fmriprep')
    ).resolve())

    subject_input_dir    = os.path.join(moco_dir,     subject, session)
    subject_output_dir   = os.path.join(output_dir,   subject, session)
    subject_fmriprep_dir = os.path.join(fmriprep_dir, subject, session)

    os.makedirs(subject_output_dir, exist_ok=True)

    mode_label = 'Filter-only (SG high-pass, no confound regression)' \
        if filter_only else 'Confound Extraction + Denoising'

    print('-' * 55)
    print('Processing: {}'.format(mode_label))
    print('-' * 55)
    print(' Moco input  : {}'.format(moco_dir))
    if not filter_only:
        print(' fMRIprep    : {}'.format(fmriprep_dir))
    print(' Output      : {}'.format(output_dir))
    print(' Subject     : {}'.format(subject))
    print(' Session     : {}'.format(session))
    if not filter_only:
        print(' PCA comps   : {}'.format(n_pca))
    print(' SG window   : {}s (order {})'.format(sg_window, sg_polyorder))
    skipped_steps = [k for k, v in sk.items() if v]
    if skipped_steps:
        print(' Force-skip  : {}'.format(', '.join(skipped_steps)))
    print('-' * 55)

    # ------------------------------------------------------------------
    # Discover BOLD runs in the moco directory
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
    for b in bold_files:
        print('  - {}'.format(os.path.basename(b)))

    all_results = {}

    for run_idx, bold_file in enumerate(bold_files, start=1):
        print('\n' + '=' * 55)
        print('Processing run {}/{}: {}'.format(
            run_idx, len(bold_files), os.path.basename(bold_file)))
        print('=' * 55)

        run_label, task_label = get_labels(bold_file)
        parts = [subject, session]
        if task_label:
            parts.append(task_label)
        if run_label:
            parts.append(run_label)

        # fMRIprep confounds + brainmask — only needed in full denoising mode
        fmriprep_confounds_tsv = None
        brainmask = None
        mcf_par_file = None

        if not filter_only:
            # ------------------------------------------------------------------
            # Locate matching MCFLIRT .par file
            # ------------------------------------------------------------------
            fallback_pat = os.path.join(
                subject_input_dir,
                '{}_*mcflirt*.par'.format('_'.join(parts)))
            par_files = sorted(glob.glob(fallback_pat))
            if not par_files:
                raise FileNotFoundError(
                    'No MCFLIRT .par file found for {}'.format(
                        os.path.basename(bold_file)))

            mcf_par_file = par_files[0]
            print('  BOLD  : {}'.format(bold_file))
            print('  PAR   : {}'.format(mcf_par_file))

            # ------------------------------------------------------------------
            # Locate fMRIprep confounds .tsv (Step 1 input)
            # ------------------------------------------------------------------
            fmriprep_conf_pattern = os.path.join(
                subject_fmriprep_dir, 'func',
                '{}*desc-confounds_timeseries.tsv'.format('_'.join(parts)))
            fmriprep_conf_hits = sorted(glob.glob(fmriprep_conf_pattern))

            if not fmriprep_conf_hits:
                raise FileNotFoundError(
                    'No fMRIprep confounds .tsv found for {}. Searched: {}'.format(
                        os.path.basename(bold_file), fmriprep_conf_pattern))

            fmriprep_confounds_tsv = fmriprep_conf_hits[0]
            print('  FMRIPREP CONF: {}'.format(fmriprep_confounds_tsv))

            # ------------------------------------------------------------------
            # Locate fMRIprep brain mask (Step 2 input for DVARS)
            # ------------------------------------------------------------------
            brainmask_pattern = os.path.join(
                subject_fmriprep_dir, 'func',
                '{}*desc-brain_mask.nii.gz'.format('_'.join(parts))
                )
            brainmask_hits = sorted(glob.glob(brainmask_pattern))

            if not brainmask_hits:
                raise FileNotFoundError(
                    'No fMRIprep brain mask found for {}. Searched: {}'.format(
                        os.path.basename(bold_file), brainmask_pattern))

            brainmask = brainmask_hits[0]
            print('  BRAINMASK    : {}'.format(brainmask))
        else:
            print('  BOLD  : {}'.format(bold_file))

        # ------------------------------------------------------------------
        # Locate surface timeseries (optional)
        # ------------------------------------------------------------------
        def _find_surf(hemi_letter):
            # *** TO UPDATE ***
            pat = os.path.join(
                subject_input_dir,
                f'*{task_label}_{run_label}*space-fsnative_hemi-{hemi_letter}_bold.func.gii'
            )
            hits = sorted(glob.glob(pat))
            return hits[0] if hits else None

        surf_lh = _find_surf('L')
        surf_rh = _find_surf('R')

        if surf_lh:
            print('  SURF-L: {}'.format(surf_lh))
        else:
            print('  SURF-L: not found — surface denoising will be skipped.')
        if surf_rh:
            print('  SURF-R: {}'.format(surf_rh))
        else:
            print('  SURF-R: not found — surface denoising will be skipped.')

        # ------------------------------------------------------------------
        # Run per-run pipeline
        # ------------------------------------------------------------------
        run_results = process_run(
            bold_file=bold_file,
            mcf_par_file=mcf_par_file,
            fmriprep_confounds_tsv=fmriprep_confounds_tsv,
            brainmask=brainmask,
            surf_lh_file=surf_lh,
            surf_rh_file=surf_rh,
            subject=subject,
            session=session,
            subject_output_dir=subject_output_dir,
            n_pca=n_pca,
            sg_window=sg_window,
            sg_polyorder=sg_polyorder,
            sg_deriv=sg_deriv,
            overwrite=ow,
            skip=sk,
            filter_only=filter_only,
        )

        key = run_label if run_label else 'run-{:02d}'.format(run_idx)
        all_results[key] = run_results
        print('\n  Run {} completed.'.format(run_idx))

    print('\n' + '=' * 55)
    print('All {} run(s) completed successfully.'.format(len(bold_files)))
    print('Output directory: {}'.format(subject_output_dir))
    print('=' * 55)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Confound extraction + denoising for BOLD runs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',    required=True,
                     help='BIDS directory')
    req.add_argument('--moco-file',   required=True,
                     help='Derivatives subdirectory with motion-corrected BOLD '
                          'and .par files (e.g. s03_motion_correction)')
    req.add_argument('--output-file', required=True,
                     help='Output derivatives subdirectory '
                          '(e.g. s4_motion_confounds)')
    req.add_argument('--sub',         required=True,
                     help='Subject label (e.g. sub-01 or 01)')

    p.add_argument('--ses',          default='ses-01',
                   help='Session label (e.g. ses-01)')
    p.add_argument('--n-pca',     type=int, default=6,
                   help='Number of PCA components to retain')
    p.add_argument('--sg-window', type=int, default=347,
                   help='Savitzky-Golay filter window length in SECONDS '
                        '(converted to volumes via TR internally)')
    p.add_argument('--sg-polyorder',  type=int, default=3,
                   help='Savitzky-Golay polynomial order')
    p.add_argument('--sg-deriv',  type=int, default=0,
                help='Savitzky-Golay polynomial order')
    
    p.add_argument(
        '--filter-only',
        action='store_true',
        default=False,
        help=(
            'Apply SG high-pass filter only — skip confound extraction and '
            'PCA regression entirely. Outputs are saved as _desc-filtered_bold '
            'so they can be compared directly against _desc-denoised_bold to '
            'isolate how much improvement comes from filtering vs denoising. '
            'Does not require fMRIprep confounds .tsv or brain mask.'
        ),
    )

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
        bids_dir=args.bids_dir,
        moco_file=args.moco_file,
        output_file=args.output_file,
        subject=args.sub,
        session=args.ses,
        n_pca=args.n_pca,
        sg_window=args.sg_window,
        sg_polyorder=args.sg_polyorder,
        sg_deriv=args.sg_deriv,
        overwrite=overwrite,
        skip=skip,
        filter_only=args.filter_only,
    )


if __name__ == '__main__':
    main()