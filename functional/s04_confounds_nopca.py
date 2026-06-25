#!/usr/bin/env python
"""
s04_confounds.py
===============
Denoise fMRI data using a V1-Optimized Connective Field Strategy.

Pipeline:
1) Extract fMRIprep noise components (removing standard motion columns).
2) Extract MCFLIRT motion and compute the Friston-24 motion model.
3) Compute FD and create dummy spike regressors for bad volumes.
4) Compute DVARS using the fMRIPrep brainmask.
5) Merge and build the Final Design Matrix:
   - Top 5 CSF & Top 5 WM components
   - All discrete cosine basis regressors (drift)
   - 24-parameter motion model
   - Dummy spike regressors
6) Denoise volumetric and surface BOLD using OLS regression.
"""

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nipype.algorithms.confounds import FramewiseDisplacement, ComputeDVARS
from sklearn.linear_model import LinearRegression

from cvl_utils.preproc_func import (
    build_output_name,
    check_skip,
    get_labels,
    make_safe_workdir,
    _stage,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'confounds',       # Steps 1-3: Merge fMRIprep noise + mcflirt motion
    'design_matrix',   # Step 4: Extract V1 regressors -> design_matrix.tsv
    'denoise_vol',     # Step 5a: OLS regression on volumetric BOLD
    'denoise_surf',    # Step 5b: OLS regression on surface BOLD
]

MOTION_PATTERNS = [
    r'^trans_[xyz]', r'^rot_[xyz]', r'^framewise_displacement',
    r'^dvars', r'^std_dvars', r'^rmsd', r'^motion_outlier',
]

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def extract_fmriprep_confounds(fmriprep_tsv: str) -> pd.DataFrame:
    """Load fMRIprep confounds and strip out its default motion columns."""
    df = pd.read_csv(fmriprep_tsv, sep='\t', na_values='n/a')
    motion_cols = [c for c in df.columns if any(re.match(p, c) for p in MOTION_PATTERNS)]
    noise_df = df.drop(columns=motion_cols)
    print(f"  fMRIprep confounds: kept {len(noise_df.columns)} columns (dropped {len(motion_cols)} motion cols).")
    return noise_df

def extract_mcflirt_confounds(mcf_nii: str, mcf_par: str, brainmask: str, work_dir: str, fd_thresh: float) -> pd.DataFrame:
    """Extract base motion, compute Friston-24, FD, DVARS, and Dummy Spikes."""
    mc = pd.read_csv(mcf_par, sep=r'\s+', header=None)
    base_cols = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
    mc.columns = base_cols

    # Friston-24
    for col in base_cols:
        mc[f'{col}_deriv1'] = mc[col].diff().fillna(0)
        mc[f'{col}_power2'] = mc[col] ** 2
        mc[f'{col}_deriv1_power2'] = mc[f'{col}_deriv1'] ** 2

    # Framewise Displacement & Spikes
    fd_node = FramewiseDisplacement(in_file=mcf_par, parameter_source='FSL', save_plot=False)
    fd_file = fd_node.run(cwd=work_dir).outputs.out_file
    fd_vals = np.concatenate([[np.nan], np.loadtxt(fd_file, skiprows=1)])
    mc['framewise_displacement'] = fd_vals

    spikes = np.where(fd_vals > fd_thresh)[0]
    for idx in spikes:
        mc[f'spike_vol_{idx}'] = (np.arange(len(fd_vals)) == idx).astype(int)
    print(f"  Generated {len(spikes)} dummy spike regressors (FD > {fd_thresh}mm).")

    # DVARS
    mcf_stage = _stage(mcf_nii, work_dir)
    bold_img = nib.load(mcf_stage)
    mask_img = nib.load(brainmask)
    if mask_img.shape[:3] != bold_img.shape[:3]:
        mask_img = resample_to_img(mask_img, bold_img, interpolation='nearest', force_resample=True)
        resampled_mask = os.path.join(work_dir, 'brainmask_resampled.nii.gz')
        nib.save(mask_img, resampled_mask)
        brainmask = resampled_mask

    dvars_node = ComputeDVARS(in_file=mcf_stage, in_mask=brainmask, save_plot=False, save_all=True)
    dvars_file = dvars_node.run(cwd=work_dir).outputs.out_all
    mc['dvars'] = np.concatenate([[np.nan], np.loadtxt(dvars_file, skiprows=1)[:, 0]])

    return mc

def build_confounds_tsv(fmriprep_tsv: str, mcf_nii: str, mcf_par: str, mask: str, out_tsv: str, work_dir: str, fd_thresh: float) -> pd.DataFrame:
    """Merge fMRIprep noise and extended MCFLIRT motion."""
    noise_df = extract_fmriprep_confounds(fmriprep_tsv)
    motion_df = extract_mcflirt_confounds(mcf_nii, mcf_par, mask, work_dir, fd_thresh)
    merged = pd.concat([motion_df.reset_index(drop=True), noise_df.reset_index(drop=True)], axis=1)
    merged.to_csv(out_tsv, sep='\t', index=False, na_rep='n/a')
    return merged

def build_design_matrix(confounds_tsv: str, output_tsv: str) -> pd.DataFrame:
    """Extract explicit V1-Optimized Connective Field regressors."""
    df = pd.read_csv(confounds_tsv, sep='\t', na_values='n/a')
    
    target_cols = []
    target_cols += [f'c_comp_cor_{i:02d}' for i in range(5)]
    target_cols += [f'w_comp_cor_{i:02d}' for i in range(5)]
    target_cols += [c for c in df.columns if c.startswith('cosine')]
    
    base_motion = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
    target_cols += base_motion + [f'{m}_deriv1' for m in base_motion] + [f'{m}_power2' for m in base_motion] + [f'{m}_deriv1_power2' for m in base_motion]
    target_cols += [c for c in df.columns if c.startswith('spike_vol_')]

    available = [c for c in target_cols if c in df.columns]
    design_matrix = df[available].copy()
    design_matrix.fillna(design_matrix.median(), inplace=True)
    design_matrix.to_csv(output_tsv, sep='\t', index=False)
    
    print(f"  Design Matrix built with {len(design_matrix.columns)} regressors.")
    return design_matrix

def denoise_data(data: np.ndarray, design_matrix: pd.DataFrame) -> np.ndarray:
    """OLS Regression math. Data shape: (timepoints, features)."""
    X = design_matrix.values
    reg = LinearRegression(n_jobs=-1).fit(X, data)
    noise_pred = reg.predict(X)
    return data - noise_pred + data.mean(axis=0)

def denoise_volume(bold_file: str, design_matrix: pd.DataFrame, output_nii: str) -> None:
    """Apply denoising to 4D NIfTI."""
    print(f"  Loading BOLD: {bold_file}")
    img = nib.load(bold_file)
    data = img.get_fdata(dtype=np.float32)
    shape = data.shape
    
    flat_data = data.reshape(-1, shape[-1]).T
    denoised_flat = denoise_data(flat_data, design_matrix)
    denoised_data = denoised_flat.T.reshape(shape).astype(np.float32)
    
    nib.save(nib.Nifti1Image(denoised_data, img.affine, img.header), output_nii)
    print(f"  Denoised volume -> {output_nii}")

def denoise_surface(gifti_file: str, design_matrix: pd.DataFrame, output_gifti: str) -> None:
    """Apply denoising to Surface GIFTI."""
    print(f"  Loading surface: {gifti_file}")
    gii = nib.load(gifti_file)
    data = np.vstack([da.data for da in gii.darrays]).astype(float)
    
    denoised_data = denoise_data(data, design_matrix)
    
    new_darrays = [
        nib.gifti.GiftiDataArray(data=denoised_data[t, :].astype(np.float32), intent=da.intent, datatype=da.datatype, meta=da.meta) 
        for t, da in enumerate(gii.darrays)
    ]
    nib.save(nib.gifti.GiftiImage(darrays=new_darrays, meta=gii.meta), output_gifti)
    print(f"  Denoised surface -> {output_gifti}")

# ---------------------------------------------------------------------------
# Per-Run Pipeline Orchestration
# ---------------------------------------------------------------------------

def process_run(
    bold_file: str, mcf_par_file: str, fmriprep_tsv: str, brainmask: str,
    surf_lh_file: str | None, surf_rh_file: str | None, subject: str, session: str,
    output_dir: str, fd_threshold: float, overwrite: dict, skip: dict
) -> dict:
    
    ow = {k: False for k in STEP_KEYS}; ow.update(overwrite)
    sk = {k: False for k in STEP_KEYS}; sk.update(skip)

    run_label, task_label = get_labels(bold_file)
    run_suffix = '_'.join(t for t in [task_label, run_label] if t)
    work_dir = os.path.join(output_dir, run_suffix)
    os.makedirs(work_dir, exist_ok=True)
    safe_work = make_safe_workdir(work_dir)
    base = f"{subject}_{session}_{run_suffix}"

    def _final(suffix, ext='.tsv'): return build_output_name(output_dir, subject, session, suffix, extension=ext)
    def _work(filename): return os.path.join(safe_work, filename)

    results = {}

    # 1. Confounds
    conf_final = _final(f'{base}_desc-confounds')
    conf_work  = _work('confounds.tsv')
    if not check_skip({'conf': conf_final}, ow['confounds'], 'Build Confounds', {'conf': conf_work}, sk['confounds']):
        build_confounds_tsv(fmriprep_tsv, bold_file, mcf_par_file, brainmask, conf_work, safe_work, fd_threshold)
        shutil.copy(conf_work, conf_final)

    # 2. Design Matrix
    dm_final = _final(f'{base}_desc-design_matrix')
    dm_work  = _work('design_matrix.tsv')
    if not check_skip({'dm': dm_final}, ow['design_matrix'], 'Build Design Matrix', {'dm': dm_work}, sk['design_matrix']):
        dm = build_design_matrix(conf_final, dm_work)
        shutil.copy(dm_work, dm_final)
    else:
        dm = pd.read_csv(dm_final, sep='\t', na_values='n/a')

    # 3. Denoise Volume
    vol_final = _final(f'{base}_desc-denoised_bold', '.nii.gz')
    vol_work  = _work('bold_denoised.nii.gz')
    if not check_skip({'vol': vol_final}, ow['denoise_vol'], 'Denoise Volume', {'vol': vol_work}, sk['denoise_vol']):
        denoise_volume(bold_file, dm, vol_work)
        shutil.copy(vol_work, vol_final)
    results['denoised_bold'] = vol_final

    # 4. Denoise Surfaces
    for hemi, s_file, g_id in [('lh', surf_lh_file, 'L'), ('rh', surf_rh_file, 'R')]:
        if not s_file or not Path(s_file).exists(): continue
        s_final = os.path.join(output_dir, f'{base}_space-fsnative_hemi-{g_id}_desc-denoised_bold.func.gii')
        s_work = _work(f'bold_hemi-{g_id}_denoised.func.gii')
        if not check_skip({'s': s_final}, ow['denoise_surf'], f'Denoise Surf {hemi}', {'s': s_work}, sk['denoise_surf']):
            denoise_surface(s_file, dm, s_work)
            shutil.copy(s_work, s_final)
        results[f'denoised_surf_{hemi}'] = s_final
        
    results.update({'confounds': conf_final, 'design_matrix': dm_final})
    shutil.rmtree(work_dir)
    return results


def run_pipeline(
    bids_dir: str, moco_file: str, output_file: str, subject: str, session: str = 'ses-01',
    fd_threshold: float = 0.5, overwrite: dict = None, skip: dict = None
) -> dict:
    
    bids = Path(bids_dir).resolve()
    moco_dir = bids / 'derivatives' / moco_file / subject / session
    fmri_dir = bids / 'derivatives' / 'fmriprep' / subject / session
    out_dir  = bids / 'derivatives' / output_file / subject / session
    out_dir.mkdir(parents=True, exist_ok=True)

    bold_files = sorted(glob.glob(str(moco_dir / f'{subject}_{session}*bold.nii*')))
    if not bold_files: raise FileNotFoundError(f"No BOLD files found in {moco_dir}")

    all_results = {}
    for idx, bold in enumerate(bold_files, 1):
        print(f"\n{'='*55}\nProcessing run {idx}/{len(bold_files)}: {os.path.basename(bold)}\n{'='*55}")
        
        run_label, task_label = get_labels(bold)
        parts = [subject, session] + ([task_label] if task_label else []) + ([run_label] if run_label else [])
        prefix = '_'.join(parts)

        mcf_par = sorted(glob.glob(str(moco_dir / f'{prefix}_*mcflirt*.par')))[0]
        f_tsv = sorted(glob.glob(str(fmri_dir / 'func' / f'{prefix}*desc-confounds_timeseries.tsv')))[0]
        mask = sorted(glob.glob(str(fmri_dir / 'func' / f'{prefix}*desc-brain_mask.nii.gz')))[0]

        surf_lh = next(iter(sorted(glob.glob(str(moco_dir / f'*{task_label}_{run_label}*hemi-L_bold.func.gii')))), None)
        surf_rh = next(iter(sorted(glob.glob(str(moco_dir / f'*{task_label}_{run_label}*hemi-R_bold.func.gii')))), None)

        res = process_run(
            bold, mcf_par, f_tsv, mask, surf_lh, surf_rh, subject, session, str(out_dir),
            fd_threshold, overwrite or {}, skip or {}
        )
        all_results[run_label or f'run-{idx:02d}'] = res
        
    return all_results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir', required=True)
    req.add_argument('--moco-file', required=True)
    req.add_argument('--output-file', required=True)
    req.add_argument('--sub', required=True)

    p.add_argument('--ses', default='ses-01')
    p.add_argument('--fd-threshold', type=float, default=0.5, help="FD limit for dummy spikes.")

    ow_group = p.add_argument_group('overwrite / skip options')
    ow_group.add_argument('--overwrite', nargs='+', default=[], choices=STEP_KEYS)
    ow_group.add_argument('--overwrite-all', action='store_true', default=False)
    ow_group.add_argument('--skip', nargs='+', default=[], choices=STEP_KEYS)
    return p

def main():
    args = _build_parser().parse_args()
    ow = {k: True for k in STEP_KEYS} if args.overwrite_all else {k: (k in args.overwrite) for k in STEP_KEYS}
    sk = {k: (k in args.skip) for k in STEP_KEYS}
    sub = f"sub-{args.sub.removeprefix('sub-')}"
    ses = f"ses-{args.ses.removeprefix('ses-')}"

    run_pipeline(
        args.bids_dir, args.moco_file, args.output_file, sub, ses,
        args.fd_threshold, ow, sk
    )

if __name__ == '__main__':
    main()