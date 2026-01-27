#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd

from scipy.ndimage import binary_dilation
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from nipype.algorithms.confounds import FramewiseDisplacement, ComputeDVARS, CompCor


WM_LABELS = {2, 41}
GM_CORTEX_LABELS = {3, 42}
CSF_LABELS = {4, 43, 14, 15, 24}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_mask_like(ref_img, mask_bool, out_path):
    img = nib.Nifti1Image(mask_bool.astype("uint8"), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, out_path)
    return out_path


def aseg_to_masks(aseg_mgz, bold_nii, outdir, gm_dilate_iter=1):
    ensure_dir(outdir)
    aseg_img = nib.load(aseg_mgz)      # .mgz ok
    bold_img = nib.load(bold_nii)

    # Label-resample to BOLD grid
    aseg_rs = resample_to_img(aseg_img, bold_img, interpolation="nearest")
    aseg = np.asanyarray(aseg_rs.dataobj).astype(np.int32)

    brain = aseg > 0
    wm = np.isin(aseg, list(WM_LABELS))
    gm = np.isin(aseg, list(GM_CORTEX_LABELS))
    csf = np.isin(aseg, list(CSF_LABELS))

    gm_dil = binary_dilation(gm, iterations=int(gm_dilate_iter)) if gm_dilate_iter > 0 else gm
    wm_clean = wm & (~gm_dil)
    csf_clean = csf & (~gm_dil)
    wmcsf = wm_clean | csf_clean

    brain_p = save_mask_like(bold_img, brain,  os.path.join(outdir, "mask_brain.nii.gz"))
    wm_p    = save_mask_like(bold_img, wm_clean, os.path.join(outdir, "mask_wm.nii.gz"))
    csf_p   = save_mask_like(bold_img, csf_clean, os.path.join(outdir, "mask_csf.nii.gz"))
    wmcsf_p = save_mask_like(bold_img, wmcsf, os.path.join(outdir, "mask_wmcsf.nii.gz"))
    return brain_p, wm_p, csf_p, wmcsf_p


def load_mcflirt_par(par_file):
    mp = np.loadtxt(par_file)
    if mp.ndim != 2 or mp.shape[1] != 6:
        raise ValueError(f"Expected Nx6 motion params, got {mp.shape}")
    return mp


def backward_diff(x):
    dx = np.zeros_like(x)
    dx[1:] = x[1:] - x[:-1]
    return dx


def motion_df_from_mcflirt(par_file):
    mp = load_mcflirt_par(par_file)
    rot = mp[:, 0:3]
    trans = mp[:, 3:6]

    base = pd.DataFrame({
        "trans_x": trans[:, 0], "trans_y": trans[:, 1], "trans_z": trans[:, 2],
        "rot_x":   rot[:, 0],   "rot_y":   rot[:, 1],   "rot_z":   rot[:, 2],
    })

    deriv = base.apply(lambda c: backward_diff(c.to_numpy()))
    deriv.columns = [f"{c}_derivative1" for c in base.columns]

    power2 = base ** 2
    power2.columns = [f"{c}_power2" for c in base.columns]

    deriv_power2 = deriv ** 2
    deriv_power2.columns = [f"{c}_derivative1_power2" for c in base.columns]

    return pd.concat([base, deriv, power2, deriv_power2], axis=1)


def read_single_column_numeric(path):
    """
    Nipype often writes headers (e.g., 'FramewiseDisplacement').
    Use pandas, grab the first column, cast to float.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected output file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+|,|\t", engine="python")
    return df.iloc[:, 0].astype(float).to_numpy()


def pick_existing_output(outputs, candidates):
    for c in candidates:
        if hasattr(outputs, c):
            p = getattr(outputs, c)
            if p and isinstance(p, str) and os.path.exists(p):
                return p
    # sometimes nipype returns a path that exists later; check non-empty as fallback
    for c in candidates:
        if hasattr(outputs, c):
            p = getattr(outputs, c)
            if p and isinstance(p, str):
                return p
    return None


def mean_signal(bold_nii, mask_nii):
    bold_img = nib.load(bold_nii)
    mask_img = nib.load(mask_nii)
    X = apply_mask(bold_img, mask_img).astype(np.float64)  # (T,V)
    if X.shape[1] == 0:
        return np.full((bold_img.shape[3],), np.nan)
    return X.mean(axis=1)


def run_fd(par_file, tr, workdir):
    fd = FramewiseDisplacement()
    fd.inputs.in_file = par_file
    fd.inputs.parameter_source = "FSL"
    fd.inputs.series_tr = float(tr)
    fd.inputs.save_plot = False

    res = fd.run(cwd=workdir)

    # robustly find the FD file
    fd_path = pick_existing_output(res.outputs, ["out_file", "fd_file", "framewise_displacement"])
    if fd_path is None:
        raise RuntimeError(f"Could not locate FD output in: {res.outputs}")

    # at this point, Nipype has created it; read with header-safe reader
    return read_single_column_numeric(fd_path)


def run_dvars(bold_nii, brain_mask_nii, tr, workdir):
    dv = ComputeDVARS()
    dv.inputs.in_file = bold_nii
    dv.inputs.in_mask = brain_mask_nii
    dv.inputs.series_tr = float(tr)
    dv.inputs.save_std = True
    dv.inputs.save_nstd = True
    dv.inputs.save_plot = False
    dv.inputs.intensity_normalization = 1000.0

    res = dv.run(cwd=workdir)

    std_path = pick_existing_output(res.outputs, ["out_std", "std_file", "std_out"])
    nstd_path = pick_existing_output(res.outputs, ["out_nstd", "nstd_file", "nstd_out"])

    dvars_std = read_single_column_numeric(std_path) if std_path else None
    dvars_nstd = read_single_column_numeric(nstd_path) if nstd_path else None
    return dvars_std, dvars_nstd


def run_compcor(bold_nii, mask_nii, tr, workdir, header_prefix, variance_threshold=0.5):
    cc = CompCor()
    cc.inputs.realigned_file = bold_nii
    cc.inputs.mask_files = [mask_nii]
    cc.inputs.pre_filter = "cosine"
    cc.inputs.high_pass_cutoff = 128.0
    cc.inputs.repetition_time = float(tr)
    cc.inputs.variance_threshold = float(variance_threshold)
    cc.inputs.header_prefix = header_prefix
    cc.inputs.save_metadata = False

    res = cc.run(cwd=workdir)

    comp_path = pick_existing_output(res.outputs, ["components_file"])
    if comp_path is None or not os.path.exists(comp_path):
        raise RuntimeError(f"Could not locate CompCor components_file in: {res.outputs}")

    comps = np.loadtxt(comp_path)
    comps = pd.read_csv(comp_path, sep=r"\s+|,|\t", engine="python")
    if comps.ndim == 1:
        comps = comps[:, None]
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold", required=True)
    ap.add_argument("--mcpar", required=True)
    ap.add_argument("--aseg", required=True)
    ap.add_argument("--tr", required=True, type=float)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gm-dilate-iter", type=int, default=1)
    ap.add_argument("--acompcor-variance", type=float, default=0.5)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    workdir = ensure_dir(os.path.join(outdir, "nipype_work"))
    masks_dir = ensure_dir(os.path.join(outdir, "masks"))

    bold_img = nib.load(args.bold)
    T = bold_img.shape[3]
    
    
    # motion df (this is the only thing that "reads" your motion estimates)
    conf = motion_df_from_mcflirt(args.mcpar)

    T_bold = nib.load(args.bold).shape[3]
    T_par = np.loadtxt(args.mcpar).shape[0]
    print("T_bold:", T_bold, "T_par:", T_par)
    
    # sanity check alignment of rows
    if conf.shape[0] != T:
        raise ValueError(f"Motion rows ({conf.shape[0]}) != BOLD timepoints ({T}). "
                         "If you dropped volumes, drop matching rows everywhere.")

    # masks
    brain_mask, wm_mask, csf_mask, wmcsf_mask = aseg_to_masks(
        args.aseg, args.bold, masks_dir, gm_dilate_iter=args.gm_dilate_iter
    )

    # FD (nipype generates file; we read it robustly)
    fd_vals = run_fd(args.mcpar, args.tr, workdir)
    if T_bold != len(fd_vals):
        # if some versions output T-1, pad
        print('Padding 0th FD value')
        if len(fd_vals) == T_bold - 1:
            fd_vals = np.r_[0.0, fd_vals]
        else:
            raise ValueError(f"FD length ({len(fd_vals)}) != BOLD timepoints ({T}).")
    conf["framewise_displacement"] = fd_vals
    
    # DVARS
    dvars_std, dvars_nstd = run_dvars(args.bold, brain_mask, args.tr, workdir)
    if (dvars_nstd is not None) and (len(dvars_nstd) == T_bold - 2):
        print('Padding 0th and 1st DVARS_nstd values')
        dvars_std = np.r_[0.0, 0.0, dvars_std]
        dvars_nstd = np.r_[0.0, 0.0, dvars_nstd]

    print("len(dvars_nstd):", None if dvars_nstd is None else len(dvars_nstd))
    print("len(dvars_std):", None if dvars_std is None else len(dvars_std))
 
    conf["dvars"] = dvars_nstd
    conf["dvars_standardized"] = dvars_std
    
    # aCompCor (wm/csf/wmcsf)
    acomp_wm = run_compcor(args.bold, wm_mask, args.tr, workdir, "a_comp_cor_wm", args.acompcor_variance)
    acomp_csf = run_compcor(args.bold, csf_mask, args.tr, workdir, "a_comp_cor_csf", args.acompcor_variance)
    acomp_wmcsf = run_compcor(args.bold, wmcsf_mask, args.tr, workdir, "a_comp_cor", args.acompcor_variance)
 
    print(len(acomp_wm), acomp_wm.shape)
    print(len(acomp_csf), acomp_csf.shape)
    print(len(acomp_wmcsf), acomp_wmcsf.shape)
    
    for i in range(acomp_wm.shape[1]):
        conf[f"a_comp_cor_wm_{i:02d}"] = acomp_wm[:, i]
    for i in range(acomp_csf.shape[1]):
        conf[f"a_comp_cor_csf_{i:02d}"] = acomp_csf[:, i]
    for i in range(acomp_wmcsf.shape[1]):
        conf[f"a_comp_cor_{i:02d}"] = acomp_wmcsf[:, i]

    # mean signals (approx)
    conf["global_signal"] = mean_signal(args.bold, brain_mask)
    conf["white_matter"] = mean_signal(args.bold, wm_mask)
    conf["csf"] = mean_signal(args.bold, csf_mask)

    out_tsv = os.path.join(outdir, "confounds_timeseries.tsv")
    conf.to_csv(out_tsv, sep="\t", index=False, na_rep="n/a")
    print(out_tsv)


if __name__ == "__main__":
    main()
