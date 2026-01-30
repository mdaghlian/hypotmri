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


# FreeSurfer aseg labels (aseg.mgz)
WM_LABELS = {2, 41}                 # Left/Right Cerebral-White-Matter
GM_CORTEX_LABELS = {3, 42}          # Left/Right Cerebral-Cortex
CSF_LABELS = {4, 43, 14, 15, 24}    # Ventricles + CSF-ish structures


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def save_mask_like(ref_img, mask_bool, out_path):
    img = nib.Nifti1Image(mask_bool.astype("uint8"), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, out_path)
    return out_path


def aseg_to_masks(aseg_mgz, bold_nii, outdir, gm_dilate_iter=1):
    """
    Make brain/WM/CSF masks in BOLD space from aseg labels, with a simple
    GM-dilation cleanup to reduce WM/CSF partial voluming.
    """
    ensure_dir(outdir)
    aseg_img = nib.load(aseg_mgz)  # .mgz ok
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

    brain_p = save_mask_like(bold_img, brain,    os.path.join(outdir, "mask_brain.nii.gz"))
    wm_p    = save_mask_like(bold_img, wm_clean, os.path.join(outdir, "mask_wm.nii.gz"))
    csf_p   = save_mask_like(bold_img, csf_clean, os.path.join(outdir, "mask_csf.nii.gz"))
    wmcsf_p = save_mask_like(bold_img, wmcsf,    os.path.join(outdir, "mask_wmcsf.nii.gz"))
    return brain_p, wm_p, csf_p, wmcsf_p


def load_mcflirt_par(par_file):
    mp = np.loadtxt(par_file)
    if mp.ndim != 2 or mp.shape[1] != 6:
        raise ValueError(f"Expected Nx6 motion params, got {mp.shape}")
    return mp


def backward_diff(x: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(x, dtype=float)
    dx[1:] = x[1:] - x[:-1]
    return dx


def add_fmriprep_expansions(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    Add fMRIPrep-style derivative + quadratic expansions for the given base columns:
    _derivative1, _power2, _derivative1_power2
    """
    out = df.copy()

    deriv = pd.DataFrame({f"{c}_derivative1": backward_diff(out[c].to_numpy()) for c in base_cols})
    power2 = pd.DataFrame({f"{c}_power2": (out[c].to_numpy() ** 2) for c in base_cols})
    deriv_power2 = pd.DataFrame({f"{c}_derivative1_power2": (deriv[f"{c}_derivative1"].to_numpy() ** 2)
                                 for c in base_cols})

    return pd.concat([out, deriv, power2, deriv_power2], axis=1)


def motion_df_from_mcflirt(par_file):
    """
    fMRIPrep column names: trans_x/y/z, rot_x/y/z
    + expansions to 24 motion params.
    """
    mp = load_mcflirt_par(par_file)
    rot = mp[:, 0:3]
    trans = mp[:, 3:6]

    base = pd.DataFrame({
        "trans_x": trans[:, 0], "trans_y": trans[:, 1], "trans_z": trans[:, 2],
        "rot_x":   rot[:, 0],   "rot_y":   rot[:, 1],   "rot_z":   rot[:, 2],
    })
    base = add_fmriprep_expansions(base, ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"])
    return base


def read_single_column_numeric(path):
    """
    Nipype sometimes writes a header line. Read first column as float.
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
    for c in candidates:
        if hasattr(outputs, c):
            p = getattr(outputs, c)
            if p and isinstance(p, str):
                return p
    return None


def enforce_length_1d(name: str, x: np.ndarray | None, T: int, pad_value=np.nan) -> np.ndarray:
    """
    Enforce a 1D series of length T.
    Accepts lengths:
      - T   : return as-is
      - T-1 : pad 1 at start
      - T-2 : pad 2 at start
    Otherwise: error.
    """
    if x is None:
        return np.full((T,), np.nan, dtype=float)

    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) == T:
        return x
    if len(x) in (T - 1, T - 2):
        n_pad = T - len(x)
        return np.r_[np.full((n_pad,), pad_value, dtype=float), x]

    raise ValueError(f"{name} length ({len(x)}) != T ({T}), T-1 ({T-1}), or T-2 ({T-2}).")


def mean_signal(bold_nii, mask_nii):
    bold_img = nib.load(bold_nii)
    mask_img = nib.load(mask_nii)
    X = apply_mask(bold_img, mask_img).astype(np.float64)  # (T, V)
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

    fd_path = pick_existing_output(res.outputs, ["out_file", "fd_file", "framewise_displacement"])
    if fd_path is None:
        raise RuntimeError(f"Could not locate FD output in: {res.outputs}")

    return read_single_column_numeric(fd_path)


def run_dvars(bold_nii, brain_mask_nii, tr, workdir):
    dv = ComputeDVARS()
    dv.inputs.in_file = bold_nii
    dv.inputs.in_mask = brain_mask_nii
    dv.inputs.series_tr = float(tr)
    dv.inputs.save_std = True
    dv.inputs.save_nstd = True  # 
    dv.inputs.save_plot = False
    dv.inputs.intensity_normalization = 1000.0

    res = dv.run(cwd=workdir)

    std_path = pick_existing_output(res.outputs, ["out_std", "std_file", "std_out"])
    nstd_path = pick_existing_output(res.outputs, ["out_nstd", "nstd_file", "nstd_out", "out_all"])

    dvars_std = read_single_column_numeric(std_path) if std_path else None
    dvars_nstd = read_single_column_numeric(nstd_path) if nstd_path else None
    return dvars_nstd, dvars_std


def run_acompcor_single(bold_nii, wmcsf_mask_nii, tr, workdir, variance_threshold=0.5):
    """
    Single aCompCor set from combined WM+CSF mask, named a_comp_cor_00.. etc.
    """
    cc = CompCor()
    cc.inputs.realigned_file = bold_nii
    cc.inputs.mask_files = [wmcsf_mask_nii]
    cc.inputs.merge_method = "union"
    cc.inputs.pre_filter = "cosine"
    cc.inputs.high_pass_cutoff = 128.0
    cc.inputs.repetition_time = float(tr)
    cc.inputs.variance_threshold = float(variance_threshold)
    cc.inputs.header_prefix = "a_comp_cor"
    cc.inputs.save_metadata = False

    res = cc.run(cwd=workdir)

    comp_path = pick_existing_output(res.outputs, ["components_file"])
    if comp_path is None or not os.path.exists(comp_path):
        raise RuntimeError(f"Could not locate CompCor components_file in: {res.outputs}")

    comps = pd.read_csv(comp_path, sep=r"\s+|,|\t", engine="python")

    # Normalize column names to exactly a_comp_cor_00, a_comp_cor_01, ...
    # (nipype usually does this already, but we enforce for safety)
    new_cols = []
    for i, c in enumerate(comps.columns):
        if c.startswith("a_comp_cor_"):
            new_cols.append(c)
        else:
            new_cols.append(f"a_comp_cor_{i:02d}")
    comps.columns = new_cols
    return comps


def cosine_drift_terms(T: int, tr: float, cutoff: float = 128.0) -> pd.DataFrame:
    """
    Create fMRIPrep-style discrete cosine-basis regressors `cosine_XX`.

    Implementation: DCT basis (non-constant terms) with a 128s cutoff.
    Matches the general SPM/nilearn-style construction used for drift modeling.
    fMRIPrep documents these as `cosine_XX`. :contentReference[oaicite:0]{index=0}

    Note: fMRIPrep uses "effective length" excluding nonsteady-state volumes; this
    function uses all T timepoints (no NSS detection here).
    """
    # Try nilearn's internal function if available (often present)
    frame_times = np.arange(T, dtype=float) * float(tr)

    try:
        # nilearn private API (best-effort)
        from nilearn.glm.first_level.design_matrix import _cosine_drift
        C = _cosine_drift(float(tr), frame_times, float(cutoff))
        # _cosine_drift returns (T, K) including intercept? In nilearn it returns
        # only drift terms (no intercept). We still robustly drop any constant column.
        C = np.asarray(C, dtype=float)
    except Exception:
        # Fallback: standard DCT drift terms (no intercept)
        # Number of terms: floor(2 * scan_duration / cutoff)
        scan_duration = T * float(tr)
        K = int(np.floor(2.0 * scan_duration / float(cutoff)))
        if K < 1:
            return pd.DataFrame(index=np.arange(T))
        n = np.arange(T, dtype=float)
        # DCT-II style basis (common in SPM-like drift models)
        C = np.column_stack([
            np.cos((np.pi * (2.0 * n + 1.0) * k) / (2.0 * T))
            for k in range(1, K + 1)
        ]).astype(float)

    # Drop any (near-)constant columns just in case
    keep = []
    for j in range(C.shape[1]):
        if np.nanstd(C[:, j]) > 1e-12:
            keep.append(j)
    C = C[:, keep] if keep else np.zeros((T, 0), dtype=float)

    cols = [f"cosine_{i:02d}" for i in range(C.shape[1])]
    return pd.DataFrame(C, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold", required=True, help="4D BOLD NIfTI (already motion-corrected, in FS space ok)")
    ap.add_argument("--mcpar", required=True, help="MCFLIRT .par file (Nx6: rotX rotY rotZ transX transY transZ)")
    ap.add_argument("--aseg", required=True, help="FreeSurfer aseg.mgz (or NIfTI) aligned to BOLD/anat")
    ap.add_argument("--tr", required=True, type=float, help="TR (seconds)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--outfile", default="cf")
    ap.add_argument("--gm-dilate-iter", type=int, default=1)
    ap.add_argument("--acompcor-variance", type=float, default=0.5)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    outfile = args.outfile
    workdir = ensure_dir(os.path.join(outdir, outfile))
    masks_dir = ensure_dir(os.path.join(outdir, "masks"))

    bold_img = nib.load(args.bold)
    if bold_img.ndim != 4:
        raise ValueError("BOLD must be 4D.")
    T = int(bold_img.shape[3])

    # --- Length sanity checks (BOLD is truth) ---
    mp = load_mcflirt_par(args.mcpar)
    if mp.shape[0] != T:
        raise ValueError(f"Motion rows ({mp.shape[0]}) != BOLD timepoints ({T}). "
                         "If you dropped volumes, drop matching rows everywhere.")

    # --- Masks (aseg-based) ---
    brain_mask, wm_mask, csf_mask, wmcsf_mask = aseg_to_masks(
        args.aseg, args.bold, masks_dir, gm_dilate_iter=args.gm_dilate_iter
    )

    # --- Base confounds: motion (with expansions to 24 params) ---
    conf = motion_df_from_mcflirt(args.mcpar)

    # --- Global signals + expansions (fMRIPrep supports 36p strategy) :contentReference[oaicite:1]{index=1} ---
    conf["global_signal"] = mean_signal(args.bold, brain_mask)
    conf["white_matter"] = mean_signal(args.bold, wm_mask)
    conf["csf"] = mean_signal(args.bold, csf_mask)
    conf = add_fmriprep_expansions(conf, ["global_signal", "white_matter", "csf"])

    # --- FD ---
    fd_vals = run_fd(args.mcpar, args.tr, workdir)
    conf["framewise_displacement"] = enforce_length_1d(
        "framewise_displacement", fd_vals, T, pad_value=np.nan
    )

    # --- DVARS (fMRIPrep names: dvars, std_dvars) :contentReference[oaicite:2]{index=2} ---
    dvars, std_dvars = run_dvars(args.bold, brain_mask, args.tr, workdir)
    conf["dvars"] = enforce_length_1d("dvars", dvars, T, pad_value=np.nan)
    conf["std_dvars"] = enforce_length_1d("std_dvars", std_dvars, T, pad_value=np.nan)

    # --- aCompCor: single set from combined WM+CSF mask ---
    acomps = run_acompcor_single(args.bold, wmcsf_mask, args.tr, workdir, args.acompcor_variance)
    # enforce length
    if acomps.shape[0] != T:
        raise ValueError(f"aCompCor rows ({acomps.shape[0]}) != BOLD timepoints ({T}).")
    for c in acomps.columns:
        conf[c] = acomps[c].to_numpy(dtype=float)

    # --- Cosine drift regressors (128s cutoff) :contentReference[oaicite:3]{index=3} ---
    cos = cosine_drift_terms(T, args.tr, cutoff=128.0)
    for c in cos.columns:
        conf[c] = cos[c].to_numpy(dtype=float)

    # --- fMRIPrep-ish ordering ---
    motion_base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    motion_deriv = [f"{c}_derivative1" for c in motion_base]
    motion_pow2 = [f"{c}_power2" for c in motion_base]
    motion_deriv_pow2 = [f"{c}_derivative1_power2" for c in motion_base]

    gs_base = ["global_signal", "white_matter", "csf"]
    gs_deriv = [f"{c}_derivative1" for c in gs_base]
    gs_pow2 = [f"{c}_power2" for c in gs_base]
    gs_deriv_pow2 = [f"{c}_derivative1_power2" for c in gs_base]

    acomp_cols = sorted([c for c in conf.columns if c.startswith("a_comp_cor_")])
    cosine_cols = sorted([c for c in conf.columns if c.startswith("cosine_")])

    preferred = (
        motion_base + motion_deriv + motion_pow2 + motion_deriv_pow2 +
        gs_base + gs_deriv + gs_pow2 + gs_deriv_pow2 +
        ["framewise_displacement", "dvars", "std_dvars"] +
        acomp_cols +
        cosine_cols
    )
    # Append anything not covered (should be none, but safe)
    remaining = [c for c in conf.columns if c not in preferred]
    conf = conf[preferred + remaining]

    out_tsv = os.path.join(outdir, f"{outfile}.tsv")
    conf.to_csv(out_tsv, sep="\t", index=False, na_rep="n/a")
    print(out_tsv)


if __name__ == "__main__":
    main()
