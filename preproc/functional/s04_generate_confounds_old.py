#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.masking import apply_mask, compute_epi_mask
from nilearn.image import resample_to_img
from nilearn.glm.first_level import _compute_dvars
from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix

from sklearn.decomposition import PCA
from scipy.ndimage import binary_erosion


# -----------------------------
# FreeSurfer labels
# -----------------------------
WM_LABELS = [2, 41]
CSF_LABELS = [4, 43]


# -----------------------------
# Mask utilities
# -----------------------------
def fs_masks(aseg_nii, erosion_iters=1):
    img = nib.load(aseg_nii)
    data = img.get_fdata()

    wm = np.isin(data, WM_LABELS)
    csf = np.isin(data, CSF_LABELS)

    if erosion_iters > 0:
        wm = binary_erosion(wm, iterations=erosion_iters)
        csf = binary_erosion(csf, iterations=erosion_iters)

    wm_img = nib.Nifti1Image(wm.astype("uint8"), img.affine)
    csf_img = nib.Nifti1Image(csf.astype("uint8"), img.affine)

    nib.save(wm_img, "wm_mask.nii.gz")
    nib.save(csf_img, "csf_mask.nii.gz")

    return "wm_mask.nii.gz", "csf_mask.nii.gz"

# -----------------------------
# DVARS
# -----------------------------
def compute_dvars(
    bold,
    mask,
    remove_zerovariance=False,
    intensity_normalization=1000,
    variance_tol=0.0,
):
    """
    Compute the :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Power2012]_.

    Particularly, the *standardized* :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Nichols2013]_ are computed.

    .. [Nichols2013] Nichols T, `Notes on creating a standardized version of
         DVARS <http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-\
research/nichols/scripts/fsl/standardizeddvars.pdf>`_, 2013.

    .. note:: Implementation details

      Uses the implementation of the `Yule-Walker equations
      from nitime
      <http://nipy.org/nitime/api/generated/nitime.algorithms.autoregressive.html\
#nitime.algorithms.autoregressive.AR_est_YW>`_
      for the :abbr:`AR (auto-regressive)` filtering of the fMRI signal.

    :param numpy.ndarray func: functional data, after head-motion-correction.
    :param numpy.ndarray mask: a 3D mask of the brain
    :param bool output_all: write out all dvars
    :param str out_file: a path to which the standardized dvars should be saved.
    :return: the standardized DVARS

    """
    import numpy as np
    import nibabel as nb
    import warnings

    func = np.float32(nb.load(in_file).dataobj)
    mask = np.bool_(nb.load(in_mask).dataobj)

    if len(func.shape) != 4:
        raise RuntimeError("Input fMRI dataset should be 4-dimensional")

    mfunc = func[mask]

    if intensity_normalization != 0:
        mfunc = (mfunc / np.median(mfunc)) * intensity_normalization

    # Robust standard deviation (we are using "lower" interpolation
    # because this is what FSL is doing
    try:
        func_sd = (
            np.percentile(mfunc, 75, axis=1, method="lower")
            - np.percentile(mfunc, 25, axis=1, method="lower")
        ) / 1.349
    except TypeError:  # NP < 1.22
        func_sd = (
            np.percentile(mfunc, 75, axis=1, interpolation="lower")
            - np.percentile(mfunc, 25, axis=1, interpolation="lower")
        ) / 1.349

    if remove_zerovariance:
        zero_variance_voxels = func_sd > variance_tol
        mfunc = mfunc[zero_variance_voxels, :]
        func_sd = func_sd[zero_variance_voxels]

    # Compute (non-robust) estimate of lag-1 autocorrelation
    ar1 = np.apply_along_axis(
        _AR_est_YW, 1, regress_poly(0, mfunc, remove_mean=True)[0].astype(np.float32), 1
    )

    # Compute (predicted) standard deviation of temporal difference time series
    diff_sdhat = np.squeeze(np.sqrt(((1 - ar1) * 2).tolist())) * func_sd
    diff_sd_mean = diff_sdhat.mean()

    # Compute temporal difference time series
    func_diff = np.diff(mfunc, axis=1)

    # DVARS (no standardization)
    dvars_nstd = np.sqrt(np.square(func_diff).mean(axis=0))

    # standardization
    dvars_stdz = dvars_nstd / diff_sd_mean

    with warnings.catch_warnings():  # catch, e.g., divide by zero errors
        warnings.filterwarnings("error")

        # voxelwise standardization
        diff_vx_stdz = np.square(
            func_diff / np.array([diff_sdhat] * func_diff.shape[-1]).T
        )
        dvars_vx_stdz = np.sqrt(diff_vx_stdz.mean(axis=0))

    return (dvars_stdz, dvars_nstd, dvars_vx_stdz)

# -----------------------------
# aCompCor
# -----------------------------
def acompcor(bold_img, wm_mask, csf_mask, n_components):
    def pca(mask_img):
        X = apply_mask(bold_img, mask_img)
        X -= X.mean(0)
        return PCA(n_components=n_components).fit_transform(X)

    wm = pca(wm_mask)
    csf = pca(csf_mask)

    cols = (
        [f"a_comp_cor_wm_{i}" for i in range(n_components)] +
        [f"a_comp_cor_csf_{i}" for i in range(n_components)]
    )

    return pd.DataFrame(np.hstack([wm, csf]), columns=cols)


# -----------------------------
# tCompCor
# -----------------------------
def tcompcor(bold_img, variance_percent, n_components):
    mask = compute_epi_mask(bold_img)
    data = apply_mask(bold_img, mask)

    var = data.var(axis=0)
    thresh = np.percentile(var, 100 - variance_percent)
    high_var = data[:, var >= thresh]
    high_var -= high_var.mean(0)

    comps = PCA(n_components=n_components).fit_transform(high_var)

    return pd.DataFrame(
        comps,
        columns=[f"t_comp_cor_{i}" for i in range(n_components)]
    )


# -----------------------------
# Global signal
# -----------------------------
def global_signal(bold_img):
    mask = compute_epi_mask(bold_img)
    gs = apply_mask(bold_img, mask).mean(1)
    return pd.DataFrame({"global_signal": gs})


# -----------------------------
# Motion (TSV or mcflirt .par)
# -----------------------------
def load_motion(motion_file):
    if motion_file.endswith(".par"):
        motion = pd.read_csv(
            motion_file,
            delim_whitespace=True,
            header=None,
            names=["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"],
        )
        # Convert rotations (radians) to mm (Power FD)
        radius = 50.0
        motion[["rot_x", "rot_y", "rot_z"]] *= radius
    else:
        motion = pd.read_csv(motion_file, sep="\t")

    return motion


def motion_metrics(bold_img, motion_file, fd_thresh, dvars_thresh):
    motion = load_motion(motion_file)

    fd = motion.diff().abs().sum(1).fillna(0)
    dvars = _compute_dvars(bold_img)

    df = pd.DataFrame({
        "framewise_displacement": fd,
        "dvars": dvars
    })

    outliers = (fd > fd_thresh) | (dvars > dvars_thresh)

    for i, idx in enumerate(np.where(outliers)[0]):
        spike = np.zeros(len(df))
        spike[idx] = 1
        df[f"motion_outlier_{i}"] = spike

    return df


# -----------------------------
# Cosine drifts
# -----------------------------
def cosine_drifts(n_scans, tr, highpass):
    frame_times = np.arange(n_scans) * tr
    design = make_first_level_design_matrix(
        frame_times,
        drift_model="cosine",
        high_pass=1.0 / highpass
    )
    drift_cols = [c for c in design.columns if "cosine" in c]
    return design[drift_cols].reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main(args):
    bold_img = nib.load(args.bold)
    n_scans = bold_img.shape[-1]

    aseg = args.aseg
    if aseg.endswith(".mgz"):
        os.system(f"mri_convert {aseg} aseg.nii.gz")
        aseg = "aseg.nii.gz"

    wm_mask, csf_mask = fs_masks(aseg, args.erosion)

    wm_mask = resample_to_img(wm_mask, bold_img, interpolation="nearest")
    csf_mask = resample_to_img(csf_mask, bold_img, interpolation="nearest")

    nib.save(wm_mask, "wm_bold_mask.nii.gz")
    nib.save(csf_mask, "csf_bold_mask.nii.gz")

    confounds = [
        acompcor(bold_img, "wm_bold_mask.nii.gz", "csf_bold_mask.nii.gz", args.acompcor),
        tcompcor(bold_img, args.tcompcor_percent, args.tcompcor),
        global_signal(bold_img),
        motion_metrics(bold_img, args.motion, args.fd_thresh, args.dvars_thresh),
        cosine_drifts(n_scans, args.tr, args.highpass),
    ]

    df = pd.concat(confounds, axis=1)
    df.to_csv(args.out, sep="\t", index=False)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone fMRI confound extraction (fMRIPrep-like)"
    )

    parser.add_argument("--bold", required=True)
    parser.add_argument("--aseg", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--out", default="confounds.tsv")

    parser.add_argument("--acompcor", type=int, default=5)
    parser.add_argument("--tcompcor", type=int, default=6)
    parser.add_argument("--tcompcor-percent", type=float, default=2.0)

    parser.add_argument("--erosion", type=int, default=1)

    parser.add_argument("--tr", type=float, required=True)
    parser.add_argument("--highpass", type=float, default=128.0)

    parser.add_argument("--fd-thresh", type=float, default=0.5)
    parser.add_argument("--dvars-thresh", type=float, default=1.5)

    args = parser.parse_args()
    main(args)
