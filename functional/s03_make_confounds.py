#!/usr/bin/env python
"""
s03_make_confounds.py
======================
Compute aCompCor and tCompCor noise regressors directly from this pipeline's
own outputs — s02_coreg.py's coregistered BOLD + s01_fmriprep_anat_only.sh's
tissue probability maps — WITHOUT running a full fMRIPrep BOLD workflow.

Why this works
--------------
fMRIPrep's own aCompCor/tCompCor numbers come from `nipype.algorithms.
confounds.ACompCor` / `TCompCor` (fMRIPrep's `RobustACompCor`/
`RobustTCompCor` in niworkflows are just those same classes wrapped in a
retry-on-LinAlgError loop — numerically identical for well-behaved data).
What's fMRIPrep-specific is (1) how the aCompCor tissue masks are built and
(2) the exact parameters passed in. This script replicates both, following
`fmriprep.utils.confounds.acompcor_masks()` and
`fmriprep.workflows.bold.confounds.init_bold_confs_wf()`
(nipreps/fmriprep, Apache-2.0):

  1) aCompCor masks (subject-level, built once, shared across sessions):
     - fMRIPrep has two branches, picked by whether FreeSurfer was run
       (`aCompCorMasks(is_aseg=run_reconall)`). Both start from the same
       FAST probseg maps; the default here (--freesurfer) is the
       FreeSurfer one, which is what a standard fMRIPrep run uses:
         freesurfer : GM = uint8(GM probseg) > 0 (i.e. p == 1 only);
                      CSF and WM maps Gaussian-smoothed (sigma = 0.5 *
                      BOLD zoom / anat zoom) and divided by their 99th
                      percentile (fMRIPrep's mask2vf)
         fast       : GM = GM probseg > 0.05 (--no-freesurfer)
     - dilate GM by a 3-voxel ball, subtract it from the WM and combined
       CSF+WM maps (fMRIPrep's deviation from Behzadi et al.'s erosion)
     - resample onto the BOLD grid with antsApplyTransforms -n Gaussian
       (fMRIPrep's acc_msk_tfm), mask by brain, binarize at 0.99
     Note: fMRIPrep resamples its T1w-space masks into BOLD space via its
     own boldref->T1w transform. s02_coreg.py's BOLD is already in FS-T1
     world space, and the fMRIPrep-anat probseg maps describe the same
     physical anatomy in the same world (scanner) coordinates, so an
     identity transform is used (fMRIPrep's estimate of it is ~0.1 mm).
     The BOLD grid (2 mm) is not the FS-T1 grid (1 mm): masks are built on
     volume 0 of the first BOLD run, and every run must share that grid.
     The brain mask is desc-fsbrain > 0 resampled onto the BOLD grid
     (fMRIPrep uses its BOLD-derived mask; this only changes tCompCor and
     global_signal, not aCompCor).
  2) tCompCor: run directly on the coregistered BOLD using a whole-brain
     mask (top 2% variance voxels, no tissue segmentation needed).
  3) aCompCor: run jointly across CSF / WM / combined masks, then split and
     renumber per fMRIPrep's RenameACompCor so the "combined" mask's output
     becomes the canonical a_comp_cor_XX columns.

Cosine (128s DCT high-pass) regressors and simple csf/white_matter/
global_signal mean-signal regressors are also written, so the merged
`desc-confounds_timeseries.tsv` this script produces has the same column
names fMRIPrep's own confounds file has, and can be dropped directly into
s04_confounds.py's `extract_fmriprep_confounds()` in place of a real
fMRIPrep BOLD run's confounds file.

Prerequisites (already produced upstream in this pipeline)
------------------------------------------------------------
  - s01_fmriprep_anat_only.sh  -> derivatives/<anat-file>/<sub>/[ses/]anat/
        *label-GM_probseg.nii.gz, *label-WM_probseg.nii.gz,
        *label-CSF_probseg.nii.gz
  - s02_coreg.py                -> derivatives/<coreg-file>/<sub>/
        sub-XX_desc-fsbrain.nii.gz                                  (subject-level)
        sub-XX_ses-YY_{task}_{run}_space-fsT1_desc-moco_bbreg_bold.nii.gz

Overwrite behaviour
-------------------
Existence is checked against final BIDS-named files in *output_dir*.

Skip behaviour
--------------
Steps can be force-skipped with --skip regardless of whether their outputs
exist. No files are touched. Downstream steps continue regardless — caller
is responsible for dependencies.

Arguments
---------
    --bids-dir      BIDS directory containing input and output derivatives
    --coreg-file    derivatives/<dir> with s02_coreg.py outputs
    --anat-file     derivatives/<dir> with s01_fmriprep_anat_only.sh outputs
                    (default: fmriprep)
    --output-file   Name of derivatives directory to write confounds to
    --sub           Subject label (e.g. sub-01)
    --ses           Session label (e.g. ses-01)
    --skip-vols     Non-steady-state volumes to ignore (default: auto, detected
                    per run as fMRIPrep's NonsteadyStatesDetector)
    --variance-threshold, --tcompcor-percentile, --high-pass-cutoff,
    --gm-threshold, --gm-dilate-radius, --acompcor-bin-threshold
                    fMRIPrep-matching CompCor parameters (see defaults below)
    --no-freesurfer Use fMRIPrep's no-FreeSurfer aCompCor mask branch
    --docker        Image with antsApplyTransforms (Docker tag, .sif in
                    $SIF_DIR, or "local"; default: $FPREP_IMAGE)

Usage example
-------------
python s03_make_confounds.py \\
    --bids-dir /path/to/bids_dir \\
    --coreg-file s2_coreg \\
    --anat-file fmriprep \\
    --output-file s3_make_confounds \\
    --sub 01 \\
    --ses 01
"""

import argparse
import glob
import os
opj = os.path.join
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nipype.algorithms.confounds import ACompCor, TCompCor
from scipy.ndimage import binary_dilation, gaussian_filter

from cvl_utils.preproc_func import (
    build_output_name,
    check_skip,
    extract_volume,
    get_labels,
    make_safe_workdir,
    run_cmd,
    _container_path,
    _get_tr,
)

# ---------------------------------------------------------------------------
# Step keys
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'acompcor_masks',   # subject-level: CSF / WM / combined / brain masks on BOLD grid
    'tcompcor',          # per-run: TCompCor + cosine (DCT) basis
    'acompcor',          # per-run: ACompCor (CSF/WM/combined) -> renamed columns
    'gather_confounds',  # per-run: merge into desc-confounds_timeseries.tsv
]


# ---------------------------------------------------------------------------
# aCompCor mask construction (subject-level)
# ---------------------------------------------------------------------------

def _ball_structure(radius: int) -> np.ndarray:
    """
    Spherical structuring element, equivalent to skimage.morphology.ball(),
    without adding a scikit-image dependency.
    """
    r = int(radius)
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2 + zz ** 2) <= r ** 2


def _find_anat_file(anat_dir: str, subject: str, session: str, suffix: str) -> str:
    """
    Locate a single fMRIPrep anatomical-derivatives file under
    <anat_dir>/<subject>/**/anat/, matching '<subject>*<suffix>'. fMRIPrep
    may or may not nest anat outputs under a session folder depending on how
    many sessions were staged for the anat-only run, so the search is
    recursive; if more than one match is found, one containing *session* in
    its path is preferred. Template-space copies (*_space-*) are ignored, so
    only the T1w-space map is returned.
    """
    search = opj(anat_dir, subject, '**', 'anat', '{}*{}'.format(subject, suffix))
    hits = sorted(h for h in glob.glob(search, recursive=True)
                  if '_space-' not in os.path.basename(h))
    if not hits:
        raise FileNotFoundError(
            'No fMRIprep anat file matching "{}" found under {}/{}. '
            'Run s01_fmriprep_anat_only.sh first.'.format(suffix, anat_dir, subject))
    ses_hits = [h for h in hits if session in h]
    return ses_hits[0] if ses_hits else hits[0]


def _mask2vf(img: nib.Nifti1Image, zooms: np.ndarray) -> np.ndarray:
    """
    fMRIPrep's `mask2vf()`: turn a (FreeSurfer-route) tissue map into a
    volume-fraction-like map by Gaussian smoothing with sigma scaled by the
    BOLD/anat zoom ratio, then normalising by the 99th percentile.
    """
    sigma = 0.5 * (np.asarray(zooms, dtype=float)
                   / np.array(img.header.get_zooms()[:3], dtype=float))
    data = gaussian_filter(img.get_fdata(dtype=np.float32), sigma=sigma)
    max_data = np.percentile(data[data > 0], 99)
    return np.clip(data / max_data, a_min=0, a_max=1)


def build_acompcor_masks(
    gm_probseg: str,
    wm_probseg: str,
    csf_probseg: str,
    fs_t1_ref: str,
    bold_ref: str,
    work_dir: str,
    docker_image: str,
    freesurfer: bool = True,
    gm_threshold: float = 0.05,
    gm_dilate_radius: int = 3,
    bin_threshold: float = 0.99,
) -> dict:
    """
    Build the CSF / WM / combined aCompCor masks and a whole-brain mask on
    the BOLD grid, replicating fMRIPrep's
    `fmriprep.utils.confounds.acompcor_masks()` and the resample -> mask ->
    binarize chain in `init_bold_confs_wf` (nipreps/fmriprep, Apache-2.0
    licensed):

      - GM exclusion: *freesurfer* -> uint8(GM probseg) > 0 (fMRIPrep's
        is_aseg branch); else GM probseg > *gm_threshold*. Dilated by a
        *gm_dilate_radius*-voxel ball.
      - *freesurfer*: CSF and WM maps passed through _mask2vf()
      - dilated GM subtracted from WM and combined (CSF+WM) maps
        (CSF itself is passed through unmodified — CSF regions are small,
        so fMRIPrep does not erode/correct them)
      - resampled onto *bold_ref*'s grid with antsApplyTransforms
        -n Gaussian (identity transform), masked by brain, binarized at
        *bin_threshold*

    Parameters
    ----------
    gm_probseg, wm_probseg, csf_probseg : T1w-space tissue probability maps
        from the anat-only fMRIPrep run (label-{GM,WM,CSF}_probseg.nii.gz).
    fs_t1_ref : FreeSurfer T1 NIfTI from s02_coreg.py
        (sub-XX_desc-fsbrain.nii.gz), used to build the brain mask.
    bold_ref : 3D volume on the grid every coregistered BOLD run shares —
        the resample target.
    work_dir : scratch directory for intermediate files (no spaces; it is
        mounted into the container).
    docker_image : image providing antsApplyTransforms, passed to run_cmd().
    freesurfer : use fMRIPrep's FreeSurfer (is_aseg) mask branch.

    Returns
    -------
    dict with keys 'csf', 'wm', 'combined', 'brain' -> mask NIfTI paths.
    """
    gm_img  = nib.load(gm_probseg)
    wm_img  = nib.load(wm_probseg)
    csf_img = nib.load(csf_probseg)
    ref_img = nib.load(bold_ref)
    zooms   = np.array(ref_img.header.get_zooms()[:3], dtype=float)

    if freesurfer:
        gm_data  = np.asanyarray(gm_img.dataobj, np.uint8) > 0
        wm_data  = _mask2vf(wm_img, zooms)
        csf_data = _mask2vf(csf_img, zooms)
    else:
        gm_data  = gm_img.get_fdata(dtype=np.float32) > gm_threshold
        wm_data  = wm_img.get_fdata(dtype=np.float32).copy()
        csf_data = csf_img.get_fdata(dtype=np.float32).copy()

    gm_dilated = binary_dilation(gm_data, structure=_ball_structure(gm_dilate_radius))
    wm_data[gm_dilated] = 0

    combined_data = csf_data + wm_data
    combined_data[gm_dilated] = 0
    # csf_data itself is left unmodified, matching fmriprep's acompcor_masks()

    def _save_native(data, name):
        out = opj(work_dir, 'acompcor_{}_native.nii.gz'.format(name))
        nib.save(nib.Nifti1Image(data.astype(np.float32), gm_img.affine, gm_img.header), out)
        return out

    native = {
        'csf':      _save_native(csf_data, 'csf'),
        'wm':       _save_native(wm_data, 'wm'),
        'combined': _save_native(combined_data, 'combined'),
    }

    ref_stage = opj(work_dir, 'bold_ref.nii.gz')
    if os.path.abspath(bold_ref) != os.path.abspath(ref_stage):
        shutil.copy(bold_ref, ref_stage)
    fs_brain = nib.load(fs_t1_ref)
    brain_rs = resample_to_img(
        nib.Nifti1Image((fs_brain.get_fdata(dtype=np.float32) > 0).astype(np.float32),
                        fs_brain.affine),
        ref_img, interpolation='linear', force_resample=True, copy_header=True,
    )
    brain_data = brain_rs.get_fdata(dtype=np.float32) >= 0.5

    masks = {}
    for label, native_path in native.items():
        rs_name = 'acompcor_{}_boldgrid.nii.gz'.format(label)
        run_cmd(
            ['antsApplyTransforms', '-d', '3',
             '-i', _container_path(work_dir, os.path.basename(native_path), docker_image),
             '-r', _container_path(work_dir, 'bold_ref.nii.gz', docker_image),
             '-o', _container_path(work_dir, rs_name, docker_image),
             '-n', 'Gaussian', '-t', 'identity'],
            work_dir=work_dir,
            docker_image=docker_image,
        )
        rs_img = nib.load(opj(work_dir, rs_name))
        rs_data = rs_img.get_fdata(dtype=np.float32)
        rs_data[~brain_data] = 0
        bin_data = (rs_data >= bin_threshold).astype(np.uint8)

        out_path = opj(work_dir, 'acompcor_mask-{}.nii.gz'.format(label))
        nib.save(nib.Nifti1Image(bin_data, ref_img.affine, ref_img.header), out_path)
        masks[label] = out_path

        n_vox = int(bin_data.sum())
        print('    {} aCompCor mask: {} voxels'.format(label, n_vox))
        if n_vox == 0:
            print('    WARNING: {} aCompCor mask is empty at threshold {} '
                  '— ACompCor will fail for this mask.'.format(label, bin_threshold))

    brain_path = opj(work_dir, 'brain_mask.nii.gz')
    nib.save(nib.Nifti1Image(brain_data.astype(np.uint8), ref_img.affine, ref_img.header), brain_path)
    masks['brain'] = brain_path

    return masks


# ---------------------------------------------------------------------------
# CompCor
# ---------------------------------------------------------------------------

def compute_tcompcor(
    bold_file: str,
    brain_mask: str,
    tr: float,
    work_dir: str,
    skip_vols: int = 0,
    variance_threshold: float = 0.5,
    percentile_threshold: float = 0.02,
    high_pass_cutoff: float = 128.0,
):
    """
    Run nipype's TCompCor — the exact interface fMRIPrep wraps as
    RobustTCompCor (a retry-on-LinAlgError shim; numerically identical for
    well-behaved data) — with fMRIPrep's own parameters
    (init_bold_confs_wf: percentile_threshold=0.02, pre_filter='cosine',
    variance_threshold=0.5).
    """
    tcc = TCompCor(
        realigned_file=bold_file,
        mask_files=[brain_mask],
        percentile_threshold=percentile_threshold,
        pre_filter='cosine',
        high_pass_cutoff=high_pass_cutoff,
        save_pre_filter=True,
        save_metadata=True,
        variance_threshold=variance_threshold,
        repetition_time=tr,
        ignore_initial_volumes=skip_vols,
        header_prefix='t_comp_cor_',
        components_file='tcompcor.tsv',
        failure_mode='NaN',
    )
    return tcc.run(cwd=work_dir).outputs


def compute_acompcor(
    bold_file: str,
    csf_mask: str,
    wm_mask: str,
    combined_mask: str,
    tr: float,
    work_dir: str,
    skip_vols: int = 0,
    variance_threshold: float = 0.5,
    high_pass_cutoff: float = 128.0,
):
    """
    Run nipype's ACompCor jointly across the CSF, WM and combined masks,
    with fMRIPrep's own parameters (pre_filter='cosine',
    variance_threshold=0.5, merge_method='none'). Columns must be split
    and renamed afterwards with rename_acompcor_columns() — fMRIPrep's
    canonical a_comp_cor_XX columns come specifically from the "combined"
    mask, not directly from this raw output.
    """
    acc = ACompCor(
        realigned_file=bold_file,
        mask_files=[csf_mask, wm_mask, combined_mask],
        mask_names=['CSF', 'WM', 'combined'],
        merge_method='none',
        pre_filter='cosine',
        high_pass_cutoff=high_pass_cutoff,
        save_pre_filter=True,
        save_metadata=True,
        variance_threshold=variance_threshold,
        repetition_time=tr,
        ignore_initial_volumes=skip_vols,
        header_prefix='a_comp_cor_',
        components_file='acompcor.tsv',
        failure_mode='NaN',
    )
    return acc.run(cwd=work_dir).outputs


def rename_acompcor_columns(components_file: str, metadata_file: str) -> pd.DataFrame:
    """
    Split the jointly-computed aCompCor table into per-mask column groups
    and renumber each group from 0, replicating fMRIPrep's RenameACompCor:
        CSF mask      -> c_comp_cor_XX
        WM mask       -> w_comp_cor_XX
        combined mask -> a_comp_cor_XX   (fMRIPrep's canonical aCompCor)
    """
    components = pd.read_csv(components_file, sep='\t')
    metadata   = pd.read_csv(metadata_file, sep='\t')

    if 'retained' in metadata.columns:
        metadata = metadata[metadata['retained']]

    prefixes = {'CSF': 'c_comp_cor_', 'WM': 'w_comp_cor_', 'combined': 'a_comp_cor_'}
    out = pd.DataFrame(index=components.index)
    for mask_label, prefix in prefixes.items():
        cols = metadata.loc[metadata['mask'] == mask_label, 'component'].tolist()
        for i, col in enumerate(cols):
            if col in components.columns:
                out['{}{:02d}'.format(prefix, i)] = components[col]
    return out


def rename_cosine_columns(pre_filter_file: str) -> pd.DataFrame:
    """
    Load the DCT high-pass basis nipype writes alongside CompCor and rename
    its columns to fMRIPrep's cosineXX convention. With ignore_initial_volumes
    nipype appends one-hot non-steady-state columns to this file; those are
    dropped here (gather writes them as non_steady_state_outlierXX).
    """
    df = pd.read_csv(pre_filter_file, sep='\t')
    df = df[[c for c in df.columns if c.startswith('Cosine')]]
    df.columns = ['cosine{:02d}'.format(i) for i in range(len(df.columns))]
    return df


def mean_signal(bold_file: str, mask_file: str) -> np.ndarray:
    """
    Mean BOLD signal timeseries within a binary mask — fMRIPrep's
    SignalExtraction step for the csf / white_matter / global_signal
    confound columns.
    """
    bold = nib.load(bold_file).get_fdata(dtype=np.float32)
    mask = nib.load(mask_file).get_fdata() > 0
    if not mask.any():
        return np.full(bold.shape[-1], np.nan)
    return bold[mask].mean(axis=0)


def detect_nonsteady_states(bold_file: str, n_volumes: int = 40) -> int:
    """
    Number of initial non-steady-state volumes, as fMRIPrep's
    `niworkflows.interfaces.bold.NonsteadyStatesDetector` (defaults:
    nonnegative, n_volumes=40): clip the first *n_volumes* to
    [0, 99.8th percentile], then `is_outlier()` on the spatial mean. fMRIPrep
    passes this to CompCor as ignore_initial_volumes unless --dummy-scans is
    given.
    """
    from nipype.algorithms.confounds import is_outlier
    img = nib.load(bold_file)
    data = np.asarray(img.dataobj[..., :n_volumes], dtype=np.float32)
    data = np.clip(data, a_min=0.0, a_max=np.percentile(data, 99.8))
    return int(is_outlier(np.mean(data, axis=(0, 1, 2))))


# ---------------------------------------------------------------------------
# Per-run pipeline
# ---------------------------------------------------------------------------

def process_run(
    bold_file: str,
    masks: dict,
    subject: str,
    session: str,
    subject_output_dir: str,
    skip_vols: int,
    variance_threshold: float,
    percentile_threshold: float,
    high_pass_cutoff: float,
    overwrite: dict,
    skip: dict,
) -> dict:
    """
    Execute tCompCor, aCompCor and confound-gathering for a single
    coregistered BOLD run. Returns a dict of final output paths.
    """
    ow = {k: False for k in STEP_KEYS}
    ow.update(overwrite)

    sk = {k: False for k in STEP_KEYS}
    if skip:
        sk.update(skip)

    run_label, task_label = get_labels(bold_file)
    run_suffix = '_'.join(t for t in [task_label, run_label] if t)

    work_dir = opj(subject_output_dir, run_suffix)
    os.makedirs(work_dir, exist_ok=True)
    safe_work_dir = make_safe_workdir(work_dir)

    def _final(suffix, ext='.tsv'):
        return build_output_name(
            subject_output_dir, subject, session, suffix, extension=ext)

    tr = _get_tr(bold_file)

    if skip_vols is None:
        skip_vols = detect_nonsteady_states(bold_file)
        print('\n  Non-steady-state volumes (detected): {}'.format(skip_vols))
    else:
        print('\n  Non-steady-state volumes (given): {}'.format(skip_vols))

    # ------------------------------------------------------------------
    # tCompCor
    # ------------------------------------------------------------------
    print('\n  [tCompCor] top {:.0%}-variance voxels within brain mask...'.format(
        percentile_threshold))

    tcompcor_final = _final('{}_desc-tcompcor'.format(run_suffix))
    cosine_final   = _final('{}_desc-cosine'.format(run_suffix))

    if not check_skip(
        {'tcompcor': tcompcor_final, 'cosine': cosine_final},
        ow['tcompcor'],
        'tCompCor',
        force_skip=sk['tcompcor'],
    ):
        tcc_out = compute_tcompcor(
            bold_file=bold_file,
            brain_mask=masks['brain'],
            tr=tr,
            work_dir=safe_work_dir,
            skip_vols=skip_vols,
            variance_threshold=variance_threshold,
            percentile_threshold=percentile_threshold,
            high_pass_cutoff=high_pass_cutoff,
        )
        shutil.copy(tcc_out.components_file, tcompcor_final)
        rename_cosine_columns(tcc_out.pre_filter_file).to_csv(
            cosine_final, sep='\t', index=False, na_rep='n/a')

    print('  tCompCor     : {}'.format(tcompcor_final))
    print('  Cosine basis : {}'.format(cosine_final))

    # ------------------------------------------------------------------
    # aCompCor
    # ------------------------------------------------------------------
    print('\n  [aCompCor] CSF / WM / combined masks...')

    acompcor_final = _final('{}_desc-acompcor'.format(run_suffix))

    if not check_skip(
        {'acompcor': acompcor_final},
        ow['acompcor'],
        'aCompCor',
        force_skip=sk['acompcor'],
    ):
        acc_out = compute_acompcor(
            bold_file=bold_file,
            csf_mask=masks['csf'],
            wm_mask=masks['wm'],
            combined_mask=masks['combined'],
            tr=tr,
            work_dir=safe_work_dir,
            skip_vols=skip_vols,
            variance_threshold=variance_threshold,
            high_pass_cutoff=high_pass_cutoff,
        )
        acompcor_df = rename_acompcor_columns(acc_out.components_file, acc_out.metadata_file)
        acompcor_df.to_csv(acompcor_final, sep='\t', index=False, na_rep='n/a')

    print('  aCompCor     : {}'.format(acompcor_final))

    # ------------------------------------------------------------------
    # Gather -> single fMRIPrep-equivalent confounds file
    # ------------------------------------------------------------------
    print('\n  [Gather] merging into desc-confounds_timeseries.tsv...')

    confounds_final = _final('{}_desc-confounds_timeseries'.format(run_suffix))

    if not check_skip(
        {'confounds': confounds_final},
        ow['gather_confounds'],
        'Gather confounds',
        force_skip=sk['gather_confounds'],
    ):
        acompcor_df = pd.read_csv(acompcor_final, sep='\t', na_values='n/a')
        tcompcor_df = pd.read_csv(tcompcor_final, sep='\t', na_values='n/a')
        cosine_df   = pd.read_csv(cosine_final, sep='\t', na_values='n/a')

        merged = pd.concat(
            [acompcor_df.reset_index(drop=True),
             tcompcor_df.reset_index(drop=True),
             cosine_df.reset_index(drop=True)],
            axis=1,
        )
        merged['csf']           = mean_signal(bold_file, masks['csf'])
        merged['white_matter']  = mean_signal(bold_file, masks['wm'])
        merged['global_signal'] = mean_signal(bold_file, masks['brain'])
        for i in range(skip_vols):
            merged['non_steady_state_outlier{:02d}'.format(i)] = (merged.index == i).astype(float)

        merged.to_csv(confounds_final, sep='\t', index=False, na_rep='n/a')

    print('  Confounds    : {}'.format(confounds_final))

    shutil.rmtree(work_dir)
    return {
        'tcompcor':  tcompcor_final,
        'cosine':    cosine_final,
        'acompcor':  acompcor_final,
        'confounds': confounds_final,
    }


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    bids_dir: str,
    coreg_file: str,
    output_file: str,
    subject: str,
    session: str = 'ses-01',
    anat_file: str = 'fmriprep',
    skip_vols: int = None,
    variance_threshold: float = 0.5,
    percentile_threshold: float = 0.02,
    high_pass_cutoff: float = 128.0,
    gm_threshold: float = 0.05,
    gm_dilate_radius: int = 3,
    acompcor_bin_threshold: float = 0.99,
    freesurfer: bool = True,
    docker_image: str = None,
    overwrite: dict = None,
    skip: dict = None,
) -> dict:
    """
    Run the full aCompCor/tCompCor pipeline for every coregistered BOLD run
    of one subject/session.

    aCompCor masks are subject-level (built once, reused across sessions —
    every run shares the same BOLD grid via s02_coreg.py's applyxfm4D; this
    is checked for every run).

    Parameters
    ----------
    freesurfer   : use fMRIPrep's FreeSurfer (is_aseg) aCompCor mask branch.
    docker_image : image providing antsApplyTransforms (Docker tag, .sif, or
                   'local'); defaults to $FPREP_IMAGE.
    overwrite : dict mapping step key -> bool; True forces re-run even if
                outputs exist. Valid keys: STEP_KEYS.
    skip      : dict mapping step key -> bool; True hard-skips the step with
                no file checks. Valid keys: STEP_KEYS.

    Returns a dict mapping run keys -> per-run output dicts.
    """
    ow = {k: False for k in STEP_KEYS}
    if overwrite:
        unknown = set(overwrite) - set(STEP_KEYS)
        if unknown:
            raise ValueError(
                'Unknown overwrite key(s): {}.  Valid keys: {}'.format(
                    sorted(unknown), STEP_KEYS))
        ow.update(overwrite)

    sk = {k: False for k in STEP_KEYS}
    if skip:
        unknown = set(skip) - set(STEP_KEYS)
        if unknown:
            raise ValueError(
                'Unknown skip key(s): {}.  Valid keys: {}'.format(
                    sorted(unknown), STEP_KEYS))
        sk.update(skip)

    if docker_image is None:
        docker_image = os.environ.get('FPREP_IMAGE', 'local')

    coreg_dir  = str(Path(opj(bids_dir, 'derivatives', coreg_file)).resolve())
    output_dir = str(Path(opj(bids_dir, 'derivatives', output_file)).resolve())
    anat_dir   = str(Path(opj(bids_dir, 'derivatives', anat_file)).resolve())

    subject_coreg_dir   = opj(coreg_dir, subject)
    subject_output_dir  = opj(output_dir, subject, session)
    subject_main_dir    = opj(output_dir, subject)  # above session level, for masks

    os.makedirs(subject_output_dir, exist_ok=True)
    main_work_dir = opj(subject_main_dir, '_mask_work')
    os.makedirs(main_work_dir, exist_ok=True)
    safe_main_work = make_safe_workdir(main_work_dir)

    print('-' * 55)
    print('Processing: aCompCor / tCompCor from coregistered BOLD')
    print('-' * 55)
    print(' Coreg input : {}'.format(coreg_dir))
    print(' Anat input  : {}'.format(anat_dir))
    print(' Output      : {}'.format(output_dir))
    print(' Subject     : {}'.format(subject))
    print(' Session     : {}'.format(session))
    print(' Mask branch : {}'.format('freesurfer' if freesurfer else 'fast'))
    print(' ANTs image  : {}'.format(docker_image))
    print('-' * 55)

    # ------------------------------------------------------------------
    # Discover coregistered BOLD runs
    # ------------------------------------------------------------------
    bold_pattern = opj(
        coreg_dir, subject, session,
        '{}_{}*space-fsT1_desc-moco_bbreg_bold.nii*'.format(subject, session))
    bold_files = sorted(glob.glob(bold_pattern))

    if not bold_files:
        raise FileNotFoundError(
            'No coregistered BOLD files found for {}_{}. Searched: {}'.format(
                subject, session, bold_pattern))

    print('\nFound {} BOLD run(s).'.format(len(bold_files)))
    for b in bold_files:
        print('  - {}'.format(os.path.basename(b)))

    # ------------------------------------------------------------------
    # aCompCor masks (subject-level)
    # ------------------------------------------------------------------
    print('\n[Step] Building aCompCor masks (subject-level)...')

    fs_t1_ref = build_output_name(subject_coreg_dir, subject, None, 'desc-fsbrain')
    if not Path(fs_t1_ref).exists():
        raise FileNotFoundError(
            'FS-T1 reference not found: {}. Run s02_coreg.py first.'.format(fs_t1_ref))

    mask_labels = ['csf', 'wm', 'combined', 'brain']
    mask_finals = {
        label: build_output_name(
            subject_main_dir, subject, None,
            'desc-acompcor{}_mask'.format(label.capitalize()))
        for label in mask_labels
    }

    if not check_skip(
        mask_finals,
        ow['acompcor_masks'],
        'aCompCor masks',
        force_skip=sk['acompcor_masks'],
    ):
        gm_probseg  = _find_anat_file(anat_dir, subject, session, 'label-GM_probseg.nii.gz')
        wm_probseg  = _find_anat_file(anat_dir, subject, session, 'label-WM_probseg.nii.gz')
        csf_probseg = _find_anat_file(anat_dir, subject, session, 'label-CSF_probseg.nii.gz')

        print('  GM probseg  : {}'.format(gm_probseg))
        print('  WM probseg  : {}'.format(wm_probseg))
        print('  CSF probseg : {}'.format(csf_probseg))

        bold_ref = extract_volume(bold_files[0], opj(safe_main_work, 'bold_vol0.nii.gz'))
        print('  BOLD grid   : {} (volume 0)'.format(os.path.basename(bold_files[0])))

        masks_work = build_acompcor_masks(
            gm_probseg=gm_probseg,
            wm_probseg=wm_probseg,
            csf_probseg=csf_probseg,
            fs_t1_ref=fs_t1_ref,
            bold_ref=bold_ref,
            work_dir=safe_main_work,
            docker_image=docker_image,
            freesurfer=freesurfer,
            gm_threshold=gm_threshold,
            gm_dilate_radius=gm_dilate_radius,
            bin_threshold=acompcor_bin_threshold,
        )
        for label in mask_labels:
            shutil.copy(masks_work[label], mask_finals[label])

    masks = mask_finals
    print('  CSF mask      : {}'.format(masks['csf']))
    print('  WM mask       : {}'.format(masks['wm']))
    print('  Combined mask : {}'.format(masks['combined']))
    print('  Brain mask    : {}'.format(masks['brain']))

    # Masks are reused across sessions, so every run must be on their grid
    mask_img = nib.load(masks['brain'])
    for b in bold_files:
        b_img = nib.load(b)
        if (b_img.shape[:3] != mask_img.shape[:3]
                or not np.allclose(b_img.affine, mask_img.affine, atol=1e-3)):
            raise ValueError(
                'BOLD run {} is not on the aCompCor mask grid ({} vs {}). '
                'Rebuild the masks with --overwrite acompcor_masks.'.format(
                    os.path.basename(b), b_img.shape[:3], mask_img.shape[:3]))

    # ------------------------------------------------------------------
    # Per-run processing
    # ------------------------------------------------------------------
    all_results = {}

    for run_idx, bold_file in enumerate(bold_files, start=1):
        print('\n' + '=' * 55)
        print('Processing run {}/{}: {}'.format(
            run_idx, len(bold_files), os.path.basename(bold_file)))
        print('=' * 55)

        run_results = process_run(
            bold_file=bold_file,
            masks=masks,
            subject=subject,
            session=session,
            subject_output_dir=subject_output_dir,
            skip_vols=skip_vols,
            variance_threshold=variance_threshold,
            percentile_threshold=percentile_threshold,
            high_pass_cutoff=high_pass_cutoff,
            overwrite=ow,
            skip=sk,
        )

        run_label, _ = get_labels(bold_file)
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
        description='aCompCor / tCompCor computed directly from coreg + anat-only outputs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',    required=True,
                     help='BIDS directory')
    req.add_argument('--coreg-file',  required=True,
                     help='Derivatives subdirectory with s02_coreg.py outputs')
    req.add_argument('--output-file', required=True,
                     help='Output derivatives subdirectory '
                          '(e.g. s3_make_confounds)')
    req.add_argument('--sub',         required=True,
                     help='Subject label (e.g. sub-01 or 01)')

    p.add_argument('--ses',          default='ses-01',
                   help='Session label (e.g. ses-01)')
    p.add_argument('--anat-file',    default='fmriprep',
                   help='Derivatives subdirectory with s01_fmriprep_anat_only.sh '
                        'outputs (tissue probseg maps)')
    p.add_argument('--skip-vols',    default='auto',
                   type=lambda v: None if v == 'auto' else int(v),
                   help='Number of non-steady-state volumes to ignore (nipype '
                        'ignore_initial_volumes); "auto" detects them per run '
                        "as fMRIPrep does (fMRIPrep's --dummy-scans overrides)")
    p.add_argument('--variance-threshold', type=float, default=0.5,
                   help='Retain components explaining this fraction of '
                        'variance within each CompCor mask (fMRIPrep default: 0.5)')
    p.add_argument('--tcompcor-percentile', type=float, default=0.02,
                   help='tCompCor high-variance voxel percentile threshold '
                        '(fMRIPrep default: 0.02)')
    p.add_argument('--high-pass-cutoff', type=float, default=128.0,
                   help='DCT high-pass cutoff (s) applied before CompCor PCA '
                        '(fMRIPrep default: 128.0)')
    p.add_argument('--gm-threshold', type=float, default=0.05,
                   help='GM probseg threshold used to build the GM exclusion '
                        'mask for aCompCor (fMRIPrep default: 0.05)')
    p.add_argument('--gm-dilate-radius', type=int, default=3,
                   help='Voxel radius for dilating the GM exclusion mask '
                        '(fMRIPrep default: 3)')
    p.add_argument('--acompcor-bin-threshold', type=float, default=0.99,
                   help='Threshold for binarizing resampled aCompCor masks '
                        '(fMRIPrep default: 0.99)')
    p.add_argument('--freesurfer', action=argparse.BooleanOptionalAction, default=True,
                   help="Use fMRIPrep's FreeSurfer (is_aseg) aCompCor mask branch; "
                        '--no-freesurfer matches an fMRIPrep --fs-no-reconall run '
                        '(only GM probseg > --gm-threshold is excluded)')
    p.add_argument('--docker',       default=None,
                   help='Image with antsApplyTransforms: Docker tag, .sif file '
                        'in $SIF_DIR, or "local" (default: $FPREP_IMAGE, else local)')

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
            'Hard-skip one or more named steps — no file checks. '
            'Downstream steps continue regardless; caller is responsible '
            'for any missing dependencies.'
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
        coreg_file=args.coreg_file,
        output_file=args.output_file,
        subject=args.sub,
        session=args.ses,
        anat_file=args.anat_file,
        skip_vols=args.skip_vols,
        variance_threshold=args.variance_threshold,
        percentile_threshold=args.tcompcor_percentile,
        high_pass_cutoff=args.high_pass_cutoff,
        gm_threshold=args.gm_threshold,
        gm_dilate_radius=args.gm_dilate_radius,
        acompcor_bin_threshold=args.acompcor_bin_threshold,
        freesurfer=args.freesurfer,
        docker_image=args.docker,
        overwrite=overwrite,
        skip=skip,
    )


if __name__ == '__main__':
    main()
