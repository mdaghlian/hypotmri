"""
preproc_utils.py
================
Pure-function utilities for MP2RAGE preprocessing and FreeSurfer reconstruction.
No nipype dependency — every step is a plain Python function that can be
called directly or imported into other scripts.

Public API
----------
General I/O helpers
    get_stem              Strip .nii.gz / .nii to a bare file stem
    stage_inputs          Copy files into a working directory
    check_result          Raise on non-zero subprocess exit
    run_cmd               Run a subprocess, stream output, raise on failure
    run_docker            Run a command inside a Docker container

File / image utilities
    backup_file           Copy a file to a timestamped backup
    resample_to_mgh       Resample any image to an MGHImage in a reference space

Pipeline flow control
    check_skip            Decide whether a pipeline step should be skipped;
                          optionally restore outputs from outdir → workdir

FreeSurfer helpers
    mri_dir               Return the mri/ subdirectory for a subject
    launch_freeview       Open freeview non-blocking (silently skips if absent)

MP2RAGE preprocessing steps
    spm_bias_correct      Step 0  – SPM bias-field correction
    mprage_ise            Step 1  – MPRAGEise (background suppression)
    cat12_seg             Step 1d – CAT12 segmentation
    spm_seg               Step 1d – SPM segmentation
    warp_atlas_sag_sinus  Step 1e – Warp atlas sagittal sinus mask → T1w space
    nighres_skull_strip   Step 1b – Nighres brain mask
    apply_mask            Step 1c – Apply binary brain mask
    nighres_mgdm          Step 2  – Nighres MGDM segmentation
    nighres_dura_estimation Step 3 – Nighres dura estimation
    combine_brain_masks   Step 4a – Combine nighres + CAT12 masks with
                                    dura/MGDM-guided surface erosion
    refine_sss_mask       Step 4b – Refine SSS: atlas × INV2 dark signal
                                    × dura probability
    make_brain_mask_nosss Step 4c – Subtract dilated SSS from combined brain
                                    mask → FreeSurfer-ready final mask
"""

import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# General I/O helpers
# ---------------------------------------------------------------------------

def get_stem(path: Path) -> str:
    """Strip both .nii.gz and .nii extensions to return the bare file stem."""
    return Path(path.stem).stem if path.suffix == '.gz' else path.stem


def stage_inputs(work_dir: str, *paths: str) -> None:
    """Copy files into work_dir if not already there."""
    for src in paths:
        dst = os.path.join(work_dir, os.path.basename(src))
        if os.path.realpath(src) != os.path.realpath(dst):
            shutil.copy(src, dst)


def check_result(result, tool_name: str) -> None:
    """Raise RuntimeError with full stdout/stderr if a subprocess failed."""
    if result.returncode != 0:
        raise RuntimeError(
            '{} failed (exit {}).\n'
            '--- stdout ---\n{}\n'
            '--- stderr ---\n{}'.format(
                tool_name, result.returncode, result.stdout, result.stderr)
        )


def run_cmd(cmd: list, tool_name: str, env: dict = None,
            timeout: int = None) -> None:
    """
    Run a subprocess, print its output line by line, raise on failure.

    stdout and stderr are merged into a single stream so output appears in
    the order it was produced.

    Parameters
    ----------
    cmd       : Command and arguments as a list of strings
    tool_name : Label used in log prefixes and error messages
    env       : Extra environment variables merged with os.environ
    timeout   : Maximum seconds to wait (None = no limit)
    """
    merged_env = {**os.environ, **(env or {})}
    print('[{}] Running: {}'.format(tool_name, ' '.join(str(c) for c in cmd)))
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
        timeout=timeout,
    )
    if result.stdout:
        for line in result.stdout.splitlines():
            print('[{}] {}'.format(tool_name, line))
    check_result(result, tool_name)


def run_docker(work_dir: str, docker_image: str, cmd: list,
               env_vars: dict = None, verbose: bool = True) -> None:
    """
    Run *cmd* inside *docker_image*, mounting *work_dir* as /data.

    Streams stdout/stderr in real time. Raises RuntimeError on non-zero exit.

    Parameters
    ----------
    work_dir     : Host directory mounted as /data inside the container
    docker_image : Docker image tag
    cmd          : Command to run inside the container
    env_vars     : Environment variables passed via -e flags
    verbose      : If True, stream container output to stdout
    """
    env_flags = []
    for k, v in (env_vars or {}).items():
        env_flags += ['-e', '{}={}'.format(k, v)]

    proc = subprocess.Popen(
        ['docker', 'run', '--rm',
         *env_flags,
         '-v', '{}:/data'.format(work_dir),
         docker_image] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_lines, stderr_lines = [], []

    def _reader(stream, store, label):
        for line in stream:
            store.append(line)
            if verbose:
                print('[docker {}] {}'.format(label, line), end='', flush=True)

    t_out = threading.Thread(target=_reader,
                             args=(proc.stdout, stdout_lines, 'stdout'))
    t_err = threading.Thread(target=_reader,
                             args=(proc.stderr, stderr_lines, 'stderr'))
    t_out.start(); t_err.start()
    t_out.join();  t_err.join()
    proc.wait()

    class _Result:
        returncode = proc.returncode
        stdout     = ''.join(stdout_lines)
        stderr     = ''.join(stderr_lines)

    check_result(_Result(), 'Docker container ({})'.format(docker_image))


# ---------------------------------------------------------------------------
# File / image utilities
# ---------------------------------------------------------------------------

def backup_file(path: Path) -> Path:
    """
    Copy *path* to a timestamped backup alongside the original.

    The backup is named ``<stem>_backup.mgz``.  If that already exists a
    timestamp suffix is added to avoid clobbering it.

    Returns the path to the newly created backup file.
    """
    base_name = path.stem.split('.')[0]
    backup = path.parent / '{}_backup.mgz'.format(base_name)
    if backup.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = path.parent / '{}_backup_{}.mgz'.format(base_name, timestamp)
    shutil.copyfile(str(path), str(backup))
    print('[backup] {} -> {}'.format(path.name, backup.name))
    return backup


def resample_to_mgh(src, ref_mgz: Path) -> 'nib.freesurfer.MGHImage':
    """
    Resample *src* to the voxel grid of *ref_mgz* using nearest-neighbour
    interpolation and return a FreeSurfer MGHImage.

    Parameters
    ----------
    src     : NIfTI path (str/Path), nibabel NIfTI image, or MGHImage
    ref_mgz : Reference MGZ whose header / affine define the target space
    """
    from nilearn import image as nli
    resampled = nli.resample_to_img(src, str(ref_mgz), interpolation='nearest')
    return nib.freesurfer.MGHImage(
        resampled.get_fdata().astype(np.float32),
        affine=resampled.affine,
    )


# ---------------------------------------------------------------------------
# Pipeline flow control
# ---------------------------------------------------------------------------

def check_skip(
    outdir_paths: dict,
    overwrite: bool,
    step_name: str,
    workdir_paths: dict = None,
) -> bool:
    """
    Decide whether a pipeline step should be skipped.

    Checks whether every output in *outdir_paths* already exists.

    * If **none** exist          → always run (returns False).
    * If **some but not all**    → raises RuntimeError (partial/corrupt state).
    * If **all** exist and *overwrite* is True  → log and run (returns False).
    * If **all** exist and *overwrite* is False → log, optionally restore
      files to *workdir_paths*, and return True (skip).

    The optional *workdir_paths* argument supports the MP2RAGE preprocessing
    pipeline, where intermediate outputs live in a separate working directory
    and must be copied back so that downstream steps can find them.  The
    FreeSurfer scripts write outputs directly into the subject directory and
    do not need this copy-back behaviour — simply omit *workdir_paths*.

    Parameters
    ----------
    outdir_paths  : ``{label: path}`` mapping of expected final outputs.
    overwrite     : If True, never skip regardless of existing outputs.
    step_name     : Human-readable label used in log messages.
    workdir_paths : ``{label: path}`` mapping of corresponding working-
                    directory paths (same keys as *outdir_paths*).  When
                    provided, existing outputs are copied here on skip so
                    downstream steps can read from the working directory as
                    normal.  Optional — pass ``None`` to disable copy-back.

    Returns
    -------
    True  if the step should be skipped.
    False if the step should run.

    Raises
    ------
    RuntimeError
        If some but not all expected outputs exist (partial/corrupt state).
    """
    existing = [k for k, p in outdir_paths.items() if     Path(p).exists()]
    missing  = [k for k, p in outdir_paths.items() if not Path(p).exists()]

    if not existing:
        print('  [run] {}'.format(step_name))
        return False

    if missing:
        raise RuntimeError(
            '{}: partial outputs found — some exist, some are missing.\n'
            '  Present : {}\n'
            '  Missing : {}\n'
            'Delete the partial outputs or re-run with overwrite=True.'.format(
                step_name,
                [str(outdir_paths[k]) for k in existing],
                [str(outdir_paths[k]) for k in missing],
            )
        )

    # All outputs exist
    if overwrite:
        print('  [overwrite] {} — existing output(s) will be replaced.'.format(
            step_name))
        return False

    print('  [skip] {} — output(s) already exist.{}'.format(
        step_name,
        ' Restoring to workdir.' if workdir_paths else '',
    ))
    if workdir_paths:
        for label, src in outdir_paths.items():
            dst = workdir_paths[label]
            if Path(src).resolve() != Path(dst).resolve():
                shutil.copy(str(src), str(dst))
    return True


# ---------------------------------------------------------------------------
# FreeSurfer helpers
# ---------------------------------------------------------------------------

def mri_dir(subjects_dir: str, subject: str) -> Path:
    """Return the mri/ subdirectory for a FreeSurfer subject."""
    return Path(subjects_dir) / subject / 'mri'


def launch_freeview(*paths: str) -> None:
    """
    Open freeview non-blocking with the supplied path arguments.

    Silently skips if freeview is not on PATH.  Paths are passed verbatim,
    so the caller can include freeview overlay syntax
    (e.g. ``'image.mgz:colormap=heat:opacity=0.4'``).
    """
    if shutil.which('freeview'):
        try:
            subprocess.Popen(['freeview'] + list(paths))
            print('[QC] freeview launched in background.')
        except Exception as exc:
            print('[QC] Could not launch freeview: {}'.format(exc))
    else:
        print('[QC] freeview not found on PATH — open files manually.')


# ---------------------------------------------------------------------------
# Step 0 – SPM bias-field correction
# ---------------------------------------------------------------------------

def spm_bias_correct(
    input_image: str,
    out_dir: str,
    mp2rage_script_dir: str,
    spm_script: str = 's01_spmbc',
    spm_standalone: str = None,
    mcr_path: str = None,
) -> str:
    """
    Run SPM bias-field correction on *input_image*.

    Parameters
    ----------
    input_image        : Path to input NIfTI (.nii or .nii.gz)
    out_dir            : Directory where outputs will be written
    mp2rage_script_dir : Directory containing the SPM m-script
    spm_script         : SPM m-script name (default: s01_spmbc)
    spm_standalone     : Path to SPM standalone executable (optional)
    mcr_path           : Path to MATLAB MCR directory (required if
                         spm_standalone is set)

    Returns
    -------
    Path to the bias-corrected image (.nii.gz)
    """
    input_path = Path(input_image).resolve()
    stem       = get_stem(input_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spm_out_dir = out_dir / '{}_spmbc'.format(stem)
    spm_out_dir.mkdir(parents=True, exist_ok=True)

    staged_input = out_dir / input_path.name
    if staged_input.resolve() != input_path.resolve():
        shutil.copy(str(input_path), str(staged_input))

    matlab_expr = "{script}('{input}', '{outdir}');".format(
        script=spm_script,
        input=str(staged_input),
        outdir=str(spm_out_dir),
    )

    if spm_standalone and mcr_path:
        cmd = [spm_standalone, mcr_path, 'script', matlab_expr]
    else:
        cmd = ['matlab', '-nodisplay', '-nosplash', '-nodesktop',
               '-batch', matlab_expr]

    result = subprocess.run(
        cmd,
        shell=False,
        cwd=mp2rage_script_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3600,
    )
    check_result(result, 'SPM bias correction')

    spm_biascorrected = spm_out_dir / '{}_biascorrected.nii'.format(stem)
    if not spm_biascorrected.exists():
        raise FileNotFoundError(
            'SPM bias correction completed but expected output not found:\n'
            '  {}'.format(spm_biascorrected)
        )

    out_path = out_dir / '{}_spmbc.nii.gz'.format(stem)
    nib.save(nib.load(str(spm_biascorrected)), str(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 1 – MPRAGEise
# ---------------------------------------------------------------------------

def mprage_ise(uni_file: str, inv2_file: str, out_dir: str) -> str:
    """
    Suppress MP2RAGE background noise by multiplying UNI by normalised INV2.

    The INV2 image is normalised to its 99th percentile (over positive voxels)
    before multiplication, driving background noise toward zero while
    preserving grey/white contrast.

    Parameters
    ----------
    uni_file  : Path to UNI image (.nii or .nii.gz)
    inv2_file : Path to bias-corrected INV2 image (.nii or .nii.gz)
    out_dir   : Directory where the output will be written

    Returns
    -------
    Path to the MPRAGEised image (.nii.gz)
    """
    uni_path = Path(uni_file).resolve()
    stem     = get_stem(uni_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uni_img  = nib.load(str(uni_path))
    inv2_img = nib.load(str(Path(inv2_file).resolve()))
    uni_data  = uni_img.get_fdata()
    inv2_data = inv2_img.get_fdata()

    positive_voxels = inv2_data[inv2_data > 0]
    if positive_voxels.size == 0:
        raise ValueError(
            'INV2 image contains no positive voxels — '
            'check that the correct image was supplied: {}'.format(inv2_file)
        )

    norm_factor = np.percentile(positive_voxels, 99)
    if norm_factor == 0:
        raise ValueError(
            '99th-percentile of INV2 positive voxels is zero — '
            'normalisation would produce NaN/Inf values.'
        )

    out_path = out_dir / '{}_mpragised.nii.gz'.format(stem)
    nib.save(
        nib.Nifti1Image(
            (inv2_data / norm_factor) * uni_data,
            uni_img.affine,
            uni_img.header,
        ),
        str(out_path),
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 1d – Segmentation (shared private helper + two public wrappers)
# ---------------------------------------------------------------------------

def _run_spm_seg_script(
    input_image: str,
    seg_out_dir: str,
    mp2rage_script_dir: str,
    spm_script: str,
    tool_name: str,
    spm_standalone: str = None,
    mcr_path: str = None,
) -> str:
    """
    Run a single SPM/CAT12 segmentation m-script and return the output dir.

    Private helper shared by cat12_seg() and spm_seg().  The caller is
    responsible for constructing *seg_out_dir* with the appropriate naming
    convention (e.g. <stem>_cat12seg or <stem>_spmseg).

    Parameters
    ----------
    input_image        : Path to input NIfTI (.nii or .nii.gz)
    seg_out_dir        : Dedicated output subdirectory for this segmentation
    mp2rage_script_dir : Directory containing the m-script; used as cwd
    spm_script         : MATLAB function name to call
    tool_name          : Human-readable label for log / error messages
    spm_standalone     : Path to SPM standalone executable (optional)
    mcr_path           : Path to MATLAB MCR directory (required if
                         spm_standalone is set)

    Returns
    -------
    Path to the segmentation output directory (str)
    """
    input_path  = Path(input_image).resolve()
    seg_out_dir = Path(seg_out_dir)
    seg_out_dir.mkdir(parents=True, exist_ok=True)

    matlab_expr = "{script}('{input}', '{outdir}');".format(
        script=spm_script,
        input=str(input_path),
        outdir=str(seg_out_dir),
    )

    if spm_standalone and mcr_path:
        cmd = [spm_standalone, mcr_path, 'script', matlab_expr]
    else:
        cmd = ['matlab', '-nodisplay', '-nosplash', '-nodesktop',
               '-batch', matlab_expr]

    result = subprocess.run(
        cmd,
        shell=False,
        cwd=mp2rage_script_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=7200,
    )
    check_result(result, tool_name)

    if not any(seg_out_dir.iterdir()):
        raise FileNotFoundError(
            '{} completed but output directory is empty:\n'
            '  {}'.format(tool_name, seg_out_dir)
        )

    return str(seg_out_dir)

def cat12_seg(
    input_image: str,
    out_dir: str,
    mp2rage_script_dir: str,
    spm_script: str = 'preproc_cat12seg',
    spm_standalone: str = None,
    mcr_path: str = None,
) -> str:
    """
    Run CAT12 segmentation on *input_image* via preproc_cat12seg.m.

    The input should be the MPRAGEised UNI **with skull** — CAT12 performs
    its own skull stripping internally and the skull is needed for accurate
    tissue segmentation.

    Parameters
    ----------
    input_image        : Path to input NIfTI (.nii or .nii.gz)
    out_dir            : Directory where the output folder will be written
    mp2rage_script_dir : Directory containing preproc_cat12seg.m
    spm_script         : MATLAB function name (default: preproc_cat12seg)
    spm_standalone     : Path to SPM standalone executable (optional)
    mcr_path           : Path to MATLAB MCR directory (required if
                         spm_standalone is set)

    Returns
    -------
    Path to the CAT12 output directory (<out_dir>/<stem>_cat12seg)
    """
    stem        = get_stem(Path(input_image))
    seg_out_dir = Path(out_dir) / '{}_cat12seg'.format(stem)
    return _run_spm_seg_script(
        input_image=input_image,
        seg_out_dir=str(seg_out_dir),
        mp2rage_script_dir=mp2rage_script_dir,
        spm_script=spm_script,
        tool_name='CAT12 segmentation',
        spm_standalone=spm_standalone,
        mcr_path=mcr_path,
    )


def spm_seg(
    input_image: str,
    out_dir: str,
    mp2rage_script_dir: str,
    spm_script: str = 'preproc_spmseg',
    spm_standalone: str = None,
    mcr_path: str = None,
) -> str:
    """
    Run SPM segmentation on *input_image* via preproc_spmseg.m.

    The input should be the MPRAGEised UNI **with skull**.  SPM segmentation
    is run in parallel with CAT12 (Step 1d) to allow direct comparison of
    tissue probability maps and brain masks.

    Parameters
    ----------
    input_image        : Path to input NIfTI (.nii or .nii.gz)
    out_dir            : Directory where the output folder will be written
    mp2rage_script_dir : Directory containing preproc_spmseg.m
    spm_script         : MATLAB function name (default: preproc_spmseg)
    spm_standalone     : Path to SPM standalone executable (optional)
    mcr_path           : Path to MATLAB MCR directory (required if
                         spm_standalone is set)

    Returns
    -------
    Path to the SPM output directory (<out_dir>/<stem>_spmseg)
    """
    stem        = get_stem(Path(input_image))
    seg_out_dir = Path(out_dir) / '{}_spmseg'.format(stem)
    return _run_spm_seg_script(
        input_image=input_image,
        seg_out_dir=str(seg_out_dir),
        mp2rage_script_dir=mp2rage_script_dir,
        spm_script=spm_script,
        tool_name='SPM segmentation',
        spm_standalone=spm_standalone,
        mcr_path=mcr_path,
    )


# ---------------------------------------------------------------------------
# Step 1e – Warp atlas sagittal sinus mask → T1w space
# ---------------------------------------------------------------------------

def warp_atlas_sag_sinus(
    t1w_image: str,
    atlas_mask: str,
    out_dir: str,
    fsl_dir: str = None,
    out_name: str = 'SSS-atlas-in-T1.nii.gz',
) -> str:
    """
    Register T1w image to MNI (FLIRT affine), invert the transform, and
    warp the dilated atlas sagittal sinus mask into T1w subject space.

    Three FSL calls are made:
      1. flirt       T1w → MNI affine
      2. convert_xfm invert → MNI → T1w
      3. flirt       warp atlas mask → T1w space (nearest-neighbour)

    Parameters
    ----------
    t1w_image  : T1w reference — should be the MPRAGEised UNI with skull
    atlas_mask : Dilated atlas SSS mask in MNI space
    out_dir    : Directory where transforms and warped mask are written
    fsl_dir    : FSL installation root (default: $FSLDIR or /usr/local/fsl)
    out_name   : Filename for the warped mask

    Returns
    -------
    Path to the warped atlas SSS mask in T1w space (.nii.gz)
    """
    fsl_dir  = fsl_dir or os.environ.get('FSLDIR', '/usr/local/fsl')
    mni_brain = Path(fsl_dir) / 'data/standard/MNI152_T1_1mm_brain.nii.gz'
    if not mni_brain.exists():
        raise FileNotFoundError(
            'MNI brain template not found: {}\n'
            'Set $FSLDIR or pass fsl_dir explicitly.'.format(mni_brain)
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_to_mni   = out_dir / 'T1_to_MNI_affine.mat'
    mni_to_t1   = out_dir / 'MNI_to_T1_affine.mat'
    atlas_in_t1 = out_dir / out_name

    run_cmd(
        ['flirt',
         '-in',   str(t1w_image),
         '-ref',  str(mni_brain),
         '-omat', str(t1_to_mni),
         '-out',  str(out_dir / 'T1_in_MNI.nii.gz')],
        'FLIRT T1→MNI',
    )
    run_cmd(
        ['convert_xfm',
         '-omat',    str(mni_to_t1),
         '-inverse', str(t1_to_mni)],
        'convert_xfm invert',
    )
    run_cmd(
        ['flirt',
         '-in',       str(atlas_mask),
         '-ref',      str(t1w_image),
         '-applyxfm',
         '-init',     str(mni_to_t1),
         '-interp',   'nearestneighbour',
         '-out',      str(atlas_in_t1)],
        'FLIRT atlas→T1',
    )

    if not atlas_in_t1.exists():
        raise FileNotFoundError(
            'Atlas warp completed but output not found: {}'.format(atlas_in_t1)
        )
    return str(atlas_in_t1)


# ---------------------------------------------------------------------------
# Step 1b – Nighres skull stripping (brain mask only)
# ---------------------------------------------------------------------------

def nighres_skull_strip(
    inv2_image: str,
    uni_image: str,
    out_dir: str,
    t1map_image: str = None,
    docker_image: str = 'nighres/nighres:latest',
) -> str:
    """
    Derive a binary brain mask using nighres.brain.mp2rage_skullstripping.

    Only the brain_mask output is returned — nighres-masked images are
    discarded.  All downstream masking is done explicitly via apply_mask().

    Parameters
    ----------
    inv2_image   : Bias-corrected INV2 (second_inversion input to nighres)
    uni_image    : Raw UNI (t1_weighted input — required by nighres)
    out_dir      : Working/output directory (mounted as /data inside Docker)
    t1map_image  : T1 map (optional but recommended for 7T)
    docker_image : Nighres Docker image tag

    Returns
    -------
    Path to the binary brain mask (.nii.gz)
    """
    out_dir   = Path(out_dir)
    inv2_path = Path(inv2_image).resolve()
    uni_path  = Path(uni_image).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_inputs(str(out_dir), str(inv2_path), str(uni_path))
    stem = get_stem(inv2_path)

    t1map_kwarg = ''
    if t1map_image:
        t1map_path = Path(t1map_image).resolve()
        stage_inputs(str(out_dir), str(t1map_path))
        t1map_kwarg = '    t1_map="/data/{}", '.format(t1map_path.name)

    python_script = (
        'import nighres, json; '
        'r = nighres.brain.mp2rage_skullstripping('
        '    second_inversion="/data/{inv2}", '
        '    t1_weighted="/data/{uni}", '
        + t1map_kwarg +
        '    save_data=True, '
        '    output_dir="/data", '
        '    file_name="{stem}"); '
        'paths = {{k: str(v) for k, v in r.items()}}; '
        'open("/data/skullstrip_outputs.json", "w").write(json.dumps(paths)); '
    ).format(inv2=inv2_path.name, uni=uni_path.name, stem=stem)

    run_docker(
        work_dir=str(out_dir),
        docker_image=docker_image,
        cmd=['python3', '-c', python_script],
    )

    json_path = out_dir / 'skullstrip_outputs.json'
    if not json_path.exists():
        raise FileNotFoundError(
            'Skull stripping completed but output JSON not found: {}'.format(
                json_path)
        )

    with open(json_path) as f:
        out_paths = json.load(f)

    brain_mask = Path(out_paths['brain_mask'].replace('/data', str(out_dir)))
    if not brain_mask.exists():
        raise FileNotFoundError(
            'Expected brain mask not found: {}'.format(brain_mask))

    return str(brain_mask)


# ---------------------------------------------------------------------------
# Step 1c – Apply brain mask
# ---------------------------------------------------------------------------

def apply_mask(
    input_image: str,
    mask_image: str,
    out_dir: str,
    out_suffix: str = '_masked',
) -> str:
    """
    Apply a binary brain mask to a NIfTI image.

    Voxels where the mask is zero are set to zero.  The mask is resampled to
    the input grid if their shapes differ (nearest-neighbour).

    Parameters
    ----------
    input_image : Image to mask (.nii or .nii.gz)
    mask_image  : Binary brain mask (.nii or .nii.gz)
    out_dir     : Directory where the output will be written
    out_suffix  : Suffix appended to the stem (default: '_masked')

    Returns
    -------
    Path to the masked image (.nii.gz)
    """
    input_path = Path(input_image).resolve()
    stem       = get_stem(input_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img  = nib.load(str(input_path))
    mask = nib.load(str(Path(mask_image).resolve()))

    img_data  = img.get_fdata()
    mask_data = mask.get_fdata()

    if mask_data.shape != img_data.shape:
        from nilearn.image import resample_to_img
        mask_data = resample_to_img(
            mask, img, interpolation='nearest'
        ).get_fdata()

    out_path = out_dir / '{}{}.nii.gz'.format(stem, out_suffix)
    nib.save(
        nib.Nifti1Image(
            img_data * (mask_data > 0).astype(img_data.dtype),
            img.affine,
            img.header,
        ),
        str(out_path),
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 2 – Nighres MGDM segmentation
# ---------------------------------------------------------------------------

def nighres_mgdm(
    input_image: str,
    out_dir: str,
    docker_image: str = 'nighres/nighres:latest',
    contrast_type: str = 'Mp2rage7T',
    t1map_image: str = None,
    atlas: str = None,
) -> dict:
    """
    Run nighres MGDM brain segmentation inside a Docker container.

    Expects skull-stripped inputs.  Pass the raw (non-MPRAGEised) UNI image
    so MGDM's Mp2rage7T atlas priors match the expected intensity distribution.

    Parameters
    ----------
    input_image   : Skull-stripped UNI image (.nii or .nii.gz)
    out_dir       : Working/output directory (mounted as /data inside Docker)
    docker_image  : Nighres Docker image tag
    contrast_type : MGDM contrast type (default: Mp2rage7T)
    t1map_image   : Skull-stripped T1 map (optional)
    atlas         : MGDM atlas file (optional; uses nighres default if unset)

    Returns
    -------
    dict with keys: segmentation, memberships, labels, distance
    """
    out_dir    = Path(out_dir)
    input_path = Path(input_image).resolve()
    stem       = get_stem(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_inputs(str(out_dir), str(input_path))

    contrast2_kwargs = ''
    if t1map_image:
        t1map_path = Path(t1map_image).resolve()
        stage_inputs(str(out_dir), str(t1map_path))
        contrast2_kwargs = (
            '    contrast_image2="/data/{t1map}", '
            '    contrast_type2="T1map7T", '
        ).format(t1map=t1map_path.name)

    atlas_kwarg = ''
    if atlas:
        atlas_kwarg = '    atlas_file="{}", '.format(atlas)

    python_script = (
        'import nighres, json; '
        'r = nighres.brain.mgdm_segmentation('
        '    contrast_image1="/data/{input}", '
        '    contrast_type1="{contrast}", '
        + contrast2_kwargs
        + atlas_kwarg +
        '    save_data=True, '
        '    output_dir="/data", '
        '    file_name="{stem}"); '
        'paths = {{k: str(v) for k, v in r.items()}}; '
        'open("/data/mgdm_outputs.json", "w").write(json.dumps(paths)); '
    ).format(input=input_path.name, contrast=contrast_type, stem=stem)

    run_docker(
        work_dir=str(out_dir),
        docker_image=docker_image,
        cmd=['python3', '-c', python_script],
    )

    json_path = out_dir / 'mgdm_outputs.json'
    if not json_path.exists():
        raise FileNotFoundError(
            'MGDM completed but output JSON not found: {}'.format(json_path)
        )

    with open(json_path) as f:
        out_paths = json.load(f)

    def _remap(p):
        return Path(str(p).replace('/data', str(out_dir)))

    outputs = {
        'segmentation': _remap(out_paths['segmentation']),
        'memberships':  _remap(out_paths['memberships']),
        'labels':       _remap(out_paths['labels']),
        'distance':     _remap(out_paths['distance']),
    }
    for key, path in outputs.items():
        if not path.exists():
            raise FileNotFoundError(
                'MGDM completed but expected {} output not found: {}'.format(
                    key, path)
            )

    return {k: str(v) for k, v in outputs.items()}


# ---------------------------------------------------------------------------
# Step 3 – Nighres dura estimation
# ---------------------------------------------------------------------------

def nighres_dura_estimation(
    inv2_image: str,
    brain_mask: str,
    out_dir: str,
    docker_image: str = 'nighres/nighres:latest',
    background_distance: float = 5.0,
) -> str:
    """
    Estimate dura matter probability using nighres.brain.mp2rage_dura_estimation.

    Parameters
    ----------
    inv2_image          : Bias-corrected INV2 image (second_inversion input)
    brain_mask          : Brain mask from skull stripping (skullstrip_mask)
    out_dir             : Working/output directory (mounted as /data in Docker)
    docker_image        : Nighres Docker image tag
    background_distance : Maximum distance within mask for dura (default: 5.0)

    Returns
    -------
    Path to the dura probability image (.nii.gz)
    """
    out_dir   = Path(out_dir)
    inv2_path = Path(inv2_image).resolve()
    mask_path = Path(brain_mask).resolve()
    stem      = get_stem(inv2_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_inputs(str(out_dir), str(inv2_path), str(mask_path))

    python_script = (
        'import nighres, json; '
        'r = nighres.brain.mp2rage_dura_estimation('
        '    second_inversion="/data/{inv2}", '
        '    skullstrip_mask="/data/{mask}", '
        '    background_distance={bg_dist}, '
        '    save_data=True, '
        '    output_dir="/data", '
        '    file_name="{stem}"); '
        'paths = {{k: str(v) for k, v in r.items()}}; '
        'open("/data/dura_outputs.json", "w").write(json.dumps(paths)); '
    ).format(
        inv2=inv2_path.name,
        mask=mask_path.name,
        bg_dist=background_distance,
        stem=stem,
    )

    run_docker(
        work_dir=str(out_dir),
        docker_image=docker_image,
        cmd=['python3', '-c', python_script],
    )

    json_path = out_dir / 'dura_outputs.json'
    if not json_path.exists():
        raise FileNotFoundError(
            'Dura estimation completed but output JSON not found: {}'.format(
                json_path)
        )

    with open(json_path) as f:
        out_paths = json.load(f)

    dura_proba = Path(out_paths['result'].replace('/data', str(out_dir)))
    if not dura_proba.exists():
        raise FileNotFoundError(
            'Dura estimation completed but expected output not found: '
            '{}'.format(dura_proba)
        )
    return str(dura_proba)


# ---------------------------------------------------------------------------
# Internal geometry helpers (shared by Steps 4a–4c)
# ---------------------------------------------------------------------------

def _vox_size_mm(img: nib.Nifti1Image) -> float:
    """Return the mean isotropic voxel size in mm from image zooms."""
    zooms = img.header.get_zooms()[:3]
    return float(np.mean(np.abs(zooms)))


def _sphere_se(radius_vox: int) -> np.ndarray:
    """Return a boolean spherical structuring element of *radius_vox*."""
    coords = np.mgrid[
        -radius_vox : radius_vox + 1,
        -radius_vox : radius_vox + 1,
        -radius_vox : radius_vox + 1,
    ]
    return (coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2) <= radius_vox ** 2


def _binary_dilate_mm(mask: np.ndarray, radius_mm: float,
                      vox_mm: float) -> np.ndarray:
    """Dilate a binary mask by *radius_mm* mm using a spherical SE."""
    import math
    from scipy.ndimage import binary_dilation
    radius_vox = int(math.ceil(radius_mm / vox_mm))
    return binary_dilation(mask.astype(bool), structure=_sphere_se(radius_vox))


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask retaining only the largest connected component."""
    from scipy.ndimage import label
    labelled, _ = label(mask)
    if labelled.max() == 0:
        return mask.astype(bool)
    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    return (labelled == counts.argmax()).astype(bool)


# ---------------------------------------------------------------------------
# Step 4a – Combine brain masks with dura/MGDM-guided surface erosion
# ---------------------------------------------------------------------------

def combine_brain_masks(
    nighres_mask: str,
    cat12_mask: str,
    dura_proba: str,
    mgdm_memberships: str,
    out_dir: str,
    dura_threshold: float = 0.7,
    mgdm_bg_threshold: float = 0.7,
    closing_radius_mm: float = 2.0,
    cortical_shell_mm: float = 4.0,
) -> str:
    """
    Combine nighres and CAT12 brain masks, then erode surface non-brain tissue
    guided by dura probability and MGDM membership maps.

    Strategy
    --------
    1. Union of nighres + CAT12 masks — catches what either method missed.
    2. Cortical shell: outermost *cortical_shell_mm* of the union.  Erosion
       is only applied within this shell so deep WM is never touched.
    3. Within the shell, remove voxels where:
         a. dura probability > *dura_threshold*
         b. MGDM max-membership < (1 − *mgdm_bg_threshold*)
       Either criterion is sufficient to remove a voxel.
    4. Morphological closing to bridge any sulcal gaps reopened by erosion.
    5. Fill internal holes (slice-wise axial, then 3-D).
    6. Keep only the largest connected component.

    Parameters
    ----------
    nighres_mask      : Binary brain mask from nighres skull stripping
    cat12_mask        : Brain mask from CAT12 (*_brainmask.nii)
    dura_proba        : Nighres dura probability image
    mgdm_memberships  : Nighres MGDM memberships volume (4-D x,y,z,labels).
                        Maximum across label channels is used.
    out_dir           : Directory where the output mask is written
    dura_threshold    : Dura probability above which a shell voxel is removed
                        (default: 0.7 — conservative)
    mgdm_bg_threshold : Shell voxels where max-MGDM-membership < (1 − this)
                        are treated as background (default: 0.7)
    closing_radius_mm : Morphological closing radius after erosion (default: 2.0)
    cortical_shell_mm : Shell thickness within which erosion is applied
                        (default: 4.0 mm)

    Returns
    -------
    Path to combined brain mask (.nii.gz)
    """
    import math
    from scipy.ndimage import binary_erosion, binary_closing, binary_fill_holes
    from nilearn.image import resample_to_img as rti

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_img = nib.load(str(nighres_mask))
    vox_mm  = _vox_size_mm(ref_img)

    def _load_binary(p):
        img = nib.load(str(p))
        if img.shape[:3] != ref_img.shape:
            img = rti(img, ref_img, interpolation='nearest')
        return img.get_fdata(dtype=np.float32) > 0.5

    def _load_prob(p):
        img = nib.load(str(p))
        if img.shape[:3] != ref_img.shape:
            img = rti(img, ref_img, interpolation='continuous')
        return img.get_fdata(dtype=np.float32)

    nighres_data = _load_binary(nighres_mask)
    cat12_data   = _load_binary(cat12_mask)
    union        = nighres_data | cat12_data

    # Cortical shell
    shell_vox    = int(math.ceil(cortical_shell_mm / vox_mm))
    eroded_union = binary_erosion(union, structure=_sphere_se(shell_vox))
    shell        = union & ~eroded_union

    # Dura criterion
    dura_data   = _load_prob(dura_proba)
    dura_remove = shell & (dura_data > dura_threshold)

    # MGDM criterion
    mgdm_raw = _load_prob(mgdm_memberships)
    mgdm_max = mgdm_raw.max(axis=-1) if mgdm_raw.ndim == 4 else mgdm_raw
    mgdm_remove = shell & (mgdm_max < (1.0 - mgdm_bg_threshold))

    combined = union.copy()
    combined[dura_remove | mgdm_remove] = False

    print('  Dura criterion removed  : {:,} shell voxels'.format(int(dura_remove.sum())))
    print('  MGDM criterion removed  : {:,} shell voxels'.format(int(mgdm_remove.sum())))
    print('  Removed by both         : {:,} shell voxels'.format(int((dura_remove & mgdm_remove).sum())))

    # Closing, hole fill, LCC
    close_vox = int(math.ceil(closing_radius_mm / vox_mm))
    combined  = binary_closing(combined, structure=_sphere_se(close_vox))
    for z in range(combined.shape[2]):
        combined[:, :, z] = binary_fill_holes(combined[:, :, z])
    combined = binary_fill_holes(combined)
    combined = _largest_connected_component(combined)

    n_vox   = int(combined.sum())
    vol_cm3 = n_vox * (vox_mm ** 3) / 1000.0
    print('  Combined brain mask     : {:,} voxels  ({:.1f} cm³)'.format(n_vox, vol_cm3))
    if vol_cm3 < 900:
        print('  WARN: volume suspiciously small — check inputs')
    elif vol_cm3 > 1800:
        print('  WARN: volume suspiciously large — check for dura/skull inclusion')
    else:
        print('  OK: volume in expected range')

    out_path = out_dir / 'brain-mask-combined.nii.gz'
    nib.save(
        nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine, ref_img.header),
        str(out_path),
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 4b – Refine SSS mask: atlas × INV2 dark signal × dura probability
# ---------------------------------------------------------------------------

def refine_sss_mask(
    atlas_sss_in_t1: str,
    inv2_image: str,
    dura_proba: str,
    brain_mask: str,
    out_dir: str,
    inv2_percentile: float = 15.0,
    dura_threshold: float = 0.3,
) -> str:
    """
    Refine the atlas-prior SSS mask using INV2 dark signal and dura probability.

    A voxel is kept only if it satisfies all three:
      - Inside the warped atlas SSS prior  (spatial constraint)
      - INV2 ≤ p{inv2_percentile} within the atlas prior  (dark-signal gate)
      - Dura probability > {dura_threshold}  (structural gate)

    The dura threshold is intentionally low (0.3) because the atlas prior and
    INV2 criterion already provide strong constraints.

    Parameters
    ----------
    atlas_sss_in_t1 : Warped atlas SSS mask in T1w space (binary .nii.gz)
    inv2_image      : Bias-corrected INV2 — used as spatial reference grid
    dura_proba      : Nighres dura probability image
    brain_mask      : Brain mask used to restrict the INV2 percentile
                      computation to intra-cranial voxels
    out_dir         : Directory where the refined mask is written
    inv2_percentile : Percentile of INV2 intensity within the atlas prior
                      below which a voxel is 'dark' (default: 15.0)
    dura_threshold  : Minimum dura probability for inclusion (default: 0.3)

    Returns
    -------
    Path to the refined SSS mask (.nii.gz)
    """
    from nilearn.image import resample_to_img as rti

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_img   = nib.load(str(inv2_image))
    inv2_data = ref_img.get_fdata(dtype=np.float32)
    vox_mm    = _vox_size_mm(ref_img)

    def _load_resamp(p, interp='nearest'):
        img = nib.load(str(p))
        if img.shape[:3] != ref_img.shape:
            img = rti(img, ref_img, interpolation=interp)
        return img.get_fdata(dtype=np.float32)

    atlas_data = _load_resamp(atlas_sss_in_t1)         > 0.5
    dura_data  = _load_resamp(dura_proba, 'continuous')
    brain_data = _load_resamp(brain_mask)               > 0.5

    roi_vals = inv2_data[atlas_data & brain_data & (inv2_data > 0)]
    if roi_vals.size == 0:
        raise ValueError(
            'No positive INV2 voxels found within the atlas SSS prior.\n'
            'Check atlas warp quality (Step 1e output).'
        )

    inv2_thresh = np.percentile(roi_vals, inv2_percentile)
    print('  SSS INV2 threshold : {:.1f}  '
          '(≤p{:.0f} within atlas prior)'.format(inv2_thresh, inv2_percentile))

    inv2_dark = (inv2_data > 0) & (inv2_data <= inv2_thresh)
    dura_pos  = dura_data > dura_threshold
    refined   = (atlas_data & inv2_dark & dura_pos).astype(np.uint8)

    n_vox   = int(refined.sum())
    vol_mm3 = n_vox * (vox_mm ** 3)
    print('  Refined SSS mask   : {:,} voxels  '
          '({:.0f} mm³ / {:.2f} cm³)'.format(n_vox, vol_mm3, vol_mm3 / 1000))
    if vol_mm3 < 500:
        print('  WARN: SSS mask very small — check INV2 threshold / atlas warp')
    elif vol_mm3 > 12_000:
        print('  WARN: SSS mask very large — may include non-SSS tissue')
    else:
        print('  OK: SSS volume in plausible range (~2000–6000 mm³ expected)')

    out_path = out_dir / 'SSS-mask-refined.nii.gz'
    nib.save(
        nib.Nifti1Image(refined, ref_img.affine, ref_img.header),
        str(out_path),
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 4c – Subtract dilated SSS from combined brain mask
# ---------------------------------------------------------------------------

def make_brain_mask_nosss(
    brain_mask: str,
    sss_mask: str,
    out_dir: str,
    sss_dilation_mm: float = 3.5,
) -> str:
    """
    Subtract a dilated SSS mask from the combined brain mask.

    The SSS is dilated by *sss_dilation_mm* before subtraction to create a
    safety margin preventing the FreeSurfer pial surface from reaching into
    the sinus.  A 1-voxel morphological closing is applied after subtraction
    to smooth the boundary, followed by hole-filling and an LCC pass.

    Parameters
    ----------
    brain_mask      : Combined brain mask from Step 4a (.nii.gz)
    sss_mask        : Refined SSS mask from Step 4b (.nii.gz) — may have
                      been manually edited at the QC checkpoint
    out_dir         : Directory where the output mask is written
    sss_dilation_mm : SSS dilation radius in mm (default: 3.5)

    Returns
    -------
    Path to the final brain mask with SSS cavity excluded (.nii.gz)
    """
    import math
    from scipy.ndimage import binary_closing, binary_fill_holes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_img    = nib.load(str(brain_mask))
    brain_data = ref_img.get_fdata(dtype=np.float32) > 0.5
    vox_mm     = _vox_size_mm(ref_img)

    sss_img  = nib.load(str(sss_mask))
    sss_data = sss_img.get_fdata(dtype=np.float32)
    if sss_data.shape != brain_data.shape:
        from nilearn.image import resample_to_img
        sss_data = resample_to_img(
            sss_img, ref_img, interpolation='nearest'
        ).get_fdata(dtype=np.float32)

    sss_bin     = sss_data > 0.5
    sss_dilated = _binary_dilate_mm(sss_bin, sss_dilation_mm, vox_mm)
    print('  SSS dilation       : {:.1f} mm  '
          '({:,} → {:,} voxels)'.format(
              sss_dilation_mm, int(sss_bin.sum()), int(sss_dilated.sum())))

    result = brain_data & ~sss_dilated

    r      = max(1, int(math.ceil(1.0 / vox_mm)))
    result = binary_closing(result, structure=_sphere_se(r))
    result = binary_fill_holes(result)
    result = _largest_connected_component(result)

    removed_vox = int(brain_data.sum()) - int(result.sum())
    print('  Voxels removed (SSS hole) : {:,}  '
          '({:.1f} cm³)'.format(removed_vox, removed_vox * vox_mm ** 3 / 1000))

    out_path = out_dir / 'brain-mask-final.nii.gz'
    nib.save(
        nib.Nifti1Image(result.astype(np.uint8), ref_img.affine, ref_img.header),
        str(out_path),
    )
    return str(out_path)