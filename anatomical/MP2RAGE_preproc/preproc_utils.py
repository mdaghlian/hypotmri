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
    nighres_skull_strip   Step 1b – Nighres brain mask
    apply_mask            Step 1c – Apply binary brain mask
    nighres_mgdm          Step 2  – Nighres MGDM segmentation
    nighres_dura_estimation Step 3 – Nighres dura estimation
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
    Copy *path* to a backup alongside the original.

    The backup is named ``<stem>_backup.mgz``.  If that already exists a
    timestamp suffix is added to avoid clobbering it.

    Parameters
    ----------
    path : File to back up (must exist)

    Returns
    -------
    Path to the newly created backup file
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

    Returns
    -------
    nibabel.freesurfer.MGHImage in the space of ref_mgz
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

    * If **none** exist → always run (returns False).
    * If **some but not all** exist → raises RuntimeError (partial/corrupt
      state).
    * If **all** exist and *overwrite* is True → log and run (returns False).
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
                    Values may be ``str`` or ``Path``.
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
    existing = [k for k, p in outdir_paths.items() if Path(p).exists()]
    missing  = [k for k, p in outdir_paths.items() if not Path(p).exists()]
    
    if not existing:
        print('  [run] {}'.format(step_name))
        return False

    if missing and existing:
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
    stem = get_stem(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spm_out_dir = out_dir / '{}_spm_biascorrect'.format(stem)
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
    stem = get_stem(uni_path)
    out_dir = Path(out_dir)
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

def apply_mask(input_image: str, mask_image: str, out_dir: str,
               out_suffix: str = '_masked') -> str:
    """
    Apply a binary brain mask to a NIfTI image.

    Voxels where the mask is zero are set to zero in the output.  The mask
    is resampled to the input image grid if their shapes differ
    (nearest-neighbour).

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
    stem = get_stem(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img  = nib.load(str(input_path))
    mask = nib.load(str(Path(mask_image).resolve()))

    img_data  = img.get_fdata()
    mask_data = mask.get_fdata()

    if mask_data.shape != img_data.shape:
        from nilearn.image import resample_to_img
        mask      = resample_to_img(mask, img, interpolation='nearest')
        mask_data = mask.get_fdata()

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
    so MGDM's Mp2rage7T atlas priors match the expected intensity
    distribution.

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
    Estimate dura matter probability using
    nighres.brain.mp2rage_dura_estimation.

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