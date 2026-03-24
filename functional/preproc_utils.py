import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
import hashlib
import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bold_base(bold_file: str, subject: str, session: str) -> str:
    """
    Return the BIDS entity string for a BOLD file with subject/session
    prefix removed, suitable for use as the *suffix* argument to
    build_output_name.

    E.g. given bold_file ending in
        sub-hp01_ses-01_task-pRFLE_run-03_sdc-bold.nii.gz
    returns
        task-pRFLE_run-03_sdc-bold
    """
    name = _strip_extensions(Path(bold_file)).name
    prefix = '_'.join(t for t in [subject, session] if t) + '_'
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name


def _strip_extensions(p: Path) -> Path:
    """Strip .nii and/or .gz suffixes from a Path."""
    while p.suffix in ('.gz', '.nii'):
        p = p.with_suffix('')
    return p

def _stage(src: str, work_dir: str) -> str:
    """
    Copy *src* into *work_dir* if it is not already there.
    Returns the host path of the copy (work_dir / basename(src)).
    """
    dst = os.path.join(work_dir, os.path.basename(src))
    if str(Path(src).resolve()) != str(Path(dst).resolve()):
        if Path(src).is_dir():
            if Path(dst).exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
    return dst

def _gunzip_to(src: str, dst: str) -> None:
    """Decompress *src* (.nii.gz) to *dst* (.nii)."""
    with open(dst, 'wb') as fh:
        subprocess.run(['gunzip', '-c', src], stdout=fh, check=True)


def _container_path(work_dir: str, filename: str, docker: str) -> str:
    """
    Return the path to *filename* as seen from inside the execution context:
    /data/<filename> for Docker, work_dir/<filename> for local.
    Handles empty filename (returns mount root).
    """
    if docker and docker != 'local':
        return '/data/{}'.format(filename).rstrip('/') if filename else '/data'
    return os.path.join(work_dir, filename) if filename else work_dir


def make_safe_workdir(work_dir: str) -> str:
    """
    If work_dir contains spaces, create a symlink under /tmp pointing to it
    and return the symlink path. Otherwise return work_dir unchanged.
    The symlink persists for the duration of the pipeline run.
    """
    if ' ' not in work_dir:
        return work_dir
    safe = os.path.join(
        '/tmp',
        'preproc_' + hashlib.md5(work_dir.encode()).hexdigest()[:12]
    )
    if not os.path.islink(safe):
        os.symlink(work_dir, safe)
    return safe

def get_labels(filepath: str) -> tuple:
    """
    Extract (run_label, task_label) from a BIDS filename.

    Returns ('run-XX', 'task-XX') or ('', 'task-unknown') if absent.
    """
    name = os.path.basename(filepath)
    run_match  = re.search(r'run-(\d+)', name)
    task_match = re.search(r'task-([a-zA-Z0-9]+)', name)
    run_label  = 'run-{}'.format(run_match.group(1))  if run_match  else ''
    task_label = 'task-{}'.format(task_match.group(1)) if task_match else 'task-unknown'
    return run_label, task_label


def fsl_val(nifti: str, field: str) -> str:
    """Return a single fslval field as a stripped string."""
    result = subprocess.run(
        ['fslval', nifti, field],
        capture_output=True, text=True,
    )
    check_result(result, 'fslval {} {}'.format(nifti, field))
    return result.stdout.strip()

def read_bold_meta(json_path: str):
    """
    Return (PhaseEncodingDirection, TotalReadoutTime) from a BIDS sidecar.
    """
    with open(json_path) as fh:
        meta = json.load(fh)
    return meta['PhaseEncodingDirection'], float(meta['TotalReadoutTime'])

def build_output_name(
    out_dir: str,
    subject: str,
    session: str,
    suffix: str,
    extension: str = '.nii.gz',
) -> str:
    """
    Build a BIDS-style output filename.

    Examples
    --------
    >>> build_output_name('/out', 'sub-01', 'ses-01', 'task-rest_run-1_sdc-bold')
    '/out/sub-01_ses-01_task-rest_run-1_sdc-bold.nii.gz'
    >>> build_output_name('/out', 'sub-01', None, 'task-rest_sdc-bold')
    '/out/sub-01_task-rest_sdc-bold.nii.gz'
    """
    tokens = [t for t in [subject, session, suffix] if t]
    return os.path.join(out_dir, '_'.join(tokens) + extension)


def check_result(result, label: str) -> None:
    """Raise RuntimeError if *result* has a non-zero returncode."""
    if result.returncode != 0:
        raise RuntimeError(
            '{} failed (exit {}).\nstderr:\n{}'.format(
                label, result.returncode, result.stderr)
        )


def run_docker(
    work_dir: str,
    docker_image: str,
    cmd: list,
    env_vars: dict = None,
    verbose: bool = True,
    cwd=None,
) -> None:
    """
    Run *cmd* inside *docker_image*, mounting *work_dir* as /data.

    If *docker_image* is ``"local"``, the command is run directly on the host
    instead.  In that case every argument starting with ``/data`` is rewritten
    to use *work_dir* as the root, so callers need not change anything.

    Streams stdout/stderr in real time.  Raises RuntimeError on non-zero exit.

    Parameters
    ----------
    work_dir     : Host directory mounted as /data inside the container
    docker_image : Docker image tag, or ``"local"`` to run on the host
    cmd          : Command to run inside the container (or locally)
    env_vars     : Environment variables passed via -e flags (Docker) or
                   injected into the subprocess env (local)
    verbose      : If True, stream output to stdout
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
        cwd=cwd,
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

def run_local(cmd: list, verbose: bool = True, env=None, cwd=None) -> None:
    """
    Run *cmd* as a local subprocess.

    Streams stdout/stderr in real time.  Raises RuntimeError on non-zero exit.
    Used for non-AFNI tools (fslnvols, gunzip) that run on the host.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
    )

    stdout_lines, stderr_lines = [], []

    def _reader(stream, store, label):
        for line in stream:
            store.append(line)
            if verbose:
                print('[local {}] {}'.format(label, line), end='', flush=True)
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

    check_result(_Result(), ' '.join(cmd))

def run_cmd(
    cmd: list,
    work_dir: str = None,
    docker_image: str = None,
    env_vars: dict = None,
    verbose: bool = True,
    cwd=None,
) -> None:
    if docker_image is None or docker_image == 'local':
        if work_dir is not None:
            cmd = [
                arg.replace('/data/', work_dir.rstrip('/') + '/', 1)
                if arg.startswith('/data/') else arg
                for arg in cmd
            ]
        env = {**os.environ, **(env_vars or {})} if env_vars else None
        run_local(cmd, verbose=verbose, env=env, cwd=cwd)
    else:
        if work_dir is None:
            raise ValueError('work_dir is required when running via Docker')
        run_docker(work_dir, docker_image, cmd, env_vars=env_vars, verbose=verbose)


def check_skip(
    outdir_paths: dict,
    overwrite: bool,
    step_label: str,
    workdir_paths: dict = None,
) -> bool:
    """
    Return True (skip this step) when all *outdir_paths* exist and
    *overwrite* is False.  Also copies existing outputs back to
    *workdir_paths* so downstream steps can use them.

    Return False (run this step) otherwise.
    """
    if overwrite:
        return False

    all_exist = all(Path(p).exists() for p in outdir_paths.values())
    if not all_exist:
        return False

    print('  [skip] {} — outputs already exist.'.format(step_label))
    if workdir_paths:
        for key, src in outdir_paths.items():
            dst = workdir_paths.get(key)
            if dst and src != dst:
                os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
                if os.path.isdir(src):
                    # if directory copy whole thing
                    shutil.copytree(src,dst, dirs_exist_ok=True)
                else:
                    shutil.copy(src, dst)
    return True


def get_nvols(nifti_path: str) -> int:
    """Return number of volumes in a NIfTI file via fslnvols."""
    result = subprocess.run(
        ['fslnvols', nifti_path],
        capture_output=True, text=True,
    )
    check_result(result, 'fslnvols {}'.format(nifti_path))
    return int(result.stdout.strip())


def read_pe_direction(json_path: str) -> str:
    """Extract PhaseEncodingDirection from a BIDS sidecar JSON."""
    with open(json_path) as fh:
        meta = json.load(fh)
    return meta['PhaseEncodingDirection']


def _to_afni_prefix(work_dir: str, name: str) -> str:
    """
    Return the AFNI +orig prefix path for *name* inside *work_dir*
    (i.e. /data/<name> as seen from inside the container).
    """
    return '/data/{}'.format(name)

