#!/usr/bin/env python
#$ -V
#$ -cwd
"""
s01_sdc_AFNI.py
===============
Run AFNI-based susceptibility distortion correction (SDC) for all BOLD runs
belonging to a given subject / session / task, using AFNI tools either
locally or inside a Docker container.

Pipeline per run
----------------
    Step 1  - Convert BOLD to AFNI format          (3dcopy)
    Step 2  - Convert reverse-PE EPI to AFNI       (3dcopy)
    Step 3  - Run unWarpEPIfloat.py                (cwd=work_dir, -s TS, -w /data)
    Step 4  - Apply warp to SBREF                  (3dNwarpApply)

Overwrite behaviour
-------------------
Each step checks for its output file in the run work directory.
If it exists and overwrite is False, the step is skipped.

    python s01_sdc_AFNI.py ... --overwrite convert_bold convert_reverse
    python s01_sdc_AFNI.py ... --overwrite-all

Valid step names:
    convert_bold      Step 1
    convert_reverse   Step 2
    unwarp            Step 3
    apply_warp_sbref  Step 4
    sdc_clean         Step 5

Usage example
-------------
python s01_sdc_AFNI.py \\
    --bids-dir    /data/bids \\
    --output-dir  /data/derivatives/sdc \\
    --sub         sub-01 \\
    --ses         ses-01 \\
    --task        rest \\
    --afni-docker afni/afni_make_build:latest
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

from cvl_utils.preproc_func import (
    build_output_name,
    make_safe_workdir,
    run_cmd,
    read_pe_direction,
    check_skip,
    get_nvols,
    _gunzip_to, 
    _container_path,
    _stage
)

STEP_KEYS = [
    'convert_bold',
    'convert_reverse',
    'unwarp',
    'apply_warp_sbref',
    'sdc_clean',
]

# unWarpEPIfloat.py -s argument: controls output dir name and dataset prefixes.
# Keep as 'TS' to match the reference bash script behaviour.
_UNWARP_SID = 'TS'


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def convert_to_afni(
    nifti_path: str,
    out_prefix: str,
    work_dir: str,
    afni_docker: str,
) -> str:
    """
    Convert *nifti_path* to AFNI +orig format in *work_dir*.
    Returns the host path to <out_prefix>+orig.HEAD.
    """
    # --- Handle input ---
    if nifti_path.endswith('.gz'):
        # if .nii.gz unzip as _temp.nii & copy to folder
        tmp_nii = os.path.join(work_dir, out_prefix + '_temp.nii')
        _gunzip_to(nifti_path, tmp_nii)
        src = _container_path(work_dir, out_prefix + '_temp.nii', afni_docker)
    else:
        # Otherwise just copy to folder
        tmp_nii = os.path.join(work_dir, out_prefix + '.nii')
        if not Path(tmp_nii).exists():
            shutil.copy(nifti_path, tmp_nii)
        src = _container_path(work_dir, tmp_nii, afni_docker)

    # --- Remove existing AFNI dataset if present ---
    head = Path(work_dir) / f"{out_prefix}+orig.HEAD"
    brik = Path(work_dir) / f"{out_prefix}+orig.BRIK"

    for f in (head, brik):
        if f.exists():
            f.unlink()

    # --- Run conversion ---
    dst = _container_path(work_dir, out_prefix, afni_docker)
    run_cmd(
        work_dir=work_dir,
        docker_image=afni_docker,
        cmd=['3dcopy', src, dst],
    )
    # --- Cleanup temp ---
    if nifti_path.endswith('.gz'):
        tmp_nii = os.path.join(work_dir, out_prefix + '_temp.nii')
        if Path(tmp_nii).exists():
            os.remove(tmp_nii)

    return str(head)

def run_unwarp(
    work_dir: str,
    idx_epi: str,
    idx_rev: str,
    afni_docker: str,
) -> str:
    """
    Run unWarpEPIfloat.py with:
      - cwd / -w set to work_dir (so bare relative paths resolve correctly)
      - -s TS (fixed subject ID, matching reference bash script)

    Returns host path to the corrected 06_*_HWV.nii.gz.
    """
    unwarp_script_src = os.path.join(
        os.environ['PIPELINE_DIR'], 'functional', 'unWarpEPIfloat.py')
    unwarp_script_dst = os.path.join(work_dir, 'unWarpEPIfloat.py')
    if str(Path(unwarp_script_src).resolve()) != str(Path(unwarp_script_dst).resolve()):
        shutil.copy(unwarp_script_src, unwarp_script_dst)

    workdir_arg = '/data' if (afni_docker and afni_docker != 'local') else work_dir

    run_cmd(
        work_dir=work_dir,
        docker_image=afni_docker,
        cmd=[
            'python',
            _container_path(work_dir, 'unWarpEPIfloat.py', afni_docker),
            '-f', 'bold+orig{}'.format(idx_epi),
            '-r', 'reverse+orig{}'.format(idx_rev),
            '-d', 'bold',
            '-s', _UNWARP_SID,
            '-w', workdir_arg,
        ],
        cwd=work_dir,
    )

    matches = glob.glob(os.path.join(
        work_dir, 'unWarpOutput_{}'.format(_UNWARP_SID), '06_*_HWV.nii.gz'))
    if not matches:
        raise FileNotFoundError(
            'AFNI unwarp output 06_*_HWV.nii.gz not found in '
            '{}/unWarpOutput_{}'.format(work_dir, _UNWARP_SID)
        )
    return matches[0]


def apply_warp_to_sbref(
    sbref_path: str,
    warp_file: str,          # Warp from run_unwarp
    master_path: str,        # used as reference to ensure correct geometry
    work_dir: str,
    afni_docker: str,
) -> str:
    convert_to_afni(sbref_path, 'sbref', work_dir, afni_docker)

    warp_dst = os.path.join(work_dir, os.path.basename(warp_file))
    if str(Path(warp_file).resolve()) != str(Path(warp_dst).resolve()):
        shutil.copy(warp_file, warp_dst)

    sdc_sbref = os.path.join(work_dir, 'sdc_sbref.nii.gz')
    if os.path.exists(sdc_sbref):
        os.unlink(sdc_sbref)
    warp_file_stage = _stage(warp_file, work_dir)

    run_cmd(
        work_dir=work_dir,
        docker_image=afni_docker,
        cmd=[
            '3dNwarpApply',
            '-source', _container_path(work_dir, 'sbref+orig', afni_docker),
            '-nwarp',  _container_path(work_dir, warp_file_stage, afni_docker),
            '-master', _container_path(work_dir, master_path, afni_docker),
            '-interp', 'wsinc5',
            '-prefix', _container_path(work_dir, 'sdc_sbref.nii.gz', afni_docker),
        ]
    )
    
    # Match unWarpEPIfloat.py: restore obliquity from the *forward calibration*
    # reference space, not from the original distorted sbref
    run_cmd(
        work_dir=work_dir,
        docker_image=afni_docker,
        cmd=[
            '3drefit',
            '-atrcopy', _container_path(work_dir, master_path, afni_docker),
            'IJK_TO_DICOM_REAL',
            _container_path(work_dir, 'sdc_sbref.nii.gz', afni_docker),
        ]
    )
    
    return sdc_sbref


def _find_forward_warp(work_dir: str) -> str:
    """
    Locate the forward warp produced by unWarpEPIfloat.py.

    Checks the copy persisted directly in *work_dir* first (kept around by
    sdc_clean), falling back to unWarpOutput_TS/ if cleanup hasn't run yet.
    Returns '' if no forward warp can be found.
    """
    matches = glob.glob(os.path.join(work_dir, '*_Forward_WARP.nii.gz'))
    if not matches:
        matches = glob.glob(os.path.join(
            work_dir, 'unWarpOutput_{}'.format(_UNWARP_SID), '*_Forward_WARP.nii.gz'))
    return matches[0] if matches else ''


def sdc_clean(work_dir: str, keep_paths: list) -> list:
    """
    Remove everything directly inside *work_dir* except the paths in
    *keep_paths*.  Returns the list of removed paths.
    """
    keep = {str(Path(p).resolve()) for p in keep_paths if p}
    removed = []
    for entry in sorted(Path(work_dir).iterdir()):
        if str(entry.resolve()) in keep:
            continue
        removed.append(str(entry))
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return removed


# ---------------------------------------------------------------------------
# Per-run pipeline
# ---------------------------------------------------------------------------

def process_run(
    bold_path: str,
    topup_path: str,
    sbref_path: str,
    subject: str,
    session: str,
    task: str,
    run_label: str,
    subject_output_dir: str,
    afni_docker: str,
    overwrite: dict,
) -> dict:
    """
    Execute all SDC steps for a single BOLD run.
    Returns a dict of final output paths.
    """
    ow = {k: False for k in STEP_KEYS}
    if overwrite:
        ow.update(overwrite)

    run_suffix_tokens = [t for t in [('task-' + task) if task else None,
                                     run_label] if t]
    run_suffix = '_'.join(run_suffix_tokens) if run_suffix_tokens else 'run'

    work_dir = os.path.join(subject_output_dir, run_suffix)
    os.makedirs(work_dir, exist_ok=True)

    # All AFNI calls use the space-free symlinked path; Python-side file
    # operations (glob, shutil, existence checks) use the real work_dir.
    safe_work_dir = make_safe_workdir(work_dir)

    def _out(suffix, ext='.nii.gz'):
        return build_output_name(work_dir, subject, session, suffix, extension=ext)

    # Working copies (live in work_dir while processing this run) and final
    # copies (promoted to subject_output_dir, alongside the other runs —
    # this is what's left once sdc_clean removes the per-run duplicates).
    unwarp_out  = _out('{}_sdc_bold'.format(run_suffix))
    sdc_sbref   = _out('{}_sdc_sbref'.format(run_suffix))
    final_bold  = os.path.join(subject_output_dir, os.path.basename(unwarp_out))
    final_sbref = os.path.join(subject_output_dir, os.path.basename(sdc_sbref))

    # If the final outputs already exist (in work_dir or already promoted),
    # the AFNI +orig conversions (Steps 1-2) may have been removed by
    # sdc_clean — skip regenerating them unless unwarp is being forced
    # (which needs bold+orig / reverse+orig to exist).
    skip_conversions = (
        not ow['unwarp']
        and (Path(unwarp_out).exists() or Path(final_bold).exists())
        and (Path(sdc_sbref).exists() or Path(final_sbref).exists())
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    n_vols_bold  = get_nvols(bold_path)
    n_vols_topup = get_nvols(topup_path)

    bold_json  = re.sub(r'\.nii(\.gz)?$', '.json', bold_path)
    topup_json = re.sub(r'\.nii(\.gz)?$', '.json', topup_path)
    # bold_pe    = read_pe_direction(bold_json)
    # topup_pe   = read_pe_direction(topup_json)

    print('  Volume information:')
    print('    BOLD volumes      : {}'.format(n_vols_bold))
    print('    Reverse-PE volumes: {}'.format(n_vols_topup))
    print('  Phase encoding:')
    # print('    BOLD PE           : {}'.format(bold_pe))
    # print('    Reverse-PE        : {}'.format(topup_pe))

    start_idx = n_vols_bold - n_vols_topup
    idx_epi   = '[{}..{}]'.format(start_idx, n_vols_bold - 1)
    idx_rev   = '[0..{}]'.format(n_vols_topup - 1)

    print('  BOLD subset for unwarp : {}'.format(idx_epi))
    print('  Reverse-PE subset      : {}'.format(idx_rev))

    #TODO: 1. maybe before the first step, check if the final outputs already exist and skip everything if so? 
    #TODO: 2. delete intermitent steps outputs at the end of the pipeline, or at least have an option to do so? 
     
    # ------------------------------------------------------------------
    # Step 1 — Convert BOLD to AFNI format
    # ------------------------------------------------------------------
    print('\n  [Step 1] Converting BOLD to AFNI format...')

    bold_afni = os.path.join(work_dir, 'bold+orig.HEAD')

    if not check_skip({'bold_afni': bold_afni}, ow['convert_bold'],
                      'Step 1: convert BOLD to AFNI',
                      force_skip=skip_conversions):
        convert_to_afni(bold_path, 'bold', safe_work_dir, afni_docker)

    print('    -> {}'.format(bold_afni))
    # ------------------------------------------------------------------
    # Step 2 — Convert reverse-PE to AFNI format
    # ------------------------------------------------------------------
    print('\n  [Step 2] Converting reverse-PE to AFNI format...')

    reverse_afni = os.path.join(work_dir, 'reverse+orig.HEAD')

    if not check_skip({'reverse_afni': reverse_afni}, ow['convert_reverse'],
                      'Step 2: convert reverse-PE to AFNI',
                      force_skip=skip_conversions):
        convert_to_afni(topup_path, 'reverse', safe_work_dir, afni_docker)

    print('    -> {}'.format(reverse_afni))

    # ------------------------------------------------------------------
    # Step 3 — Run unWarpEPIfloat.py
    # ------------------------------------------------------------------
    print('\n  [Step 3] Running unWarpEPIfloat.py...')

    # unwarp_out (work_dir) or final_bold (already promoted + cleaned) —
    # whichever exists indicates this step has already been done.
    unwarp_target = unwarp_out if Path(unwarp_out).exists() else final_bold

    if not check_skip({'unwarp_out': unwarp_target}, ow['unwarp'],
                      'Step 3: unWarpEPIfloat'):
        # Remove stale unWarpOutput_TS — script refuses to run if it exists
        stale = os.path.join(safe_work_dir, 'unWarpOutput_{}'.format(_UNWARP_SID))
        if Path(stale).exists():
            shutil.rmtree(stale)

        unwarp_nii = run_unwarp(
            work_dir=safe_work_dir,
            idx_epi=idx_epi,
            idx_rev=idx_rev,
            afni_docker=afni_docker,
        )
        shutil.copy(unwarp_nii, unwarp_out)
    else:
        # Resolved lazily in Step 4, only if it's actually needed there.
        unwarp_nii = None

    print('    -> {}'.format(unwarp_out))

    # ------------------------------------------------------------------
    # Step 4 — Apply warp to SBREF
    # ------------------------------------------------------------------
    print('\n  [Step 4] Applying warp to SBREF...')

    # sdc_sbref (work_dir) or final_sbref (already promoted + cleaned) —
    # whichever exists indicates this step has already been done.
    sbref_target = sdc_sbref if Path(sdc_sbref).exists() else final_sbref

    # - find the forward warp produced by Step 3
    fwd_warp_match = _find_forward_warp(work_dir)
    if not check_skip({'sdc_sbref': sbref_target}, ow['apply_warp_sbref'],
                    'Step 4: apply warp to SBREF'):
        if unwarp_nii is None:
            # unwarp_out is itself a copy of 06_*_HWV.nii.gz, so it (or the
            # promoted final_bold, restaged) can stand in as the reference
            # even after unWarpOutput_TS has been cleaned up.
            unwarp_nii = unwarp_out if Path(unwarp_out).exists() \
                else _stage(final_bold, work_dir)
        if fwd_warp_match:
            result = apply_warp_to_sbref(
                sbref_path=sbref_path,
                warp_file=fwd_warp_match,
                master_path=unwarp_nii, # - use the output as the reference
                work_dir=safe_work_dir,
                afni_docker=afni_docker,
            )
            shutil.copy(result, sdc_sbref)
        else:
            print('    [warn] Warp file not found — copying original SBREF as placeholder.')
            shutil.copy(sbref_path, sdc_sbref)
    print('    -> {}'.format(sdc_sbref))

    # ------------------------------------------------------------------
    # Promote fresh outputs to the subject-level output directory
    # ------------------------------------------------------------------
    for src, dst in ((unwarp_out, final_bold), (sdc_sbref, final_sbref)):
        if Path(src).exists():
            shutil.copy(src, dst)
            print('  Copied {} -> {}'.format(os.path.basename(src), subject_output_dir))

    # ------------------------------------------------------------------
    # Step 5 — Remove intermediate files (including the per-run copies of
    # sdc_bold / sdc_sbref, which now live in subject_output_dir)
    # ------------------------------------------------------------------
    print('\n  [Step 5] Cleaning up intermediate files...')

    # Persist the forward warp at the top of work_dir (used to apply this
    # run's SDC to other images) before everything else is removed.
    fwd_warp_match = _find_forward_warp(work_dir)
    fwd_warp_keep = None
    if fwd_warp_match:
        fwd_warp_keep = os.path.join(work_dir, os.path.basename(fwd_warp_match))
        if str(Path(fwd_warp_match).resolve()) != str(Path(fwd_warp_keep).resolve()):
            shutil.copy(fwd_warp_match, fwd_warp_keep)

    keep_paths = [fwd_warp_keep]
    already_clean = sorted(p.name for p in Path(work_dir).iterdir()) == \
        sorted(os.path.basename(p) for p in keep_paths if p)

    if not ow['sdc_clean'] and already_clean:
        print('  [skip] Step 5: sdc_clean — already clean.')
    else:
        removed = sdc_clean(work_dir, keep_paths)
        for p in removed:
            print('    removed {}'.format(os.path.basename(p)))
        if keep_paths:
            print('    -> kept: {}'.format(
                ', '.join(os.path.basename(p) for p in keep_paths if p)))

    return {
        'sdc_bold':  final_bold,
        'sdc_sbref': final_sbref,
    }


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    bids_dir: str,
    output_file: str,
    subject: str,
    session: str,
    task: str = '',
    afni_docker: str = os.environ.get('AFNI_IMAGE', 'afni/afni_make_build:latest'),
    overwrite: dict = None,
) -> dict:
    """
    Discover all BOLD runs for *subject* / *session* / *task* and run the
    full SDC pipeline on each.
    Returns a dict mapping run labels to per-run output dicts.
    """
    ow = {k: False for k in STEP_KEYS}
    if overwrite:
        unknown = set(overwrite) - set(STEP_KEYS)
        if unknown:
            raise ValueError('Unknown overwrite key(s): {}.  Valid: {}'.format(
                sorted(unknown), STEP_KEYS))
        ow.update(overwrite)

    bids_dir   = str(Path(bids_dir))
    output_dir = str(Path(os.path.join(bids_dir, 'derivatives', output_file)))

    func_dir = os.path.join(bids_dir, subject, session, 'func')
    fmap_dir = os.path.join(bids_dir, subject, session, 'fmap')

    subject_output_dir = os.path.join(output_dir, subject, session)
    os.makedirs(subject_output_dir, exist_ok=True)

    print('-' * 55)
    print('Processing: SDC (AFNI Method)')
    print('-' * 55)
    print(' BIDS Root : {}'.format(bids_dir))
    print(' Output    : {}'.format(output_dir))
    print(' Subject   : {}'.format(subject))
    print(' Session   : {}'.format(session))
    print(' Task      : {}'.format(task))
    print('-' * 55)

    task_glob    = 'task-{}'.format(task) if task else 'task-*'
    bold_pattern = os.path.join(
        func_dir, '{}_{}_{}_*_bold.nii*'.format(subject, session, task_glob))
    bold_files   = sorted(glob.glob(bold_pattern))

    if not bold_files:
        raise FileNotFoundError(
            'No BOLD files found for {}_{}_{}.  Searched: {}'.format(
                subject, session, task_glob, bold_pattern))

    print('\nFound {} BOLD run(s).'.format(len(bold_files)))

    all_results = {}

    for run_idx, bold_path in enumerate(bold_files, start=1):
        print('\n' + '=' * 55)
        print('Processing run {}/{}: {}'.format(
            run_idx, len(bold_files), os.path.basename(bold_path)))
        print('=' * 55)

        run_match = re.search(r'run-(\d+)', os.path.basename(bold_path))
        run_label = 'run-{}'.format(run_match.group(1)) if run_match else ''

        if run_label:
            topup_matches = glob.glob(os.path.join(
                fmap_dir,
                '{}_{}*_{}*_{}*_epi.nii*'.format(
                    subject, session, task_glob, run_label)))
            sbref_matches = glob.glob(os.path.join(
                func_dir,
                '{}_{}_{}_{}*_sbref.nii*'.format(
                    subject, session, task_glob, run_label)))
        else:
            topup_matches = [
                f for f in glob.glob(os.path.join(
                    fmap_dir,
                    '{}_{}_{}_*_epi.nii*'.format(subject, session, task_glob)))
                if not re.search(r'run-\d+', os.path.basename(f))
            ]
            sbref_matches = [
                f for f in glob.glob(os.path.join(
                    func_dir,
                    '{}_{}_{}_*_sbref.nii*'.format(subject, session, task_glob)))
                if not re.search(r'run-\d+', os.path.basename(f))
            ]

        if not topup_matches:
            raise FileNotFoundError(
                'No reverse-PE EPI found for {}.'.format(bold_path))
        if not sbref_matches:
            raise FileNotFoundError(
                'No SBREF found for {}.'.format(bold_path))

        topup_path = topup_matches[0]
        sbref_path = sbref_matches[0]

        print('  BOLD      : {}'.format(bold_path))
        print('  Reverse-PE: {}'.format(topup_path))
        print('  SBREF     : {}'.format(sbref_path))

        run_results = process_run(
            bold_path=bold_path,
            topup_path=topup_path,
            sbref_path=sbref_path,
            subject=subject,
            session=session,
            task=task,
            run_label=run_label,
            subject_output_dir=subject_output_dir,
            afni_docker=afni_docker,
            overwrite=ow,
        )

        key = run_label if run_label else 'run-{:02d}'.format(run_idx)
        all_results[key] = run_results
        print('\n  Run {} completed.'.format(run_idx))


    print('\n' + '=' * 55)
    print('All {} run(s) completed.'.format(len(bold_files)))
    print('Output directory: {}'.format(subject_output_dir))
    print('=' * 55)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='AFNI-based susceptibility distortion correction (SDC)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',   required=True, help='BIDS root directory')
    req.add_argument('--output-file', required=True, help='Output derivatives file')
    req.add_argument('--sub',        required=True, help='Subject label (e.g. sub-01)')
    p.add_argument('--ses',          default='ses-01', help='Session label')
    p.add_argument('--task',         required=True, help='Task label')
    p.add_argument('--afni-docker',
                   default=os.environ.get('AFNI_IMAGE', 'afni/afni_make_build:latest'),
                   help='AFNI Docker image tag, or "local" to run on host')

    ow = p.add_argument_group('overwrite options',
                              'Valid step names: ' + ', '.join(STEP_KEYS))
    ow.add_argument('--overwrite', nargs='+', metavar='STEP',
                    default=[], choices=STEP_KEYS,
                    help='Force re-run for named step(s).')
    ow.add_argument('--overwrite-all', action='store_true', default=False,
                    help='Force re-run for all steps.')
    return p


def main():
    args = _build_parser().parse_args()
    if args.overwrite_all:
        overwrite = {k: True for k in STEP_KEYS}
    else:
        overwrite = {k: (k in args.overwrite) for k in STEP_KEYS}
    args.sub = "sub-" + args.sub.removeprefix("sub-")
    args.ses = "ses-" + args.ses.removeprefix("ses-")
    #TODO:add option to just give an input file name and an output file name, and skip the BIDS layout discovery
    run_pipeline(
        bids_dir=args.bids_dir,
        output_file=args.output_file,
        subject=args.sub,
        session=args.ses,
        task=args.task,
        afni_docker=args.afni_docker,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()