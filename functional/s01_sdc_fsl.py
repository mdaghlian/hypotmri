#!/usr/bin/env python
"""
s01_sdc_fsl.py
================
Run FSL topup-based susceptibility distortion correction (SDC) for all BOLD
runs belonging to a given subject / session / task.

Pipeline per run
----------------
    Step 1  - Extract paired volumes          (fslroi + fslmerge)
    Step 2  - Build acqparams.txt             (JSON sidecar parsing)
    Step 3  - Run topup                       (topup)
    Step 4  - Apply topup to SBREF            (applytopup)
    Step 5  - Apply topup to full BOLD        (applytopup)

Overwrite behaviour
-------------------
Existence is checked against final BIDS-named files in *output_dir*.
If an output already exists and overwrite is False the step is skipped
and the file is restored to *work_dir* so downstream steps can use it.

    python run_sdc_topup.py ... --overwrite extract_pair run_topup

    python run_sdc_topup.py ... --overwrite-all

Valid step names for --overwrite:
    extract_pair   Step 1+2 - Extract paired volumes and write acqparams
    run_topup      Step 3   - Run FSL topup
    apply_sbref    Step 4   - Apply topup to SBREF
    apply_bold     Step 5   - Apply topup to BOLD

Usage example
-------------
python run_sdc_topup.py \\
    --bids-dir    /data/bids \\
    --output-dir  /data/derivatives/sdc \\
    --sub         sub-01 \\
    --ses         ses-01 \\
    --task        rest
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from preproc_utils import (
    build_output_name,
    check_result,
    check_skip,
    run_docker,
    run_local,
    get_nvols,
    read_bold_meta

)


# ---------------------------------------------------------------------------
# Step keys
# ---------------------------------------------------------------------------

STEP_KEYS = [
    'extract_pair',
    'run_topup',
    'apply_sbref',
    'apply_bold',
]

# ---------------------------------------------------------------------------
# PE-direction to FSL acquisition vector
# ---------------------------------------------------------------------------

_PE_TO_VECTOR = {
    'j-': '0 -1 0',
    'j':  '0  1 0',
    'i-': '-1 0 0',
    'i':  '1  0 0',
    'k-': '0  0 -1',
    'k':  '0  0  1',
}


def pe_to_vector(pe_dir: str) -> str:
    """Convert a BIDS PhaseEncodingDirection string to an FSL acqparams vector."""
    if pe_dir not in _PE_TO_VECTOR:
        raise ValueError('Unknown PE direction: {}'.format(pe_dir))
    return _PE_TO_VECTOR[pe_dir]

# ---------------------------------------------------------------------------
# Per-step functions
# ---------------------------------------------------------------------------

def extract_pair_and_acqparams(
    bold_path: str,
    topup_path: str,
    work_dir: str,
    n_vols_bold: int,
    n_vols_topup: int,
    bold_pe_vec: str,
    topup_pe_vec: str,
    bold_trt: float,
    topup_trt: float,
):
    """
    Extract the last *n_vols_topup* volumes from BOLD and all volumes from
    the reverse-PE image, merge them, and write acqparams.txt.

    Returns (fw_bw_pair path, acqparams path).
    """
    fw   = os.path.join(work_dir, 'fw.nii.gz')
    bw   = os.path.join(work_dir, 'bw.nii.gz')
    pair = os.path.join(work_dir, 'fw_bw_pair.nii.gz')

    start = n_vols_bold - n_vols_topup
    run_local(['fslroi', bold_path,  fw, str(start),        str(n_vols_topup)])
    run_local(['fslroi', topup_path, bw, '0', str(n_vols_topup)])
    run_local(['fslmerge', '-t', pair, fw, bw])

    acqparams = os.path.join(work_dir, 'acqparams.txt')
    with open(acqparams, 'w') as fh:
        for _ in range(n_vols_topup):
            fh.write('{} {}\n'.format(bold_pe_vec,  bold_trt))
        for _ in range(n_vols_topup):
            fh.write('{} {}\n'.format(topup_pe_vec, topup_trt))

    return pair, acqparams


def run_topup_cmd(
    pair_image: str,
    acqparams: str,
    work_dir: str,
    topup_config: str = 'b02b0.cnf',
) -> str:
    """Run FSL topup. Returns the topup results prefix (host path)."""
    results_prefix = os.path.join(work_dir, 'topup_results')
    run_local([
        'topup',
        '--imain={}'.format(pair_image),
        '--datain={}'.format(acqparams),
        '--config={}'.format(topup_config),
        '--out={}'.format(results_prefix),
    ])
    return results_prefix


def apply_topup_cmd(
    input_image: str,
    acqparams: str,
    topup_prefix: str,
    out_path: str,
    index: int = 1,
    method: str = 'jac',
) -> str:
    """
    Apply a topup correction field to *input_image*.  Returns *out_path*
    (with .nii.gz extension normalised).
    """
    # applytopup appends .nii.gz - strip if already present
    out_stem = out_path
    for ext in ('.nii.gz', '.nii'):
        if out_stem.endswith(ext):
            out_stem = out_stem[: -len(ext)]

    run_local([
        'applytopup',
        '--imain={}'.format(input_image),
        '--datain={}'.format(acqparams),
        '--inindex={}'.format(index),
        '--topup={}'.format(topup_prefix),
        '--method={}'.format(method),
        '--out={}'.format(out_stem),
    ])
    return out_stem + '.nii.gz'


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
    topup_config: str,
    overwrite: dict,
) -> dict:
    """
    Execute all SDC steps for a single BOLD run.
    Returns a dict of final output paths for this run.
    """
    ow = {k: False for k in STEP_KEYS}
    if overwrite:
        ow.update(overwrite)

    run_suffix_tokens = [t for t in ['task-' + task if task else None,
                                     run_label] if t]
    run_suffix = '_'.join(run_suffix_tokens) if run_suffix_tokens else 'run'

    work_dir = os.path.join(subject_output_dir, run_suffix)
    os.makedirs(work_dir, exist_ok=True)

    def _final(suffix, ext='.nii.gz'):
        return build_output_name(
            subject_output_dir, subject, session, suffix, extension=ext)

    def _work(filename):
        return os.path.join(work_dir, filename)

    # ------------------------------------------------------------------
    # Volume counts and phase-encoding metadata
    # ------------------------------------------------------------------
    n_vols_bold  = get_nvols(bold_path)
    n_vols_topup = get_nvols(topup_path)

    bold_json  = re.sub(r'\.nii(\.gz)?$', '.json', bold_path)
    topup_json = re.sub(r'\.nii(\.gz)?$', '.json', topup_path)
    bold_pe,  bold_trt  = read_bold_meta(bold_json)
    topup_pe, topup_trt = read_bold_meta(topup_json)

    bold_pe_vec  = pe_to_vector(bold_pe)
    topup_pe_vec = pe_to_vector(topup_pe)

    print('  Volume information:')
    print('    BOLD volumes      : {}'.format(n_vols_bold))
    print('    Reverse-PE volumes: {}'.format(n_vols_topup))
    print('  Phase encoding:')
    print('    BOLD PE           : {}  ->  {}'.format(bold_pe,  bold_pe_vec))
    print('    Reverse-PE        : {}  ->  {}'.format(topup_pe, topup_pe_vec))
    print('    BOLD TRT          : {}'.format(bold_trt))
    print('    Reverse-PE TRT    : {}'.format(topup_trt))

    # ------------------------------------------------------------------
    # Step 1+2 - Extract paired volumes and write acqparams
    # ------------------------------------------------------------------
    print('\n  [Step 1] Extracting paired volumes and writing acqparams...')

    pair_final      = _final('{}_fw-bw-pair'.format(run_suffix))
    acqparams_final = _final('{}_acqparams'.format(run_suffix), ext='.txt')
    pair_work       = _work('fw_bw_pair.nii.gz')
    acqparams_work  = _work('acqparams.txt')

    if not check_skip(
        {'pair': pair_final, 'acqparams': acqparams_final},
        ow['extract_pair'],
        'Step 1: extract pair + acqparams',
        workdir_paths={'pair': pair_work, 'acqparams': acqparams_work},
    ):
        pair_work, acqparams_work = extract_pair_and_acqparams(
            bold_path=bold_path,
            topup_path=topup_path,
            work_dir=work_dir,
            n_vols_bold=n_vols_bold,
            n_vols_topup=n_vols_topup,
            bold_pe_vec=bold_pe_vec,
            topup_pe_vec=topup_pe_vec,
            bold_trt=bold_trt,
            topup_trt=topup_trt,
        )
        shutil.copy(pair_work,      pair_final)
        shutil.copy(acqparams_work, acqparams_final)

    print('    pair      -> {}'.format(pair_work))
    print('    acqparams -> {}'.format(acqparams_work))

    # ------------------------------------------------------------------
    # Step 3 - Run topup
    # ------------------------------------------------------------------
    print('\n  [Step 3] Running FSL topup...')

    # Track the fieldcoef file as the sentinel for topup completion
    topup_sentinel_final = _final('{}_topup-fieldcoef'.format(run_suffix))
    topup_sentinel_work  = _work('topup_results_fieldcoef.nii.gz')
    topup_prefix_work    = _work('topup_results')

    if not check_skip(
        {'topup_fieldcoef': topup_sentinel_final},
        ow['run_topup'],
        'Step 3: FSL topup',
        workdir_paths={'topup_fieldcoef': topup_sentinel_work},
    ):
        topup_prefix_work = run_topup_cmd(
            pair_image=pair_work,
            acqparams=acqparams_work,
            work_dir=work_dir,
            topup_config=topup_config,
        )
        shutil.copy(
            topup_prefix_work + '_fieldcoef.nii.gz',
            topup_sentinel_final,
        )

    print('    topup prefix -> {}'.format(topup_prefix_work))

    # ------------------------------------------------------------------
    # Step 4 - Apply topup to SBREF
    # ------------------------------------------------------------------
    print('\n  [Step 4] Applying topup to SBREF...')

    sdc_sbref_final = _final('{}_sdc-sbref'.format(run_suffix))
    sdc_sbref_work  = _work('sdc_sbref.nii.gz')

    if not check_skip(
        {'sdc_sbref': sdc_sbref_final},
        ow['apply_sbref'],
        'Step 4: applytopup -> SBREF',
        workdir_paths={'sdc_sbref': sdc_sbref_work},
    ):
        sdc_sbref_work = apply_topup_cmd(
            input_image=sbref_path,
            acqparams=acqparams_work,
            topup_prefix=topup_prefix_work,
            out_path=sdc_sbref_work,
        )
        shutil.copy(sdc_sbref_work, sdc_sbref_final)

    print('    -> {}'.format(sdc_sbref_work))

    # ------------------------------------------------------------------
    # Step 5 - Apply topup to full BOLD
    # ------------------------------------------------------------------
    print('\n  [Step 5] Applying topup to full BOLD...')

    sdc_bold_final = _final('{}_sdc-bold'.format(run_suffix))
    sdc_bold_work  = _work('sdc_bold.nii.gz')

    if not check_skip(
        {'sdc_bold': sdc_bold_final},
        ow['apply_bold'],
        'Step 5: applytopup -> BOLD',
        workdir_paths={'sdc_bold': sdc_bold_work},
    ):
        sdc_bold_work = apply_topup_cmd(
            input_image=bold_path,
            acqparams=acqparams_work,
            topup_prefix=topup_prefix_work,
            out_path=sdc_bold_work,
        )
        shutil.copy(sdc_bold_work, sdc_bold_final)

    print('    -> {}'.format(sdc_bold_work))

    return {
        'sdc_bold':  sdc_bold_final,
        'sdc_sbref': sdc_sbref_final,
    }


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    bids_dir: str,
    output_dir: str,
    subject: str,
    session: str = 'ses-01',
    task: str = '',
    topup_config: str = 'b02b0.cnf',
    overwrite: dict = None,
) -> dict:
    """
    Discover all BOLD runs for *subject* / *session* / *task* and run the
    full FSL topup SDC pipeline on each.

    Returns a dict mapping run labels -> per-run output dicts.
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

    bids_dir   = str(Path(bids_dir).resolve())
    output_dir = str(Path(output_dir).resolve())

    func_dir = os.path.join(bids_dir, subject, session, 'func')
    fmap_dir = os.path.join(bids_dir, subject, session, 'fmap')

    subject_output_dir = os.path.join(output_dir, subject, session)
    os.makedirs(subject_output_dir, exist_ok=True)

    print('-' * 55)
    print('Processing: SDC (FSL topup)')
    print('-' * 55)
    print(' BIDS Root : {}'.format(bids_dir))
    print(' Output    : {}'.format(output_dir))
    print(' Subject   : {}'.format(subject))
    print(' Session   : {}'.format(session))
    print(' Task      : {}'.format(task))
    print('-' * 55)

    # Discover BOLD files
    task_glob    = 'task-{}'.format(task) if task else 'task-*'
    bold_pattern = os.path.join(
        func_dir,
        '{}_{}_{}_*_bold.nii*'.format(subject, session, task_glob))
    bold_files = sorted(glob.glob(bold_pattern))

    if not bold_files:
        raise FileNotFoundError(
            'No BOLD files found for {}_{}_{}. '
            'Searched: {}'.format(subject, session, task_glob, bold_pattern)
        )

    print('\nFound {} BOLD run(s).'.format(len(bold_files)))

    all_results = {}

    for run_idx, bold_path in enumerate(bold_files, start=1):
        print('\n' + '=' * 55)
        print('Processing run {}/{}: {}'.format(
            run_idx, len(bold_files), os.path.basename(bold_path)))
        print('=' * 55)

        run_match = re.search(r'run-(\d+)', os.path.basename(bold_path))
        run_label = 'run-{}'.format(run_match.group(1)) if run_match else ''

        # Locate reverse-PE and SBREF
        if run_label:
            topup_matches = glob.glob(os.path.join(
                fmap_dir,
                '{}_{}_{}_{}*_epi.nii*'.format(
                    subject, session, task_glob, run_label)))
            sbref_matches = glob.glob(os.path.join(
                func_dir,
                '{}_{}_{}_{}*_sbref.nii*'.format(
                    subject, session, task_glob, run_label)))
        else:
            topup_all = glob.glob(os.path.join(
                fmap_dir,
                '{}_{}_{}_*_epi.nii*'.format(subject, session, task_glob)))
            sbref_all = glob.glob(os.path.join(
                func_dir,
                '{}_{}_{}_*_sbref.nii*'.format(subject, session, task_glob)))
            topup_matches = [f for f in topup_all
                             if not re.search(r'run-\d+', os.path.basename(f))]
            sbref_matches = [f for f in sbref_all
                             if not re.search(r'run-\d+', os.path.basename(f))]

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
            topup_config=topup_config,
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
        description='FSL topup susceptibility distortion correction (SDC)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',   required=True,
                     help='BIDS root directory')
    req.add_argument('--output-dir', required=True,
                     help='Output derivatives directory')
    req.add_argument('--sub',        required=True,
                     help='Subject label (e.g. sub-01)')

    p.add_argument('--ses',          default='ses-01',
                   help='Session label')
    p.add_argument('--task',         default='',
                   help='Task label (empty = all tasks)')
    p.add_argument('--topup-config', default='b02b0.cnf',
                   help='FSL topup configuration file')

    ow_group = p.add_argument_group(
        'overwrite options',
        'By default, steps whose outputs already exist in output-dir are '
        'skipped.  Use the flags below to force re-runs.\n'
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

    return p


def main():
    args = _build_parser().parse_args()

    if args.overwrite_all:
        overwrite = {k: True for k in STEP_KEYS}
    else:
        overwrite = {k: (k in args.overwrite) for k in STEP_KEYS}

    run_pipeline(
        bids_dir=args.bids_dir,
        output_dir=args.output_dir,
        subject=args.sub,
        session=args.ses,
        task=args.task,
        topup_config=args.topup_config,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()