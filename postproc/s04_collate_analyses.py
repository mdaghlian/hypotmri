#!/usr/bin/env python
"""
s04_collate_analyses.py
===============
Compile everything into one csv file
- pRF parameters; benson parameters; CF parameters; CF derived from benson
- & note about the files used

Arguments:
    --bids-dir      BIDS directory containing input and output derivatives
    --prf-file      derivatives/<dir> with pRF data 
    --cf-file       derivatives/<dir> with CF data 
    --output-file   Name of derivatives directory to write combined outputs
    --sub           Subject label (e.g. sub-01)

Usage example
-------------
s04_compile.py \\
    --bids-dir /path/to/bids_dir \\
    --prf-file s03_prf \\
    --cf-file s04_cf \\
    --output-file s06_collated \\
    --sub 01 

"""

import argparse
import csv
import glob
import os
opj = os.path.join
import re
import shutil
from pathlib import Path
import yaml

import nibabel as nib
import numpy as np
import pandas as pd


from dpu_mini.fs_tools import dpu_load_roi, dpu_load_nverts 
from dpu_mini.mesh_maker import GenMeshMaker
from dpu_mini.mesh_format import dpu_pairwise_geodesic_distance
from dpu_mini.stats import dpu_coord_convert


from prfpy.stimulus import CFStimulus 
from prfpy.model import CFGaussianModel
from prfpy.fit import CFFitter

from cvl_utils.preproc_func import (
    check_skip,
    load_benson14_info,
)

from cvl_utils.prfpy_utils import (
    raw_ts_to_average_psc, 
    filter_for_nans,
    prfpy_params_dict, 
    get_dm_and_settings
)

# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    bids_dir: str,
    fs_dir: str,
    prf_file : str,
    cf_file : str,
    output_file: str,
    subject: str,
    overwrite: bool = False,
) -> dict:
    """
    Collate results across prf fitting; benson atlas; CF fitting & CF parameters derived from benson
    """
    output_dir   = str(Path(
        opj(bids_dir, 'derivatives', output_file)
    ).resolve())
    subject_output_dir   = opj(output_dir,   subject)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # ------------------------------------
    # ------------------------------------
    output_csv = opj(subject_output_dir, f'{subject}_combined.csv')
    if os.path.exists(output_csv) and not overwrite:
        print(f'Combined csv {output_csv} already exists')
        return
    
        
    pd_combined = {}
    # --- Benson 14 info --- 
    b14 = load_benson14_info(subject, fs_dir)
    total_n_vx = b14['ecc'].shape[0]
    for k in b14.keys():
        pd_combined[f'b14_{k}'] = b14[k]

    # --- PRF FILES ---
    prf_path = Path(opj(bids_dir, 'derivatives',prf_file, subject))
    print(opj(bids_dir, prf_file, subject))
    prf_csvs = list(prf_path.rglob('*iter*.csv'))
    if not prf_csvs:
        print(f'No PRF csv files found in {prf_path}')
    pd_combined = {}
    def _find_pattern(csv, target):
        match = re.search(rf'(?<=_{target}-)[^_]+', csv.name)
        return match.group() if match else None
    for prf_csv in prf_csvs:
        
        model   = _find_pattern(prf_csv, 'model')
        task    = _find_pattern(prf_csv, 'task')
        session = _find_pattern(prf_csv, 'ses')

        pd_load = pd.read_csv(prf_csv)
        index = pd_load['index'].values
        for col in pd_load.keys():
            if 'Unnamed' in col:
                continue
            if col == 'index':
                continue
            pd_combined[f'pRF_ses-{session}_{model}_{task}_{col}'] = np.zeros(total_n_vx) * np.nan
            pd_combined[f'pRF_ses-{session}_{model}_{task}_{col}'][index] = pd_load[col].values
        print(f'Loaded {session} {model} {task} pRF fits')
        print(f'(file: {prf_csv.name}, n={len(index)})')


    # --- CF FILES ---
    cf_path = Path(opj(bids_dir, 'derivatives', cf_file, subject))
    print(opj(bids_dir, cf_file, subject))
    cf_csvs = list(cf_path.rglob('*cf.csv'))
    if not cf_csvs:
        print(f'No CF csv files found in {cf_path}')

    for cf_csv in cf_csvs:
        # Fix pattern: [^_]+ matches one or more non-underscore characters
        task    = _find_pattern(cf_csv, 'task')
        session = _find_pattern(cf_csv, 'ses')

        pd_load = pd.read_csv(cf_csv)
        index   = pd_load['index'].values
        centre  = pd_load['centre'].values

        for col in pd_load.keys():
            if 'Unnamed' in col:
                continue
            if col == 'index':
                continue
            pd_combined[f'cf_ses-{session}_{task}_{col}'] = np.zeros(total_n_vx) * np.nan
            pd_combined[f'cf_ses-{session}_{task}_{col}'][index] = pd_load[col].values
        print(f'Loaded {session} {task} cf fits')
        print(f'(file: {cf_csv.name}, n={len(index)})')
        print(centre)
        # --- b14 derived CF parameters ---
        for p in ['ecc', 'pol', 'size']:
            tkey = f'cf_b14{session}_{task}_{p}'
            centre_key = np.array([b14['ecc'][int(c)] for c in centre])
            pd_combined[tkey] = np.zeros(total_n_vx) * np.nan
            pd_combined[tkey][index] = centre_key.copy()
        print(f'Added b14-derived CF parameters for {session} {task}')
    print(f'Collated parameters for {subject} (n={total_n_vx} vertices)')
    print(f'Columns include: {list(pd_combined.keys())}')
    print(f'Writing combined csv to {output_csv}')
    pd.DataFrame(pd_combined).to_csv(output_csv, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Collate results across prf fitting; benson atlas; CF fitting & CF parameters derived from benson',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group('required arguments')
    req.add_argument('--bids-dir',    required=True,
                     help='BIDS directory')
    req.add_argument('--prf-file',   required=False,
                     help='PRF parameter folder')
    req.add_argument('--cf-file',   required=False,
                    help='CF parameter folder')
    req.add_argument('--output-file', required=True,
                     help='Where to put the fits')
    req.add_argument('--sub',         required=True,
                     help='Subject label (e.g. sub-01 or 01)')

    req.add_argument('--overwrite', action='store_true', default=False,
                     help='Overwrite existing combined csv if it exists')
    # ... for consistency with other scripts, we also add an --overwrite-all flag, even though it doesn't do anything different in this script since there's only one step to overwrite
    req.add_argument('--overwrite-all', action='store_true', default=False,
                     help='Overwrite existing combined csv if it exists')

    return p


def main():
    args = _build_parser().parse_args()

    if (args.overwrite_all) or (args.overwrite):
        overwrite = True
    else:
        overwrite = False

    args.sub = 'sub-' + args.sub.removeprefix('sub-')
    
    run_pipeline(
        bids_dir        = args.bids_dir,
        fs_dir          = opj(args.bids_dir, 'derivatives', 'freesurfer'),
        prf_file        = args.prf_file,
        cf_file         = args.cf_file,
        output_file     = args.output_file,
        subject         = args.sub,
        overwrite       = overwrite,
    )


if __name__ == '__main__':
    main()