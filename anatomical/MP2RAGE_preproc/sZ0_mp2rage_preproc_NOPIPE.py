#!/usr/bin/env python
import argparse
import os
opj = os.path.join
import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib


# Get the absolute path of the directory containing the current script
mp2rage_script_dir = Path(__file__).resolve().parent

def run(cmd, **kwargs):
    """Run a shell command, raise on failure."""
    print(f'  $ {cmd}')
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def mprage_ise(uni_file, inv2_file, out_file):
    """Suppress MP2RAGE background noise by multiplying uni by normalised INV2."""
    uni_img  = nib.load(uni_file)
    inv2_img = nib.load(inv2_file)

    uni_data  = uni_img.get_fdata()
    inv2_data = inv2_img.get_fdata()

    norm_factor    = np.percentile(inv2_data[inv2_data > 0], 99)
    uni_clean_data = (inv2_data / norm_factor) * uni_data

    nib.save(
        nib.Nifti1Image(uni_clean_data, uni_img.affine, uni_img.header),
        out_file,
    )


def biascorrect_spm(input_file, output_file, spm_standalone=None, mcr_path=None):
    """Bias correct using SPM via standalone or MATLAB.
    
    Parameters
    ----------
    input_file      : str | Path  Input NIfTI file (.nii or .nii.gz)
    output_file     : str | Path  Destination for the bias-corrected NIfTI
    spm_standalone  : str | Path  Path to the SPM standalone executable (optional)
    mcr_path        : str | Path  Path to the MATLAB MCR directory (optional)
                                  Required when spm_standalone is provided
    """
    input_path  = Path(input_file)
    output_path = Path(output_file)

    script       = 'sZ0_spmbc'

    # Resolve clean stem (handles both .nii and .nii.gz)
    stem = input_path.stem
    if input_path.suffix == '.gz':
        stem = Path(stem).stem

    # SPM writes outputs into a spm_biascorrect/ subdirectory beside the input
    spm_out_dir      = output_path.parent / f'{stem}_spm_biascorrect'
    spm_biascorrected = spm_out_dir / f'{stem}_biascorrected.nii'

    print('')
    print('+' * 67)
    print('Starting SPM Bias-correction')
    print('+' * 67)
    print(f'  Input  : {input_path}')
    print(f'  Output : {output_path}')

    if spm_standalone and mcr_path:
        print(f'  Mode   : SPM standalone')
        print(f'  SPM    : {spm_standalone}')
        print(f'  MCR    : {mcr_path}')
        cmd = (
            f'"{spm_standalone}" "{mcr_path}" script'
            f' "{script}(\'{input_path}\', \'{spm_out_dir}\');"'
        )
    else:
        print(f'  Mode   : MATLAB')
        matlab_cmd = f"{script}(\'{input_path}\', \'{spm_out_dir}\'); exit;"
        cmd = f"matlab -nodisplay -nosplash -nodesktop -r \"{matlab_cmd}\""

    run(cmd, cwd=mp2rage_script_dir)

    # Verify expected output exists
    if not spm_biascorrected.exists():
        raise FileNotFoundError(
            f'SPM bias correction completed but output not found:\n'
            f'  Expected: {spm_biascorrected}'
        )

    # Copy result to the requested output location
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    run(f'gzip {spm_biascorrected} -f')
    shutil.copy(f'{spm_biascorrected}.gz', output_path)

    print('')
    print('+' * 67)
    print('Completed SPM Bias-correction')
    print('+' * 67)
    print(f'  Result : {output_path}')
    print('')


def biascorrect_ants(input_file, output_file):
    '''N4 Bias field Correction'''
    run(f'N4BiasFieldCorrection -d 3 -i {input_file} -o {output_file}')



def preprocess(sub, ses, uni_file, inv2_file, outputdir, biascorr='ants', ):
    sub = str(sub).replace('sub-', '')
    ses = str(ses).replace('ses-', '')

    outputdir = Path(outputdir).absolute()
    outputdir.mkdir(parents=True, exist_ok=True)

    uni_file  = str(Path(uni_file).absolute())
    inv2_file = str(Path(inv2_file).absolute())

    # ── 1. Bias-correct INV2 ─────────────────────────────────────────────
    inv2_bc = str(outputdir / 'inv2_bc.nii.gz')
    if biascorr == 'ants':
        print('\n[1/5] N4 bias correction on INV2 (ANTs)...')
        biascorrect_ants(inv2_file, inv2_bc)
    elif biascorr == 'spm':
        print('\n[1/5] Bias correction on INV2 (SPM)...')
        biascorrect_spm(inv2_file, inv2_bc, ) #spm_standalone, mcr_path)
    elif biascorr == 'none':
        print('\n[1/5] Skipping bias correction...')
        if inv2_file.endswith('gz'):
            shutil.copy(inv2_file, inv2_bc)
        else:
            shutil.copy(inv2_file, inv2_bc.replace('.gz',''))
            run(f"gzip {inv2_bc.replace('.gz','')} -f")

    else:
        raise ValueError(f'Unknown biascorr method: {biascorr}')
    
    # ── 2. BET on bias-corrected INV2 ───────────────────────────────────
    print('\n[2/5] Brain extraction (BET) on bc-corrected INV2...')
    bet_base = str(outputdir / 'inv2_bet')
    run(f'bet {inv2_bc} {bet_base} -m -R -f 0.3')
    mask_bet = bet_base + '_mask.nii.gz'

    # ── 3. MPRAGEise uni ─────────────────────────────────────────────────
    print('\n[3/5] MPRAGEising uni (background suppression)...')
    uni_clean = str(outputdir / 'uni_MPRAGEised.nii.gz')
    mprage_ise(uni_file, inv2_bc, uni_clean)

    uni_cleanbc = str(outputdir / 'uni_MPRAGEisedbc.nii.gz')
    biascorrect_spm(uni_clean, uni_cleanbc, ) 
    # # ── 2. BET on bias-corrected MPRAGE ───────────────────────────────────
    # print('\n[2/5] Brain extraction (BET) on bc-corrected INV2...')
    # bet_base = str(outputdir / 'mprageise_bet')
    # run(f'bet {uni_clean} {bet_base} -m -R -f 0.3')
    # # mask_bet = bet_base + '_mask.nii.gz'

    # # ── 4. Copy geometry from MPRAGEised uni onto mask ──────────────────
    # print('\n[4/5] Copying geometry to brain mask...')
    # mask_matched = str(outputdir / 'mask_matched.nii.gz')
    # shutil.copy(mask_bet, mask_matched)
    # run(f'fslcpgeom {uni_clean} {mask_matched}')


    print(f'\nDone. Outputs written to: {outputdir}')








# ************************************************************
# ************************************************************
# ************************************************************
def main():
    p = argparse.ArgumentParser(description='MP2RAGE preprocessing: denoise + bias correction')
    p.add_argument('--sub',       required=True)
    p.add_argument('--ses',       required=True)
    p.add_argument('--uni',       required=True)
    p.add_argument('--inv2',      required=True)
    p.add_argument('--outputdir', required=True)
    p.add_argument('--biascorr',  required=False, default='spm') 
    # spm generally does a better job
    
    args = p.parse_args()
    preprocess(
        sub=args.sub,
        ses=args.ses,
        uni_file=args.uni,
        inv2_file=args.inv2,
        biascorr=args.biascorr,
        outputdir=args.outputdir,
    )


if __name__ == '__main__':
    main()