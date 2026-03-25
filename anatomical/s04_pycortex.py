#!/usr/bin/env python
import os
import argparse
import numpy as np
import nibabel as nib
import cortex
import dpu_mini.pyctx_cannibalized.subsurf2 as pcx
from dpu_mini.fs_tools import *
from dpu_mini.utils import dag_hyphen_parse

def quick_pycortex_import(sub, fsdir):
    cortex.freesurfer.import_subj(
        sub, 
        pycortex_subject=sub, 
        freesurfer_subject_dir=fsdir, 
        )
    # Import flat maps from autoflatten
    cortex.freesurfer.import_flat(
        sub, 
        'autoflatten', 
        hemis=['lh', 'rh'], 
        cx_subject=None,flat_type='freesurfer', 
        auto_overwrite=True,
        freesurfer_subject_dir=fsdir, 
        clean=True)

    sub_pcx = pcx.PyctxMaker(
        sub=sub, 
        fs_dir=fsdir, 
    )
    roi_list = dag_roi_list_expand(sub=sub, fs_dir=sub_pcx.fs_dir, roi_list='b14')
    roi_list = [i for i in roi_list if "all" not in i.lower()]
    print(roi_list)
    sub_pcx.add_rois_to_svg(roi_list)
    print('You will want to edit the overlay in inkscape')
    print('All done')

def main():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument('subject', help='Subject ID (e.g., sub-01)')
    parser.add_argument('-d', '--fsdir', 
                       default=os.environ.get('SUBJECTS_DIR', ''),
                       help='FreeSurfer subjects directory')
    
    args = parser.parse_args()
    # ensure form of sub-##
    args.subject = dag_hyphen_parse('sub', args.subject)
    if not args.fsdir:
        parser.error('Set $SUBJECTS_DIR or use --fsdir')
    
    quick_pycortex_import(args.subject, args.fsdir)


if __name__ == '__main__':
    main()