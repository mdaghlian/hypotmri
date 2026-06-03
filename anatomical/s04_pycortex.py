#!/usr/bin/env python
import os
opj = os.path.join
import argparse
import numpy as np
import nibabel as nib
import cortex
import dpu_mini.pyctx.subsurf_experiment as pcx
from dpu_mini.fs_tools import *

def set_ctx_path(p=None, opt="update"):
    """set_ctx_path

    Function that changes the filestore path in the cortex config file to make changing between projects flexible. Just specify the path to the new pycortex directory to change. If you do not specify a string, it will default to what it finds in the os.environ['CTX'] variable as specified in the setup script. You can also ask for the current filestore path with "opt='show_fs'", or the path to the config script with "opt='show_pn'". To update, leave 'opt' to 'update'.

    Parameters
    ----------
    p: str, optional
        either the path we need to set `filestore` to (in combination with `opt="update"`), or None (in combination with `opt="show_fs"` or `opt="show_pn"`)
    opt: str
        either "update" to update the `filestore` with `p`; "show_pn" to show the path to the configuration file; or "show_fs" to show the current `filestore`

    Example
    ----------
    >>> set_ctx_path('path/to/pycortex', "update")
    """
    # ************************
    cortex.options.config.set('basic', 'filestore', p)
    # cortex.db.filestore=p
    # ************************
    if p == None:
        p = os.environ.get('CTX')

    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

    usercfg = cortex.options.usercfg
    import configparser
    config = configparser.ConfigParser()
    config.read(usercfg)

    # check if filestore exists
    try:
        config.get("basic", "filestore")
    except:
        config.set("basic", "filestore", p)
        with open(usercfg, 'w') as fp:
            config.write(fp)

    if opt == "show_fs":
        return config.get("basic", "filestore")
    elif opt == "show_pn":
        return usercfg
    else:
        if config.get("basic", "filestore") != p:
            config.set("basic", "filestore", p)
            with open(usercfg, 'w') as fp:
                config.write(fp)
            
            if not os.path.exists(p):
                os.makedirs(p, exist_ok=True)

            return config.get("basic", "filestore")
        else:
            return config.get("basic", "filestore")

def quick_pycortex_import(sub, fsdir, pycortex_dir):
    # force to correct path
    try: 
        set_ctx_path(p=pycortex_dir, opt="update")
    except Exception as e:
        print(f"Error setting pycortex path: {e}")
        return
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
    roi_list = dpu_roi_list_expand(sub=sub, fs_dir=sub_pcx.fs_dir, roi_list='b14')
    roi_list = [i for i in roi_list if "all" not in i.lower()]
    print(roi_list)
    sub_pcx.add_rois_to_svg(roi_list)
    print('You will want to edit the overlay in inkscape')
    print('All done')

def main():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument('--sub', type=str, help='Subject ID (e.g. sub-01)')
    parser.add_argument('--bids-dir', default=os.environ.get('BIDS_DIR', None))
    parser.add_argument('--fsdir', 
                       default=os.environ.get('SUBJECTS_DIR', None),
                       help='FreeSurfer subjects directory')
    parser.add_argument('--pycortex-dir', 
                        default=os.environ.get('PYCTX_DIR', None),
                        )
    
    args = parser.parse_args()
    # ensure form of sub-##
    args.sub = "sub-" + args.sub.removeprefix("sub-")
    if not args.fsdir:
        parser.error('Set $SUBJECTS_DIR or use --fsdir')

    if args.bids_dir is not None:
        if args.fsdir is None:
            args.fsdir = opj(args.bids_dir, 'freesurfer')
        if args.pycortex_dir is None:
            args.pycortex_dir = opj(args.bids_dir, 'pycortex_store')
    if not os.path.exists(args.pycortex_dir):
        os.makedirs(args.pycortex_dir)
    quick_pycortex_import(args.sub, args.fsdir, args.pycortex_dir)


if __name__ == '__main__':
    main()