#!/usr/bin/env python
"""
s00_prf_surf.py
===============
View prf parameters on cortical surface

Arguments:
    --prf-csv       path to prf-csv 
    --output-file   Where to dump
    --sub           Subject label (e.g. sub-01)

"""
#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
opj = os.path.join
import pickle
import sys
import numpy as np
import re
import scipy.io

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel

from cvl_utils.prfpy_surf import cvl_auto_surf_function
from dpu_mini.utils import *
def main(argv):
    '''
    ---------------------------
    Auto open a subject surface

    Args:
        --surf_type     type of surface plotter (dash, fs)
        --param_path    path to .pkl/.npy/.gii/.mgz file 
        --sub           subject number
        --fs_dir        freesurfer director
        --output_dir     where to put it
        --file_name     name of the file to dump
        --model         prfpy model to use
    
    '''
    surf_type = 'fs'
    sub = None
    fs_dir = os.environ['SUBJECTS_DIR']
    dump = False
    file_name = 'prf'
    model = None
    output_dir = opj(os.getcwd())
    hemi_markers = ['lh', 'rh']
    param_path = None
    extra_kwargs = {}
    for i,arg in enumerate(argv):        
        if '--surf_type' in arg:
            surf_type = argv[i+1]
        elif arg in ('--param_path', '--params_path', '--path'):
            param_path = argv[i+1]
        elif '--sub' in arg:
            sub = dpu_hyphen_parse('sub', argv[i+1])
        elif '--model' in arg:
            model = argv[i+1]            
        elif '--fs_dir' in arg:
            fs_dir = argv[i+1]  
        elif '--output_dir' in arg:
            output_dir = argv[i+1]              
        elif '--dump' in arg:
            dump = True
        elif '--file_name' in arg:
            file_name = argv[i+1]
        elif '--hemi_markers' in arg:
            hemi_markers = argv[i+1].split(',')  
        elif ('--' in arg) and ('_path' in arg):
            this_param = arg.replace('--', '').replace('_path', '')
            specific_param_path[this_param] = argv[i+1]
        elif arg in ('-h', '--help'):
            print(main.__doc__)
            sys.exit()

        elif '--' in arg:
            this_kwarg = arg.replace('--', '')
            this_kwarg_value = dpu_arg_checker(argv, i+1)
            extra_kwargs[this_kwarg] = this_kwarg_value
            print(f'Unknown arg: {arg}')
    extra_kwargs['title'] = extra_kwargs.get('title', f'{sub} {model} \n {param_path}')
    cvl_auto_surf_function(
        surf_type=surf_type,
        param_path=param_path,
        sub=sub,
        model=model,
        fs_dir=fs_dir,
        output_dir=output_dir,
        dump=dump,
        file_name=file_name,
        **extra_kwargs
    )


if __name__ == "__main__":
    main(sys.argv[1:])    