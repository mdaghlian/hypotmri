import os
opj = os.path.join
import pickle
import sys
import numpy as np
import re
import scipy.io
from copy import copy

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel

import nibabel as nib

import pandas as pd

from dpu_mini.fs_tools import *
from dpu_mini.utils import *
from dpu_mini.pyctx.subsurf_experiment import *

from cvl_utils.prfpy_utils import prfpy_params_dict, Prf1T1M
from cvl_utils.prfpy_plotter import TSPlotter
import cortex

def cvl_auto_surf_function(surf_type, **kwargs):
    '''
    ---------------------------
    Auto open a subject surface

    Args:
        surf_type               plot using 'dash' or 'fs'
        param_path              path to .csv file 
        sub                     subject number
        fs_dir                  freesurfer director
        output_dir               where to put it
        file_name               name of the file
        model                   prfpy model to use
        real_ts                 path to real timeseries
        dm_file                 path to design matrix file
        hemi_markers            How are hemispheres marked in file?
        dump                    dump the mesh object
        open                    open the surface
	    port 			what port to host dash server on
	    host 			what ip to host dash on
        ow_prfpy_model  overwrite the prfpy_model (if stored in pickle) 

    ''' 
    # Parse the arguments
    param_path = kwargs.pop('param_path', None)   
    sub = kwargs.pop('sub', None)
    fs_dir = kwargs.pop('fs_dir', os.environ['SUBJECTS_DIR'])
    n_vx = np.sum(dpu_load_nverts(sub, fs_dir))

    fs_dir = kwargs.pop('fs_dir', os.environ['SUBJECTS_DIR'])    
    
    if not os.path.exists(fs_dir):
        print('Could not find SUBJECTS_DIR')
        print(fs_dir)
        sys.exit()
    output_dir = kwargs.pop('output_dir', os.getcwd())
    file_name = kwargs.pop('file_name', 'auto_surf')
    hemi_markers = kwargs.pop('hemi_markers', ['lh', 'rh'])
    # Sort out how we id hemisphere
    # -> people (me) are annoyingly inconsistent with how they hame there hemispheres (I'm working on it)
    # dm_file = kwargs.pop('dm_file', None)
    # real_ts = kwargs.pop('real_ts', None)
    model = kwargs.pop('model', None)
    
    dump = kwargs.pop('dump', False)
    open_surf = kwargs.pop('open', True)
    port = kwargs.pop('port', 8000)
    host = kwargs.pop('host', '127.0.0.1')

    min_rsq = kwargs.pop('min_rsq', 0.01)
    max_ecc = kwargs.pop('max_ecc', 5)
    extra_kwargs = copy(kwargs)
    
    pd_params = pd.read_csv(param_path)

    data_sub_mask = np.zeros(n_vx, dtype=bool)
    data_sub_mask[pd_params['index']] = True
    pars_to_plot = list(pd_params.keys())[2:]
    
    # ****************************************************
    if surf_type == 'pycortex':
        ctx_method = kwargs.pop('ctx_method', 'custom')
        
        # Make the mesh dash object
        fs = PyctxMaker(
            sub=sub, 
            fs_dir=fs_dir,
            output_dir=output_dir,
            )    

        for p in pars_to_plot:
            data      = pd_params[p].to_numpy()
            if np.unique(data).shape[0]<=1:
                print(f'{p} - has no unique values')
                break
            data_rsq   = pd_params['rsq'].to_numpy()
            data_mask   = pd_params['rsq'].to_numpy()>min_rsq
            if 'ecc' in pd_params.keys():
                data_mask &= pd_params['ecc'].to_numpy()<max_ecc
            if p=='pol':
                cmap = 'marco_pol'
                vmin,vmax = -np.pi, np.pi
                ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif p=='ecc':
                cmap = 'jet'
                vmin,vmax = 0, int(np.nanmax(data))
                ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif p=='rsq':
                cmap='plasma'
                vmin,vmax = 0,1
                ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)                
            elif p in ('x', 'y'):
                cmap = 'RdBu'
                vmin,vmax = -int(np.nanmax(data)), int(np.nanmax(data))
                ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

            else:
                ctx_kwargs = {}
            ctx_kwargs['ctx_method'] = 'custom' #ctx_method
            if ctx_method == 'vertex2d':
                ctx_kwargs['data_alpha'] = data_rsq + np.random.rand(data_rsq.shape[0])*.10
            data_ = np.zeros(data_sub_mask.shape[0])
            data_[data_sub_mask] = data
            data_mask_ = np.zeros(data_sub_mask.shape[0], dtype=bool)
            data_mask_[data_sub_mask] = data_mask
            if 'rsq' in p:
                data_mask_ = np.ones_like(data_mask_, dtype=bool)

            fs.add_vertex_obj(
                data=data_, 
                data_mask=data_mask_,
                # data_sub_mask=data_sub_mask,
                surf_name=p,  
                **ctx_kwargs,  
            )
            # break
        if dump:
            # fs.return_pyc_saver(viewer=False)
            # fs.pyc.to_static(filename=file_name)
            tdict = {
                'x':fs.vertex_dict['x'],
                'y':fs.vertex_dict['y'],
            }
            cortex.webgl.make_static(
                file_name,
                tdict, 
                )
        if False: #open_surf:  
            print(fs.vertex_dict)          
            tdict = {
                'x':fs.vertex_dict['x'],
                'y':fs.vertex_dict['y'],
            }
            cortex.webgl.show(
                tdict, 
                # port=np.random.randint(8000, 9000),
                open_browser=True,
                autoclose=False,
                )
    # ****************************************************
    # ****************************************************
    # FS OBJECT
    elif surf_type == 'fs':
        # FS OBJECT
        fs = FSMaker(
            sub=sub, 
            fs_dir=fs_dir,
            )
            
        for p in pars_to_plot:
            data        = pd_params[p].to_numpy()
            data_mask   = pd_params['rsq'].to_numpy()>min_rsq
            if 'ecc' in pd_params.keys():
                data_mask &= pd_params['ecc'].to_numpy()<max_ecc
            if p=='pol':
                cmap = 'marco_pol'
                vmin,vmax = -np.pi, np.pi
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif p=='ecc':
                cmap = 'ecc2'
                vmin,vmax = 0, 5
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif p=='rsq':
                cmap='plasma'
                vmin,vmax = 0,1
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                kwargs = {}

            fs.add_surface(
                data=data, 
                data_mask = data_mask,
                data_sub_mask=data_sub_mask,
                surf_name=f'{file_name}_{p}',  
                **kwargs,  
            )
        if open_surf:
             fs.open_fs_surface(fs.surf_list, **extra_kwargs)


# def cvl_auto_from_prf_obj(prf_obj, sub, **kwargs):
#     '''
#     ---------------------------
#     Auto open a subject surface

#     Args:
#         sub                     subject number
#         prf_obj                 prf object 
#         surf_type               plot using 'dash' or 'fs'
#         fs_dir                  freesurfer director
#         output_dir               where to put it
#         file_name               name of the file
#         model                   prfpy model to use
#         dump                    dump the mesh object
#         open                    open the surface
# 	    port 			what port to host dash server on
# 	    host 			what ip to host dash on

#     ''' 
#     # Parse the arguments
#     fs_dir = kwargs.pop('fs_dir', os.environ['SUBJECTS_DIR'])    
#     if not os.path.exists(fs_dir):
#         print('Could not find SUBJECTS_DIR')
#         print(fs_dir)
#         sys.exit()
    
#     output_dir = kwargs.pop('output_dir', os.getcwd())
#     file_name = kwargs.pop('file_name', 'auto_surf')
#     surf_type = kwargs.pop('surf_type', 'dash')
#     dump = kwargs.pop('dump', False)
#     open_surf = kwargs.pop('open', False)
#     port = kwargs.pop('port', 8000)
#     host = kwargs.pop('host', '127.0.0.1')
#     pars_to_plot = kwargs.pop('pars_to_plot', None)
#     min_rsq = kwargs.pop('min_rsq', 0.1)
#     max_ecc = kwargs.pop('max_ecc', 5)
#     return_fs = kwargs.pop('return_fs', False)
#     extra_kwargs = copy(kwargs)

#     # DASH OBJECT
#     if surf_type == 'dash':
        
#         # Make the mesh dash object
#         fs = MeshDash(
#             sub=sub, 
#             fs_dir=fs_dir,
#             output_dir=output_dir,
#             )    

#         fs.web_get_ready(**extra_kwargs)
        
#         if pars_to_plot is None:
#             pars_to_plot = list(pd_params.keys())

#         for p in pars_to_plot:
#             data        = pd_params[p].to_numpy()
#             if '-' in p:
#                 # This is a multi object. Only get the rsq for the specific one...
#                 prf_id = p.split('-')[0]
#                 data4mask   = prf_obj[prf_id].pd_params['rsq'].to_numpy()
#             else:
#                 data4mask   = pd_params['rsq'].to_numpy()

#             if 'pol' in p:
#                 cmap = 'marco_pol'
#                 vmin,vmax = -np.pi, np.pi
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
#             elif 'ecc' in p:
#                 cmap = 'ecc2'
#                 vmin,vmax = 0, int(np.nanmax(data))
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
#             elif 'rsq' in p:
#                 cmap='plasma'
#                 vmin,vmax = 0,1
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)                
#             elif ('x' in p) or ('y' in p):
#                 cmap = 'RdBu'
#                 vmin,vmax = -int(np.nanmax(data)), int(np.nanmax(data))
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

#             else:
#                 kwargs = {}
            
#             fs.web_add_vx_col(
#                 data=data, 
#                 # data_alpha=data_alpha, 
#                 data4mask = data4mask,
#                 vx_col_name=p,  
#                 **kwargs,  
#             )
#             # break
#         if hasattr(prf_obj, 'id_list'):
#             # It is a multi figure...
#             for prf_id in id_list:
#                 fs.web_add_mpl_fig_maker(
#                     mpl_func=prf_obj[prf_id].prf_ts_plot, 
#                     mpl_key=f'{prf_id}_plot',
#                     mpl_kwargs={'return_fig':True},
#                 )
#         else:
#             fs.web_add_mpl_fig_maker(
#                 mpl_func=prf_ts_plot, 
#                 mpl_key='plot',
#                 mpl_kwargs={'return_fig':True},
#             )

#         if dump:            
#             dag_mesh_pickle(fs, file_name=file_name)
#         if open_surf:
#             app = fs.web_launch_with_dash()
#             # Open the app in a browser
#             # Do not show it in the notebook
#             print(f'http://localhost:{port}/')
#             # Fix for running in macs
#             import matplotlib
#             matplotlib.use('Agg')            
#             app.run_server(host=host, port=port, debug=False, use_reloader=False)             
    
#     else:
#         # FS OBJECT
#         fs = FSMaker(
#             sub=sub, 
#             fs_dir=fs_dir,
#             )
         
#         if pars_to_plot is None:
#             pars_to_plot = list(pd_params.keys())
        
#         for p in pars_to_plot:
#             data        = pd_params[p].to_numpy()
#             if '-' in p:
#                 # This is a multi object. Only get the rsq for the specific one...
#                 prf_id = p.split('-')[0]
#                 data_mask   = prf_obj[prf_id].pd_params['rsq'].to_numpy() > min_rsq
#                 if 'ecc' in prf_obj[prf_id].pd_params.keys():
#                     data_mask &= prf_obj[prf_id].pd_params['ecc'].to_numpy() < max_ecc
#             else:
#                 data_mask   = pd_params['rsq'].to_numpy()>min_rsq
#                 if 'ecc' in pd_params.keys():
#                     data_mask &= pd_params['ecc'].to_numpy()<max_ecc
 
#             if 'pol' in p:
#                 cmap = 'marco_pol'
#                 vmin,vmax = -np.pi, np.pi
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
#             elif 'ecc' in p:
#                 cmap = 'ecc2'
#                 vmin,vmax = 0, int(np.nanmax(data))
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
#             elif 'rsq' in p:
#                 cmap='plasma'
#                 vmin,vmax = 0,1
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)                
#             elif ('x' in p) or ('y' in p):
#                 cmap = 'RdBu'
#                 vmin,vmax = -int(np.nanmax(data)), int(np.nanmax(data))
#                 kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
#             else:
#                 kwargs = {}
            
#             fs.add_surface(
#                 data=data, 
#                 data_mask = data_mask,
#                 surf_name=f'{file_name}_{p}',  
#                 **kwargs,  
#             )

#         if open_surf:
#              fs.open_fs_surface(fs.surf_list, **extra_kwargs)

#     if return_fs:
#         return fs
