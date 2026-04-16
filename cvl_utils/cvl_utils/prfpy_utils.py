import numpy as np
import os
opj = os.path.join
import matplotlib.image as mpimg
from scipy import io, interpolate
from scipy.ndimage import zoom

opj = os.path.join

from cvl_utils.prf_utils import *

def prfpy_params_dict():
    '''
    Easy look up table for prfpy model parameters
    name to index...
    '''
    p_order = {}
    # [1] gauss. Note hrf_1, and hrf_2 are idx 5 and 6, if fit...
    p_order['gauss'] = {
        'mu_x'          :  0, # mu_x
        'mu_y'          :  1, # mu_y
        'size'          :  2, # size
        'beta'          :  3, # amplitude
        'baseline'      :  4, # bold baseline 
        'hrf_1'         :  5, # *hrf_derivative
        'hrf_2'         :  6, # *hrf_dispersion
        'rsq'           : -1, # ... 
    }    
    # [2] css. Note hrf_1, and hrf_2 are idx 6 and 7, if fit...
    p_order['css'] = {
        'mu_x'              :  0, # mu_x
        'mu_y'              :  1, # mu_y
        'size'              :  2, # size
        'beta'              :  3, # amplitude
        'baseline'          :  4, # bold baseline 
        'n'                 :  5, # (the exponent)
        'hrf_1'             :  6, # *hrf_derivative
        'hrf_2'             :  7, # *hrf_dispersion
        'rsq'               : -1, # ... 
    }

    # [3] dog. Note hrf_1, and hrf_2 are idx 7 and 8, if fit...
    p_order['dog'] = {
        'mu_x'              :  0, # 
        'mu_y'              :  1, # 
        'prf_size'          :  2, # 
        'prf_amplitude'     :  3, # 
        'bold_baseline'     :  4, #  
        'srf_amplitude'     :  5, # amplitude of surround
        'srf_size'          :  6, # size of surround
        'hrf_1'             :  7, # *hrf_dispersion
        'hrf_2'             :  8, # *hrf_derivative        
        'rsq'               : -1, # ... 
    }
    
    # R(norm) = ((a*prf + b) / (c*srf + d)) - (b/d)
    p_order['norm'] = {
        'mu_x'                  :  0, # mu_x
        'mu_y'                  :  1, # mu_y
        'prf_size'              :  2, # prf_size
        'prf_amplitude'         :  3, # prf_amplitude (aka "a")
        'bold_baseline'         :  4, # bold_baseline 
        'srf_amplitude'         :  5, # surround amplitude (aka "c")
        'srf_size'              :  6, # srf_size
        'neural_baseline'       :  7, # aka "b"
        'surround_baseline'     :  8, # aka "d"
        'hrf_1'     :  9, # *hrf_1
        'hrf_2'      : 10, # *hrf_2        
        'rsq'           : -1, # rsq
    }            

    p_order['csf']  ={
        'width_r'       : 0,
        'SFp'           : 1,
        'CSp'           : 2,
        'width_l'       : 3,
        'crf_exp'       : 4,
        'beta'          : 5,
        'baseline'      : 6,
        'hrf_1'         : 7,
        'hrf_2'         : 8,
        'rsq'           : -1,
    }

    return p_order

def make_vx_wise_bounds(n_vx, bounds_in, **kwargs):
    '''make_vx_wise_bounds        
    In prfpy you normally you set the bounds for all voxels to be the same
    Sometimes we want to do voxel wise bounds (e.g., to fix one parameter but 
    fit the others)
    
    Input:
    ----------
    n_vx        Number of voxels/vertices being fit
    bounds_in   list of tuples, for each parameter a min and a max 
    Optional:
    model           Which model are we setting bounds for
    fix_param_dict  Dictionary of parameter, and the values you are fixing 
                    e.g., vx_bound_dict = {'hrf_deriv' : np.ndarray} where 
                    np.ndarray is the length of the number of voxels
    
    Output:
    ----------
    vx_wise_bounds  np.ndarray, n_vx x n_pars x 2
    '''
    model=kwargs.get('model', None)
    fix_param_dict = kwargs.get('fix_param_dict', None)
    
    if isinstance(bounds_in, list):
        bounds_in = np.array(bounds_in)
    vx_wise_bounds = np.repeat(bounds_in[np.newaxis, ...], n_vx, axis=0)

    if not fix_param_dict is None:
        # Fix the specific parameters
        model_idx = prfpy_params_dict()[model]
        for p in fix_param_dict.keys():
            # Upper and lower bound the same...
            vx_wise_bounds[:,model_idx[p],0] = fix_param_dict[p]
            vx_wise_bounds[:,model_idx[p],1] = fix_param_dict[p]

    return vx_wise_bounds


# ************************************
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from copy import deepcopy
import sys
opj = os.path.join

from prfpy.csenf_plot_functions import *
from prfpy.csenf_plot_functions import ncsf_calculate_aulcsf, ncsf_calculate_sfmax

from dpu_mini.utils import *
from dpu_mini.stats import *
from dpu_mini.plot_functions import *

def quick_rf(x, y, size, **kwargs):
    '''quick_rf
    Quick calculation of a gaussian rf
    '''
    prfpy_stim = kwargs.get('prfpy_stim', None)
    normalize_RFs = kwargs.get('normalize_RFs', False)
    x_coords = kwargs.get('x_coords', None)
    y_coords = kwargs.get('y_coords', None)
    if (prfpy_stim is None) & (x_coords is None):        
        print('x and y coords not given, making some up')
        x_coords = np.linspace(-10,10,100)
        y_coords = np.linspace(-10,10,100)
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)    
    elif x_coords is None:
        x_coords = prfpy_stim.x_coordinates
        y_coords = prfpy_stim.y_coordinates
    elif len(x_coords.shape)==1:
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)
        
    rf = np.rot90(gauss2D_iso_cart(
        x=x_coords[...,np.newaxis],
        y=y_coords[...,np.newaxis],
        mu=(x,y),
        sigma=size,
        normalize_RFs=normalize_RFs).T,axes=(1,2))
    return rf

# ********** PRF OBJECTS

def pd_to_np(prf_pd, model):
    if model == 'gauss':
        prf_np = np.zeros((prf_pd.shape[0], 8))
        pdict = prfpy_params_dict()['gauss']
        for k in pdict.keys():
            prf_np[:,pdict[k]] = prf_pd[k].to_numpy()
    return prf_np

class Prf1T1M(object):
    '''Prf1T1M
    For use with prfpy. Labels determined by prfpy
    Class for parsing prfpy output for 1 subject, 1 task, 1 model
    It will hold everything in the original numpy array, and also 
    convert it to a pandas dataframe for easy plotting, and analysis.
    
    Includes useful functions for plotting and analysis:    
    >> return_vx_mask: returns a mask for voxels
    >> return_th_param: returns the specified parameters, masked by the vx_mask
    >> hist: plot a histogram of a parameter, masked by the vx_mask
    >> visual_field: plot voxels around the visual field of the voxels, masked by the vx_mask
        and colored by a parameter
    >> scatter: scatter plot of 2 parameters, masked by the vx_mask
    >> plot_ts: plot the time series of a voxel
    >> make_prf_str: make a string of the parameters for a voxel
    >> make_context_str: make a string of the task, model, and voxel index
    >> rsq_w_mean: calculate the weighted mean of a parameter, weighted by rsq

    '''
    def __init__(self, prf_params, model, **kwargs):
        '''__init__
        Input:
        ----------
        prf_params     np.ndarray, of all the parameters, i.e., output from prfpy        
        model          str, model: e.g., gauss or norm
        Optional:
        incl_hrf        bool, if True, then the hrf parameters are included
        incl_rsq        bool, if True, then the rsq is included
        task            str, task name

        '''
        self.model = model        
        self.model_labels = prfpy_params_dict()[self.model] # Get names for different model parameters...
        self.prf_params_np = prf_params.copy()
        self.saved_kwargs = kwargs
        self.incl_hrf = kwargs.get('incl_hrf', True)
        self.incl_rsq = kwargs.get('incl_rsq', True)
        print(f'prf_params.shape[-1]={prf_params.shape[-1]}')
        print(f'include hrf = {self.incl_hrf}')
        print(f'include rsq = {self.incl_rsq}')
        #
        self.task = kwargs.get('task', None)
        self.n_vox = self.prf_params_np.shape[0]

        self.params_dd = {}
        mod_labels = prfpy_params_dict()[f'{model}'] 
        for key in mod_labels.keys():
            if ('hrf' in key) and not self.incl_hrf:
                continue
            if ('rsq' in key) and not self.incl_rsq:
                continue                    
            self.params_dd[key] = self.prf_params_np[:,mod_labels[key]]
        
        # Calculate extra interesting stuff
        if self.model in ['gauss', 'norm', 'css', 'dog']:
            # Ecc, pol
            self.params_dd['ecc'], self.params_dd['pol'] = dpu_coord_convert(
                self.params_dd['mu_x'],self.params_dd['mu_y'],'cart2pol')      
            # angles like a clock
            self.params_dd['clock'] = dpu_pol_to_clock(self.params_dd['pol'])

        if self.model in ('norm', 'dog'):
            # -> size ratio:
            self.params_dd['size_ratio'] = self.params_dd['srf_size'] / self.params_dd['prf_size']
            self.params_dd['amp_ratio'] = self.params_dd['srf_amplitude'] / self.params_dd['prf_amplitude']
        if self.model == 'norm':
            self.params_dd['bd_ratio'] = self.params_dd['neural_baseline'] / self.params_dd['surround_baseline']
            # Suppression index 
            self.params_dd['sup_idx'] = (self.params_dd['prf_amplitude'] * self.params_dd['prf_size']**2) / (self.params_dd['srf_amplitude'] * self.params_dd['srf_size']**2)
        if self.model=='csf':
            self.params_dd['log10_SFp'] = np.log10(self.params_dd['SFp'])
            self.params_dd['log10_CSp'] = np.log10(self.params_dd['CSp'])
            self.params_dd['log10_crf_exp'] = np.log10(self.params_dd['crf_exp'])
            self.params_dd['sfmax'] = ncsf_calculate_sfmax(
                self.params_dd['width_r'],
                self.params_dd['SFp'],
                self.params_dd['CSp'],
            )
            self.params_dd['log10_sfmax'] = np.log10(self.params_dd['sfmax'])
            self.params_dd['aulcsf'] = ncsf_calculate_aulcsf(
                self.params_dd['width_r'],
                self.params_dd['SFp'],
                self.params_dd['CSp'],    
                self.params_dd['width_l'],            
            )

        # Convert to PD           
        self.pd_params = pd.DataFrame(self.params_dd)

    def return_vx_mask(self, th={}):
        '''return_vx_mask
        Returns a mask (boolean array) for voxels
        
        Notes: 
        ----------
        th keys must be split into 2 parts
        'comparison-param' : value
        e.g.: to exclude gauss fits with rsq less than 0.1
        th = {'min-rsq': 0.1 } 
        comparison  -> min, max,bound
        param       -> any of... (model dependent, see prfpy_params_dict)
        value       -> float, or tuple of floats (for bounds)

        A special case is applied for roi, which is a boolean array you specified previously
        

        Input:
        ----------
        th          dict, threshold for parameters

        Output:
        ----------
        vx_mask     np.ndarray, boolean array, length = n_vx
        
        '''        

        # Start with EVRYTHING         
        vx_mask = np.ones(self.n_vox, dtype=bool) 
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str: # Input roi specification...                
                vx_mask &= th[th_key]
                continue # now next item in key
            if 'idx'==th_key_str:
                # Input voxel index specification...
                idx_mask = np.zeros(self.n_vox, dtype=bool)
                idx_mask[th[th_key]] = True
                vx_mask &= idx_mask
                continue

            comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if comp=='min':
                vx_mask &= self.pd_params[p].gt(th_val)
            elif comp=='max':
                vx_mask &= self.pd_params[p].lt(th_val)
            elif comp=='bound':
                vx_mask &= self.pd_params[p].gt(th_val[0])
                vx_mask &= self.pd_params[p].lt(th_val[1])
            elif comp=='eq':
                vx_mask &= self.pd_params[p].eq(th_val)
            
            else:
                print(f'Error, {comp} is not any of min, max, or bound')
                sys.exit()
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()

        return vx_mask
    
    def return_th_params(self, px_list=None, th={}, **kwargs):
        '''return_th_param
        return all the parameters listed, masked by vx_mask        
        '''
        if px_list is None:
            px_list = list(self.pd_params.keys())
        elif not isinstance(px_list, list):
            px_list = [px_list]
                
        # relevant mask 
        vx_mask = self.return_vx_mask(th)
        # create tmp dict with relevant stuff...
        tmp_dict = {}
        for i_px in px_list:
            tmp_dict[i_px] = self.pd_params[i_px][vx_mask].to_numpy()
        return tmp_dict    
        
    def hist(self, param, th={'min-rsq':.1}, ax=None, **kwargs):
        '''hist: Plot a histogram of a parameter, masked by th'''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)        
        ax.hist(self.pd_params[param][vx_mask].to_numpy(), **kwargs)
        ax.set_title(param)
        dpu_add_ax_basics(ax=ax, **kwargs)

    def visual_field(self, th={'min-rsq':.1, 'max-ecc':5}, ax=None, dot_col='k', **kwargs):
        '''visual_field
        Plot the visual field of the voxels, masked by the vx_mask
        and colored by a parameter

        Notes:
        ----------
        Default vx mask is all voxels with rsq > 0.1 and ecc < 5

        Input:
        ----------
        Optional:
        th          dict, threshold for parameters
        ax          matplotlib.axes, if None, then plt.axes() is used
        dot_col     str, color of the dots
        kwargs      dict, kwargs for dpu_visual_field_scatter
        '''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        if isinstance(dot_col,str):
            if dot_col in self.pd_params.keys():
                dot_col = self.pd_params[dot_col][vx_mask].to_numpy()

        dpu_visual_field_scatter(
            ax=ax, 
            dot_x=self.pd_params['mu_x'][vx_mask].to_numpy(),
            dot_y=self.pd_params['mu_y'][vx_mask].to_numpy(),
            dot_col = dot_col,
            **kwargs
        )        

    def scatter(self, px, py, th={'min-rsq':.1}, ax=None, **kwargs):
        '''scatter
        Scatter plot of 2 parameters, masked by the vx_mask
        Can also color by a third parameter

        Notes:
        ----------
        Default vx mask is all voxels with rsq > 0.1

        Input:
        ----------
        px          str, parameter to plot on x axis
        py          str, parameter to plot on y axis
        Optional:
        th          dict, threshold for parameters
        ax          matplotlib.axes, if None, then plt.axes() is used
        dot_col     str, color of the dots
        dot_alpha   float, alpha of the dots
        kwargs      dict, kwargs for dpu_scatter

        '''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        pc = kwargs.get('pc', None)        
        if pc is not None:
            kwargs['dot_col'] = self.pd_params[pc][vx_mask]
        dpu_scatter(
            ax=ax,
            X=self.pd_params[px][vx_mask].to_numpy(),
            Y=self.pd_params[py][vx_mask].to_numpy(),
            **kwargs
        )    
        ax.set_xlabel(px)
        ax.set_ylabel(py)
            

    def make_prf_str(self, idx, pid_list=None):
        '''make_prf_str
        Make a string of the parameters for a voxel

        Input:
        ----------
        idx         int, which voxel to plot

        Output:
        ----------
        prf_str     str, string of the parameters for a voxel
        '''
        prf_str = f'vx_id={idx},\n '
        param_count = 0
        if pid_list is None:
            pid_list = self.model_labels
        for param_key in pid_list:
            if param_key in self.pd_params.keys():
                param_count += 1
                prf_str += f'{param_key}= {self.pd_params[param_key][idx]:8.2f};\n '
        return prf_str
    
    def rsq_w_mean(self, pid_list, th={'min-rsq':.1}):
        '''rsq_w_mean
        Calculate the weighted mean of a parameter, weighted by rsq

        Input:
        ----------
        pid_list    list, parameter id list
        Optional:
        th          dict, threshold for parameters

        Output:
        ----------
        wm_param    dict, weighted mean of the parameters
        '''
        if not isinstance(pid_list, list):
            pid_list = [pid_list]

        vx_mask = self.return_vx_mask(th)
        wm_param = {}
        for i_param in pid_list:
            wm_param[i_param] = dpu_weighted_mean(
                w=self.pd_params['rsq'][vx_mask].copy(),
                x=self.pd_params[i_param][vx_mask].copy(),
            )

        self.wm_param = wm_param
        return wm_param
    
    def multi_scatter(self, px_list, th={'min-rsq':.1}, **kwargs):
        '''multi_scatter
        Several scatter plots... multiple comparisons...
        i.e., creates a grid of scatter plots
        '''
        tmp_dict = self.return_th_params(px_list, th, **kwargs)
        fig, ax_list = dpu_multi_scatter(tmp_dict, **kwargs)            
        return fig, ax_list





class PrfMulti(object):
    '''PrfMulti
    Class for parsing prfpy output for multiple models/tasks for the *same subject*
    
    Notes:
    ----------
    It is important that there are the same number of voxels in each model/task    
    Create a list of Prf1T1M objects, and associated labels, which are all collected in
    this class.     
    It will hold all of the original Prf1T1M objects inside a dictionary    
    The idea is that it makes it easier to do comparisons across conditions/models

    Functions:
    ----------
    Data processing:
    return_vx_mask: returns a mask (boolean array) for voxels
    return_th_param: returns the specified parameters, masked by the vx_mask
    add_prf_diff: add a difference between 2 prf_obj (e.g., diff between 2 tasks)
    TODO: add_prf_mean: add a mean between 2 prf_obj (e.g., mean between 2 tasks)

    ** Plot functions **:
    hist: plot a histogram of a parameter
    scatter: scatter plot of 2 parameters
    multi_scatter: Several scatter plots... multiple comparisons...
    arrow: plot an arrow between 2 prf_obj

    TODO: ? visual_field: plot voxels around the visual field of the voxels, masked by the vx_mask
        and colored by a parameter
    '''
    def __init__(self,prf_obj_list, id_list):
        '''__init__
        
        Input:
        ----------
        prf_obj_list    list, of Prf1T1M objects
        id_list         list, of strings, to label the prf_obj_list        
        '''
        self.id_list = id_list.copy()
        self.prf_obj = {}
        self.n_vox = prf_obj_list[0].n_vox
        for i,this_id in enumerate(id_list):
            self.prf_obj[this_id] = deepcopy(prf_obj_list[i])
        total_dict = {}
        for this_id in id_list:
            for p in self.prf_obj[this_id].pd_params.keys():
                total_dict[f'{this_id}-{p}'] = self.prf_obj[this_id].pd_params[p].to_numpy()
        self.pd_params = pd.DataFrame(total_dict)

    def return_vx_mask(self, th={}):
        '''return_vx_mask
        Returns a mask (boolean array) for voxels
        
        Notes: 
        ----------
        As in Prf1T1M, but with one extra part of the key:        
        th keys must be split into 3 parts
        'id-comparison-param' : value
        th = {'prf1-min-rsq': 0.1 } 
        id          -> which prf_obj to apply the threshold to
                        Can also be 'all', which applies to all prf_obj
        comparison  -> min, max,bound
        param       -> any of... (model dependent, see prfpy_params_dict)
        value       -> float, or tuple of floats (for bounds)

        A special case is applied for roi, which is a boolean array you specified previously
        

        Input:
        ----------
        th          dict, threshold for parameters

        Output:
        ----------
        vx_mask     np.ndarray, boolean array, length = n_vx
                Returns a mask (boolean array) for voxels
        
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        if th is None:
            return vx_mask
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key
            if 'idx'==th_key_str:
                # Input voxel index specification...
                idx_mask = np.zeros(self.n_vox, dtype=bool)
                idx_mask[th[th_key]] = True
                vx_mask &= idx_mask
                continue            
            # print(th)
            id, comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if id=='all':
                # Apply to both task1 and task2:                
                for prf_id in self.id_list:
                    if 'diff_' in prf_id: # skip the diff ones...
                        print('not applying threshold to diff')
                        continue

                    p_available = list(self.prf_obj[prf_id].pd_params.keys())
                    if p in p_available:
                        vx_mask &= self.prf_obj[prf_id].return_vx_mask({f'{comp}-{p}':th_val})
                    else:
                        print(f'Warning - {p} is not a paramer for {prf_id}, ignoring...')
                continue # now next item in th_key...
            vx_mask &= self.prf_obj[id].return_vx_mask({f'{comp}-{p}':th_val})

        if not isinstance(vx_mask, np.ndarray):
            vx_mask = vx_mask.to_numpy()
        return vx_mask
    
    def return_th_params(self, px_list=None, th=None, **kwargs):
        '''return_th_param
        return all the parameters listed, masked by vx_mask        
        '''
        if px_list is None:
            px_list = list(self.pd_params.keys())
        elif not isinstance(px_list, list):
            px_list = [px_list]
        px_id = [None] * len(px_list)
        px_p = [None] * len(px_list)
        for i,p in enumerate(px_list):
            px_id[i], px_p[i] = p.split('-')
                
        if th==None:
            min_rsq = kwargs.get('min_rsq', .1)
            th = {}
            for key in list(set(px_id)):
                th[f'{key}-min-rsq'] = min_rsq
        # add extra th from the default
        th_plus = kwargs.get('th_plus', {})
        th = {**th, **th_plus}
        # relevant mask 
        vx_mask = self.return_vx_mask(th)
        # create tmp dict with relevant stuff...
        tmp_dict = {}
        for i_px_id,i_px_p in zip(px_id, px_p):
            tmp_dict[f'{i_px_id}-{i_px_p}'] = self.prf_obj[i_px_id].pd_params[i_px_p][vx_mask].to_numpy()
        return tmp_dict
    
    def return_diff_params(self, id1, id2, p_list, **kwargs):
        ''' 
        Return the difference of 2 prf objects 
        (rather than creating a whole "prf_diff" object)

        id1         str, id of the first prf_obj
        id2         str, id of the second prf_obj
        px_list     list of parameters to take the difference of
        '''         
        if not isinstance(p_list, list):
            p_list = [p_list]
        # create tmp dict with relevant stuff...        
        tmp_dict = {}
        for p in p_list:
            # special case for 'shift_mag' and 'shift_dir'
            if p in ['shift_mag', 'shift_dir']:
                dx = self.pd_params[f'{id1}-x'] - self.pd_params[f'{id2}-x']
                dy = self.pd_params[f'{id1}-y'] - self.pd_params[f'{id2}-y']
                shift_dict = {}
                shift_dict['shift_mag'], shift_dict['shift_dir'] = dpu_coord_convert(
                    dx, dy, 'cart2pol'
                )
                tmp_dict[p] = shift_dict[p].copy()
            else:
                tmp_dict[p] = self.pd_params[f'{id1}-{p}'] - self.pd_params[f'{id2}-{p}']
        return tmp_dict


    def add_prf(self, new_prf, new_id, ow=True):
        '''add_prf_obj
        Add a new prf_obj to the list
        '''
        if new_id in self.id_list:
            print(f'{new_id} already exists')
            if not ow:
                print('Not overwriting...')
                return
        
        else:
            self.id_list += [new_id]

        self.prf_obj[new_id] = new_prf
        for p in  new_prf.pd_params.keys():
            self.pd_params[f'{new_id}-{p}'] = new_prf.pd_params[p].to_numpy()



    def add_prf_diff(self, id1, id2, new_id=None):
        '''add_prf_diff
        Add a difference between 2 prf_obj (e.g., diff between 2 tasks)

        Input:
        ----------
        id1         str, id of the first prf_obj
        id2         str, id of the second prf_obj
        Optional:
        new_id      str, id of the new prf_obj
        '''
        if new_id is None:
            new_id = f'diff_{id1}_{id2}'
        if new_id in self.id_list:
            print(f'Already created {new_id}')
        else:
            self.prf_obj[new_id] = PrfDiff(
                self.prf_obj[id1], self.prf_obj[id2], diff_id=new_id,
            )
            self.id_list += [new_id]
    # TODO - add_prf_mean?    
    
    # ***************** OBJECT PLOT FUNCTIONS ***************** # 
    def hist(self, px, th=None, ax=None, **kwargs):
        '''hist: Plot a histogram of a parameter, masked by th'''
        if ax==None:
            ax = plt.axes()
        px_id, px_p = px.split('-')                
        if th==None:
            th = {f'{px_id}-min-rsq':.1}            
        vx_mask = self.return_vx_mask(th)        
        label = kwargs.get('label', f'{px_id}-{px_p}')
        kwargs['label'] = label
        ax.hist(self.prf_obj[px_id].pd_params[px_p][vx_mask].to_numpy(), **kwargs)
        ax.set_title(f'{px_id}-{px_p}')
        dpu_add_ax_basics(ax=ax, **kwargs)

    def scatter(self, px, py, th=None, ax=None, **kwargs):
        '''scatter: As in Prf1T1M, but can also specify across different prf_obj'''
        # dot_col = kwargs.get('dot_col', 'k')
        # dot_alpha = kwargs.get('dot_alpha', None)
        if ax==None:
            ax = plt.axes()
        px_id, px_p = px.split('-')
        py_id, py_p = py.split('-')
        pc = kwargs.get('pc', None) # dot_color
        if pc is not None:
            pc_id, pc_p = pc.split('-')


        if th==None:
            if 'diff' in (px_id, py_id):
                print('bloop')             
            min_rsq = kwargs.get('min_rsq', .1)
            th = {
                f'{px_id}-min-rsq':min_rsq,
                f'{py_id}-min-rsq':min_rsq,
            }
            if pc is not None:
                th[f'{pc_id}-min-rsq'] = min_rsq

        th_plus = kwargs.get('th_plus', None)
        if not th_plus is None:
            th = {**th, **th_plus}
        vx_mask = self.return_vx_mask(th)
        if pc is not None:
            kwargs['dot_col'] = self.prf_obj[pc_id].pd_params[pc_p][vx_mask]
        if vx_mask.sum()==0:
            print('Warning: no voxels found')
            return
        dpu_scatter(
            ax=ax,
            X=self.prf_obj[px_id].pd_params[px_p][vx_mask].to_numpy(),
            Y=self.prf_obj[py_id].pd_params[py_p][vx_mask].to_numpy(),
            **kwargs
        )              
        ax.set_xlabel(px)        
        ax.set_ylabel(py)                        

    def multi_scatter(self, px_list, th=None, ax=None, **kwargs):
        '''multi_scatter
        Several scatter plots... multiple comparisons...
        i.e., creates a grid of scatter plots
        '''
        tmp_dict = self.return_th_params(px_list=px_list, th=th, **kwargs)
        fig, ax_list = dpu_multi_scatter(tmp_dict, **kwargs)            
        return fig, ax_list
    
    def arrow(self, pold, pnew, ax=None, th=None, **kwargs):
        '''arrow: arrows from one prf_obj to another'''
        if ax==None:
            ax = plt.gca()
        if th is None:
            th = {
                f'{pold}-min-rsq':kwargs.get('min_rsq', 0.1),
                f'{pold}-max-ecc':kwargs.get('max_ecc', 5),
                f'{pnew}-min-rsq':kwargs.get('min_rsq', 0.1),
                f'{pnew}-max-ecc':kwargs.get('max_ecc', 5),
                }
        th_plus = kwargs.get('th_plus', {})
        th = dict(**th, **th_plus)        
            
        vx_mask = self.return_vx_mask(th)        
        kwargs['title'] = kwargs.get('title', f'{pold}-{pnew}')

        arrow_out = dpu_arrow_plot(
            ax, 
            old_x=self.prf_obj[pold].pd_params['mu_x'][vx_mask], 
            old_y=self.prf_obj[pold].pd_params['mu_y'][vx_mask], 
            new_x=self.prf_obj[pnew].pd_params['mu_x'][vx_mask], 
            new_y=self.prf_obj[pnew].pd_params['mu_y'][vx_mask], 
            # arrow_col='angle', 
            **kwargs
            )
        return arrow_out 
    
    def visual_field(self, vf_obj, col_obj_p, th=None, **kwargs):
        '''Visual field scatter
        As with Prf1T1M -> but specify which object has the x,y, coordinates (vf_obj)
        And specify the object for color, and the parameter
        e.g., 
        prf_multi.visual_field(
            vf_obj = 'gauss_obj',
            col_obj_p = 'csf_obj-SFp'
        )
        '''
        col_obj, col_p = col_obj_p.split('-')
        if th is None:
            min_rsq = kwargs.get('min_rsq', 0.1)
            max_ecc = kwargs.get('max_ecc', 5)
            th = {
                f'{vf_obj}-min-rsq':min_rsq,
                f'{vf_obj}-max-ecc': max_ecc,
                f'{col_obj}-min-rsq':min_rsq,
                }        
        th_plus = kwargs.get('th_plus', {})
        th = dict(**th, **th_plus)        
        kwargs['title'] = kwargs.get('title', f'vf={vf_obj}: col={col_obj_p}')            
        vx_mask = self.return_vx_mask(th)        
        rsq_weight = kwargs.get('rsq_weight', False) 
        if rsq_weight:
            kwargs['bin_weight'] = self.prf_obj[col_obj].pd_params['rsq'][vx_mask]
        
        for p in ['dot_size', 'dot_alpha']:
            if p not in kwargs.keys():
                continue
            if isinstance(kwargs[p], str):
                # bloop
                kwargs[p] = self.pd_params[kwargs[p]][vx_mask]

        dpu_visual_field_scatter(
            dot_x   = self.prf_obj[vf_obj].pd_params['mu_x'][vx_mask],
            dot_y   = self.prf_obj[vf_obj].pd_params['mu_y'][vx_mask],
            dot_col = self.prf_obj[col_obj].pd_params[col_p][vx_mask],
            **kwargs
        )           




class PrfDiff(object):
    '''PrfDiff
    Used with PrfMulti, to contrast 2 conditions
    '''
    def __init__(self, prf_obj1, prf_obj2, diff_id, **kwargs):
        assert ('diff' in diff_id), 'Needs a diff'
        # if not 'diff_' in id:
        #     print('needs a diff_!')  

        # self.id = id
        self.model_labels1 = list(prf_obj1.pd_params.keys())
        self.model_labels2 = list(prf_obj2.pd_params.keys())
        self.n_vox = prf_obj1.n_vox 
        self.pd_params = {}
        
        # Make mean and difference:
        for i_label in self.model_labels1:
            if i_label not in self.model_labels2:
                continue
            self.pd_params[i_label] = prf_obj1.pd_params[i_label] -  prf_obj2.pd_params[i_label]
        # For the position shift, find the direction and magnitude:
        if ('mu_x' in self.model_labels1) and ('mu_x' in self.model_labels2):
            self.pd_params['shift_mag'], self.pd_params['shift_dir'] = dpu_coord_convert(
                self.pd_params['mu_x'], self.pd_params['mu_y'], 'cart2pol'
            )        
        # some stuff needs to be recalculated?: (because they don't scale linearly...?
        self.pd_params = pd.DataFrame(self.pd_params)

    def return_vx_mask(self, th={}):
        '''
        ... as before ...
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key

            comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if comp=='min':
                vx_mask &= self.pd_params[p].gt(th_val)
            elif comp=='max':
                vx_mask &= self.pd_params[p].lt(th_val)
            elif comp=='bound':
                vx_mask &= self.pd_params[p].gt(th_val[0])
                vx_mask &= self.pd_params[p].lt(th_val[1])
            elif comp=='eq':
                vx_mask &= self.pd_params[p].eq(th_val)
            else:
                sys.exit()
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()

        return vx_mask


class PrfMean(object):
    def __init__(self, prf_obj1, prf_obj2, id):
        # self.id = id
        if not 'mean_' in id:
            print('needs a mean_!')
        self.model_labels1 = list(prf_obj1.pd_params.keys())
        self.model_labels2 = list(prf_obj2.pd_params.keys())
        self.n_vox = prf_obj1.n_vox 
        self.pd_params = {}
        
        # Make mean and difference:
        for i_label in self.model_labels1:
            if i_label not in self.model_labels2:
                continue
            self.pd_params[i_label] = (self.pd_params[self.id2][i_label] +  self.pd_params[self.id1][i_label]) / 2
        # some stuff needs to be recalculated?: (because they don't scale linearly...?
        self.pd_params = pd.DataFrame(self.pd_params)

    def return_vx_mask(self, th={}):
        '''
        ... as before ...
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key

            comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if comp=='min':
                vx_mask &= self.pd_params[p].gt(th_val)
            elif comp=='max':
                vx_mask &= self.pd_params[p].lt(th_val)
            elif comp=='bound':
                vx_mask &= self.pd_params[p].gt(th_val[0])
                vx_mask &= self.pd_params[p].lt(th_val[1])
            elif comp=='eq':
                vx_mask &= self.pd_params[p].eq(th_val)
            else:
                sys.exit()
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()

        return vx_mask