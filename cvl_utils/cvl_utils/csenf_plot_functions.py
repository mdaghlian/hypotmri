import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import patches
from scipy.stats import binned_statistic
import pandas as pd
import json
import os
opj = os.path.join

from prfpy.model import *
from prfpy.rf import *
from prfpy.stimulus import *


"""
TEMPORARY PLACEMENT OF OLD CODE 
RELATED TO PLOTTING CF FIT WITH PRFPY
"""

# csenf_plot_functions.py
# additional functions for plotting and post-processing of nCSF data

# Example parameters 
Chung_Legge_default = { # From Chung & Legge, 5 healthy controls... Used to normalize AUC
    'width_r' : 1.28,
    'SFp'     : 2.5, # c/deg
    'CSp'     : 166,
    'width_l' : 0.68, 
}

# For example... CRF slope / exponent / transducer (many names in literature )
eg_CRF = {'crf_exp' : 0.8} 
# -> generally about 2 in animals
# -> lower in fMRI... "The contrast–response exponents estimated from our fMRI measurements are significantly smaller than those measured for single cells in the primary visual cortices of both cats and primates. In animals, the exponent is 2, on average, but there is variability from cell to cell (Albrecht and Hamilton, 1982; Sclar et al., 1990)." Boynton et al. • Linear Systems Analysis of fMRI in Human V1 J. Neurosci., July 1, 1996, 16(13):4207– 4221 421



def ncsf_calculate_sfmax(width_r, SFp, CSp, max_sfmax=50):
    """calculate_sfmax    
    aka high frequency cutoff. Useful summary statistic of whole CSF curve
    set the sensitivity = 1, then solve for the corresponding SF. 
    i.e., what is the highest possible SF we can detect    
    Can be infinte (with low width_r), so we set a max value 
    """
    log10_CSp = np.log10(CSp)
    log10_SFp = np.log10(SFp)
    sfmax = 10**((np.sqrt(log10_CSp/(width_r**2)) + log10_SFp))
    if len(sfmax.shape)>=1:
        sfmax[sfmax>max_sfmax] = max_sfmax
    elif sfmax>max_sfmax:
        sfmax = max_sfmax        
    return sfmax

def ncsf_calculate_aulcsf(width_r, SFp, CSp, width_l, **kwargs):
    """calculate_aulcsf
    area under log contrast sensitivity function
    Another useful summary statistic. Taken by integrating the log CSF curve
    

    Parameters:
    width_r, SFp, CSp, width_l,         Parameters to determine the curve

    Optional 
    normalize_AUC       Normlalize AUC with respect to Chung_Legge version. Default = True
    SF_levels       which SFs are we using to generate points?
        We use trapezium method to approximate the integral
        Default is set to [ 0.5,  1.,   3.,   6.,  12.,  18. ] the SF levels we sample

    """
    normalize_AUC = kwargs.get('normalize_AUC', True)
    SF_levels = kwargs.get('SF_levels', np.array([ 0.5,  1.,   
    3.,   6.,  12.,  18. ]))
    log_SF_levels = np.log10(SF_levels)#.reshape(1,-1)
    # Generate grid to make the CSF     
    csf_curve = asymmetric_parabolic_CSF(
        SF_seq = SF_levels, 
        width_r = width_r, 
        SFp = SFp, 
        CSp = CSp, 
        width_l = width_l, 
        )
    logcsf_curve = np.log10(csf_curve)    
    logcsf_curve[logcsf_curve<0] = 0 # Cannot have negative logCSF
    aulcsf = np.trapezoid(logcsf_curve.T, x=log_SF_levels, axis=0) 
    if not normalize_AUC:
        return aulcsf
    # Chung & Legge, normalized 
    csf_curve = asymmetric_parabolic_CSF(
        SF_seq = SF_levels, 
        width_r     = Chung_Legge_default['width_r'], 
        SFp         = Chung_Legge_default['SFp'], 
        CSp         = Chung_Legge_default['CSp'], 
        width_l     = Chung_Legge_default['width_l'], 
        )
    logcsf_curve = np.log10(csf_curve)    
    logcsf_curve[logcsf_curve<0] = 0 # Cannot have negative logCSF    
    CL_aulcsf = np.trapezoid(logcsf_curve.T, x=log_SF_levels, axis=0) 
    norm_aulcsf = 100 * aulcsf / CL_aulcsf
    return norm_aulcsf

def ncsf_calculate_crf_curve(crf_exp, Q=20, C=np.linspace(0,100,100), **kwargs):
    '''ncsf_calculate_crf_curve
    To calculate the CRF derived curve 
    '''
    crf_curve = nCSF_apply_crf(
        csenf_values=Q, 
        crf_exp=crf_exp,
        CON_seq=C, 
        **kwargs
    )


    return crf_curve

# ********** PRF OBJECTS
class CSenFPlotter(object):
    '''CSenfPlotter
    For use with prfpy_csenf. 

    Includes useful functions for plotting and analysis:    
    >> return_vx_mask: returns a mask for voxels
    >> return_th_param: returns the specified parameters, masked by the vx_mask
    >> hist: plot a histogram of a parameter, masked by the vx_mask
    >> scatter: scatter plot of 2 parameters, masked by the vx_mask
    >> make_nCSF_str: make a string of the parameters for a voxel
    >> make_context_str: make a string of the task, model, and voxel index
    >> rsq_w_mean: calculate the weighted mean of a parameter, weighted by rsq

    '''
    def __init__(self, prf_params, **kwargs):
        '''__init__
        Input:
        ----------
        
        prf_params     np.ndarray, of all the parameters
        
        Optional:
        prfpy_model    prfpy model for generating TS 
        real_ts        np.ndarray of "true timeseries" i.e., the data


        '''
        self.model_labels = {
            'width_r'       : 0,
            'SFp'           : 1,
            'CSp'          : 2,
            'width_l'       : 3,
            'crf_exp'       : 4,
            'amp_1'         : 5,
            'bold_baseline' : 6,
            'hrf_1'         : 7,
            'hrf_2'         : 8,
            'rsq'           : -1,            
        }
        self.incl_hrf = kwargs.get('incl_hrf', None)
        self.incl_rsq = kwargs.get('incl_rsq', None)
        # Organize the parameters. If they are an numpy array or a dictionary
        if isinstance(prf_params, dict):
            self.params_dict_to_np(params_dict=prf_params)
        else:
            # Assume its a list or a numpy array
            self.params_np_to_dict(params_np=prf_params)
        #
        self.real_ts = kwargs.get('real_ts', None)
        self.prfpy_model = kwargs.get('prfpy_model', None)
        self.TR_in_s = kwargs.get('TR_in_s', 1 )          
        self.edge_type = kwargs.get('edge_type', 'CRF')
        if self.prfpy_model is not None:
            self.edge_type = self.prfpy_model.edge_type
            self.prfpy_stim = self.prfpy_model.stimulus
            self.TR_in_s = self.prfpy_stim.TR
        
        # SF list... cmap etc.
        self._sort_SF_list(**kwargs)

        self.params_dd = {}
        for key in self.model_labels.keys():
            if ('hrf' in key) and not self.incl_hrf:
                continue
            if ('rsq' in key) and not self.incl_rsq:
                continue                    
            self.params_dd[key] = self.prf_params_np[:,self.model_labels[key]]
        
        # Calculate extra interesting stuff
        self.params_dd['log10_SFp'] = np.log10(self.params_dd['SFp'])
        self.params_dd['log10_CSp'] = np.log10(self.params_dd['CSp'])
        self.params_dd['log10_crf_exp'] = np.log10(self.params_dd['crf_exp'])
        self.params_dd['sfmax'] = ncsf_calculate_sfmax(
            width_r = self.params_dd['width_r'],
            SFp = self.params_dd['SFp'],
            CSp = self.params_dd['CSp'],
        )
        self.params_dd['log10_sfmax'] = np.log10(self.params_dd['sfmax'])

        self.params_dd['raw_aulcsf'] = ncsf_calculate_aulcsf(
            width_r = self.params_dd['width_r'],
            SFp = self.params_dd['SFp'],
            CSp = self.params_dd['CSp'],
            width_l = self.params_dd['width_l'],
            normalize_AUC=False,
        )
        self.params_dd['aulcsf'] = ncsf_calculate_aulcsf(
            width_r = self.params_dd['width_r'],
            SFp = self.params_dd['SFp'],
            CSp = self.params_dd['CSp'],
            width_l = self.params_dd['width_l'],
            normalize_AUC=True,
        )        

        # Convert to PD           
        self.pd_params = pd.DataFrame(self.params_dd)

    def _sort_SF_list(self, **kwargs):
        self.SF_list = kwargs.get('SF_list', None)
        self.SF_cmap_name = kwargs.get('SF_cmap', 'viridis')
        if self.SF_list is None:
            self.SF_list = np.array([ 0.5,  1.,  3.,   6.,  12.,  18. ])
            if self.prfpy_model is not None:
                if self.prfpy_model.stimulus.discrete_levels:
                    self.SF_list = self.prfpy_model.stimulus.SFs        
        self.SF_cmap = mpl.cm.__dict__[self.SF_cmap_name]
        self.SF_cnorm = mpl.colors.Normalize()
        self.SF_cnorm.vmin, self.SF_cnorm.vmax = self.SF_list[0],self.SF_list[-1] # *1.5 
        self.SF_cols = {}
        for iSF, vSF in enumerate(self.SF_list):
            this_SF_col = self.SF_cmap(self.SF_cnorm(vSF))
            self.SF_cols[vSF] = this_SF_col        

    def _get_SF_cols(self, v):
        closest_key = None
        min_difference = float('inf')  # Initialize with infinity

        for key in self.SF_cols.keys():
            difference = abs(key - v)
            if difference <= 0.1 and difference < min_difference:
                min_difference = difference
                closest_key = key
        if closest_key is not None:
            this_col = self.SF_cols[closest_key]
        else:
            this_col = None
        return this_col

    def _add_SF_colorbar(self, ax):        
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=self.SF_cnorm, cmap=self.SF_cmap_name), ax=ax)
        cbar.set_label('SF')


    def params_dict_to_np(self, params_dict):
        '''
        '''
        if not isinstance(params_dict['width_r'], np.ndarray):
            for key in params_dict.keys():
                params_dict[key] = np.atleast_1d(np.array(params_dict[key]))#.squeeze()

        # if len(params_dict['width_r'].shape)==1:
        #     for key in params_dict.keys():
        #         params_dict[key] = params_dict[key].reshape(-1,1)
        
        self.n_vox = params_dict['width_r'].shape[0]                 
        if ('rsq' not in params_dict.keys()) and (self.incl_rsq is None):
            self.incl_rsq=False
        elif self.incl_rsq is None:            
            self.incl_rsq=True

        if ('hrf_1' not in params_dict.keys()) and (self.incl_hrf is None):
            self.incl_hrf=False
        elif self.incl_hrf is None:
            self.incl_hrf=True
        
        self.prf_params_np = np.zeros((self.n_vox, 7 + self.incl_rsq + 2*self.incl_hrf))
        for p in self.model_labels:
            if (p=='rsq') and (not self.incl_rsq):
                continue
            if ('hrf' in p) and (not self.incl_hrf):
                continue

            self.prf_params_np[:,self.model_labels[p]] = params_dict[p]#.squeeze()


    def params_np_to_dict(self, params_np, **kwargs):
        '''
        '''
        self.prf_params_np = params_np
        if not isinstance(self.prf_params_np, np.ndarray):            
            self.prf_params_np = np.array(self.prf_params_np)

        if len(self.prf_params_np.shape)==1:
            self.prf_params_np = self.prf_params_np.reshape(-1,1)

        # Sometimes parameters include hrfs, rsqs... 
        # Sometimes not
        print(f'prf_params.shape[-1]={params_np.shape[-1]}')
        if params_np.shape[-1]==7: # params 0-6
            hrf_message = 'hrf=default'
            rsq_message = 'rsq=not calculated'

            incl_hrf = False
            incl_rsq = False
        elif params_np.shape[-1]==8: # params 0-6 + rsq 
            hrf_message = 'hrf=default'
            rsq_message = 'rsq=params[:,7]'
            incl_hrf = False
            incl_rsq = True            
        elif params_np.shape[-1]==9: # params 0-6 + hrf_1, hrf_2
            hrf_message = 'hrf=params[:,7,8]'
            rsq_message = 'rsq=not calculated'
            incl_hrf = True
            incl_rsq = False
        elif params_np.shape[-1]==10: # Params 0-6 + hrf_1, hrf_2, rsq
            hrf_message = 'hrf=params[:,7,8]'
            rsq_message = 'rsq=params[:,9]'
            incl_hrf = True
            incl_rsq = True
        
        # Only overwrite if not already specified
        if self.incl_hrf is None:
            print(hrf_message)
            self.incl_hrf = incl_hrf
        if self.incl_rsq is None:
            print(rsq_message)
            self.incl_rsq = incl_rsq
        self.n_vox = self.prf_params_np.shape[0]

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
            else:
                print(f'Error, {comp} is not any of min, max, or bound')
                return 
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()

        return vx_mask
    
    def return_th_params(self, px_list=None, th={}, **kwargs):
        '''return_th_param
        return all the parameters listed, masked by vx_mask        
        '''
        if px_list is None:
            px_list = list(self.pd_params.keys())
                
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
        ax.set_title(param)
        ax.hist(self.pd_params[param][vx_mask].to_numpy(), **kwargs)

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

        # dot_col = kwargs.get('dot_col', 'k')
        # dot_alpha = kwargs.get('dot_alpha', None)
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        ax.set_ylabel(py)    
        ax.set_xlabel(px)
        ax.scatter(
            self.pd_params[px][vx_mask],
            self.pd_params[py][vx_mask],
            **kwargs
        )


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
    
    def csf_ts_plot_get_info(self, idx):
        '''Calculate various stuff used when plotting the CSF
        '''
        csenf_stim = self.prfpy_model.stimulus
        ncsf_info = {}
        for key in self.pd_params.keys():
            # if not isinstance(ncsf_info[key], (list, np.ndarray)):
            # ncsf_info[key] = np.array([ncsf_info[key]])                
            ncsf_info[key] = self.pd_params[key][[idx]].to_numpy()
        
        # [1] CSF in design matrix space:
        ncsf_info['part_csf_curve'] = asymmetric_parabolic_CSF(
            SF_seq = self.SF_list, 
            width_r     = ncsf_info['width_r'], 
            SFp         = ncsf_info['SFp'], 
            CSp         = ncsf_info['CSp'], 
            width_l     = ncsf_info['width_l'],                         
        ).squeeze()

        # [2] Smooth form of nCSF, i.e. not just sampling those points in stimulus
        sf_grid = np.logspace(np.log10(self.SF_list[0]),np.log10(50), 100)
        con_grid = np.logspace(np.log10(.1),np.log10(100), 100)
        full_csf = nCSF_response_grid(
            SF_list     = sf_grid, 
            CON_list    = con_grid,
            width_r     = ncsf_info['width_r'], 
            SFp         = ncsf_info['SFp'], 
            CSp         = ncsf_info['CSp'], 
            width_l     = ncsf_info['width_l'], 
            crf_exp     = ncsf_info['crf_exp'],    
            edge_type   = self.edge_type,        
            )
        full_csf_curve = asymmetric_parabolic_CSF(
            SF_seq      = sf_grid, 
            width_r     = ncsf_info['width_r'], 
            SFp         = ncsf_info['SFp'], 
            CSp         = ncsf_info['CSp'], 
            width_l     = ncsf_info['width_l'],                         
        )     
        ncsf_info['sf_grid'],ncsf_info['con_grid']  = np.meshgrid(sf_grid, con_grid)
        ncsf_info['full_csf']          = full_csf
        ncsf_info['full_csf_curve']    = full_csf_curve
        
        # Calculate the time series for the parameters
        if 'hrf_1' in ncsf_info.keys():
            hrf_1 = ncsf_info['hrf_1']
            hrf_2 = ncsf_info['hrf_2']
        else:
            hrf_1 = None
            hrf_2 = None
        ncsf_info['ts'] = self.prfpy_model.return_prediction(
            width_r     = ncsf_info['width_r'],
            SFp         = ncsf_info['SFp'],
            CSp         = ncsf_info['CSp'],
            width_l     = ncsf_info['width_l'],
            crf_exp     = ncsf_info['crf_exp'],
            beta        = ncsf_info['amp_1'],
            baseline    = ncsf_info['bold_baseline'],
            hrf_1       = hrf_1,
            hrf_2       = hrf_2,
        )
        return ncsf_info
    
    def prf_ts_plot(self, idx, time_pt=None, **kwargs):    
        return self.csf_ts_plot(idx, time_pt, **kwargs)

        
    def return_predictions(self, idx=None):
        if idx is None:
            idx = np.ones(self.n_vox, dtype=bool)
        if 'hrf_1' in self.pd_params.keys():
            hrf_1 = self.pd_params['hrf_1'][idx]
            hrf_2 = self.pd_params['hrf_2'][idx]
        else:
            hrf_1 = None            
            hrf_2 = None
        preds = self.prfpy_model.return_prediction(
            width_r     = self.pd_params['width_r'][idx],
            SFp         = self.pd_params['SFp'][idx],
            CSp         = self.pd_params['CSp'][idx],
            width_l     = self.pd_params['width_l'][idx],
            crf_exp     = self.pd_params['crf_exp'][idx],
            beta        = self.pd_params['amp_1'][idx],
            baseline    = self.pd_params['bold_baseline'][idx],
            hrf_1       = hrf_1,
            hrf_2       = hrf_2,
        )
        return preds
    
    def csf_ts_plot(self, idx, time_pt=None, **kwargs):
        '''csf_ts_plot
        Do a nice representation of the CSF timeseries model
        '''
        TR_in_s = self.TR_in_s
        do_text     = kwargs.get('do_text', True)
        do_stim_info = kwargs.get('do_stim_info', True)
        do_crf = kwargs.get('do_crf', True)
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')
        fig = kwargs.get('fig', plt.figure()) #plt.figure())
        fig.clear()
        return_fig = kwargs.get('return_fig', True)
        dpi = kwargs.get('dpi', 100)
        # Load the specified info 
        ncsf_info = self.csf_ts_plot_get_info(idx)
        ts_x = np.arange(0, ncsf_info['ts'].shape[-1]) * TR_in_s
        # Set up figure
        # SIMPLIFIED 

        grow_by = kwargs.get('grow_by', 1.5)
        if do_crf:
            width_ratios = [2, 2, 6] 
        else:
            width_ratios = [2, 6]
        if do_stim_info:
            height_ratios = [2,1]
        else:
            height_ratios = [1]
        # Apply new size & dpi
        fig.set_size_inches(sum(width_ratios)*grow_by, sum(height_ratios)*grow_by)
        fig.set_dpi(dpi)
        # Create axs
        gs = mpl.gridspec.GridSpec(len(height_ratios), len(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios)
        csf_ax = fig.add_subplot(gs[0, 0])
        ts_ax = fig.add_subplot(gs[0, -1])
        if do_stim_info:
            SF_ax = fig.add_subplot(gs[1, -1])
        if do_crf:
            crf_ax = fig.add_subplot(gs[0, 1])
        # *********** ax -1,2: Stimulus info ***********
        if do_stim_info:
            self.sub_plot_stim_info(
                ax=SF_ax, ncsf_info=ncsf_info, 
                time_pt=time_pt,kwargs=kwargs,
            )

        # CSF curve + with imshow to display CRF curve         
        self.sub_plot_csf(
            ax=csf_ax, ncsf_info=ncsf_info, 
            time_pt=time_pt, kwargs=kwargs,           
        )
        if do_crf:
            # CRF
            self.sub_plot_crf(
                ax=crf_ax, ncsf_info=ncsf_info, 
                time_pt=time_pt, kwargs=kwargs,            
            )        
        # Time series
        self.sub_plot_ts(
            ax=ts_ax, idx=idx, ncsf_info=ncsf_info, 
            time_pt=time_pt, kwargs=kwargs,       
        )


        if do_text:            
            ncsf_txt = self.make_prf_str(
                idx=idx, 
                pid_list=['width_r', 'SFp', 'CSp', 'width_l', 'crf_exp', 'aulcsf', 'rsq' ]
                )
            ts_ax.text(1.35, 0.20, ncsf_txt, transform=ts_ax.transAxes, fontsize=10, va='center', ha='right', family='monospace',)
        # ***********************************************************************
        update_fig_fontsize(fig, new_font_size=1.2, font_multiply=True)
        fig.set_tight_layout(True)
        if return_fig:
            return fig

    def sub_plot_stim_info(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        ts_x = np.arange(0, ncsf_info['ts'].shape[-1]) * self.TR_in_s

        SF_ax = ax
        # Add the stimulus plots
        SF_ax.set_yscale('log')
        SF_ax.set_xlabel('time (s)')
        SF_ax.set_ylabel('SF') # log SF', color='black')
        SF_ax.yaxis.set_label_position('right')
        SF_ax.set_yticks([])
        # -> SF sequence
        SF_seq = self.prfpy_model.stimulus.SF_seq.copy()
        if self.prfpy_model.stimulus.discrete_levels:
            # Find indices where the values change ( & are not to 0)
            change_indices = np.where((np.diff(SF_seq) != 0) & (SF_seq[1:] != 0))[0]
            # Create a list of labels corresponding to the changed values
            labels = [f'{value:0.1f}' for value in SF_seq[change_indices+1]]
            labels = [value.split('.0')[0] for value in labels]
            # Add text labels at the change points on the plot
            for id, label in zip(change_indices + 1, labels):
                SF_ax.text(
                    id*self.TR_in_s+3*self.TR_in_s, 
                    SF_seq[id], 
                    label,
                    color=self._get_SF_cols(SF_seq[id]),
                    ha='center', va='bottom', ) 
            

        SF_ax.plot(ts_x, SF_seq, 'k', linestyle='', marker='_')                
        # SF_ax.spines['right'].set_visible(False)
        SF_ax.spines['top'].set_visible(False)


        # -> contrast
        con_seq = self.prfpy_model.stimulus.CON_seq.copy()
        con_seq[con_seq==0] = np.nan
        con_ax = SF_ax.twinx()                        
        con_ax.plot(ts_x, con_seq, 'r')
        # set ylabel to red, also yticks
        con_ax.set_ylabel('contrast ', color='red', alpha=0.5)        
        con_ax.set_yscale('log')
        con_ax.tick_params(axis='y', colors='red')
        con_ax.spines['right'].set_visible(False)
        con_ax.spines['top'].set_visible(False)
        con_ax.yaxis.set_label_position('left')
        con_ax.yaxis.set_ticks_position('left')
        # Add grey patches corresponding to the nan values in con_s_seq
        y1 = np.ones_like(ts_x)*np.nanmin(con_seq)
        y2 = np.ones_like(ts_x)*np.nanmax(con_seq)
        con_ax.fill_between(ts_x, y1, y2, where=np.isnan(con_seq), facecolor='grey', alpha=0.5)
        # set xlim
        con_ax.set_xlim(0, ts_x[-1])    
        if time_pt is not None:
            con_ax.plot(
                (time_pt*self.TR_in_s, time_pt*self.TR_in_s), (y1[0], y2[0]),
                color=self.time_pt_col, linewidth=5, alpha=0.8)

        # put x axis for con_s_ax and SF_ax at the top of the axis
        # SF_ax.xaxis.tick_top()

    def sub_plot_csf(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        csf_ax = ax
        # Scatter the points sampled
        # csf_ax.scatter(
        #     self.prfpy_model.stimulus.SF_seq, 100/self.prfpy_model.stimulus.CON_seq, color='r', alpha=0.8
        # )
        csf_ax.plot(
            ncsf_info['sf_grid'][0,:],
            ncsf_info['full_csf_curve'].squeeze(),
            lw=5, color='g',
        )

        csf_ax.scatter(
            ncsf_info['sf_grid'].ravel(),
            100/ncsf_info['con_grid'].ravel(),
            c=ncsf_info['full_csf'].ravel(),
            # vmin=0, vmax=1,
            alpha=1,
            cmap='magma',
            lw=0, edgecolor=None,             
        )   
        # Add a colorbar
        # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='magma'), ax=csf_ax)
        # cbar.set_label('fMRI response (a.u.)')
        if time_pt is not None:
            csf_ax.plot(
                self.prfpy_model.stimulus.SF_seq[time_pt],
                100/self.prfpy_model.stimulus.CON_seq[time_pt],
                color=time_pt_col, marker='*', markersize=20,
            )

        csf_ax.set_xlabel('SF (c/deg)')
        csf_ax.set_ylabel('contrast sensitivity')
        csf_ax.set_xscale('log')
        csf_ax.set_yscale('log')  
        xticklabels = ['0.5', '1', '10', '50']
        xticks = [float(i) for i in xticklabels]
        xlim = [xticks[0], xticks[-1]]
        yticklabels = ['1', '10', '100']
        yticks = [float(i) for i in yticklabels]
        ylim = [1, 500]
        csf_ax.set_box_aspect(1)
        csf_ax.set_xticks(xticks) 
        csf_ax.set_xticklabels(xticklabels) 
        csf_ax.set_xlim(xlim) 
        csf_ax.set_yticks(yticks)
        csf_ax.set_yticklabels(yticklabels)
        csf_ax.set_ylim(ylim)
        csf_ax.spines['right'].set_visible(False)
        csf_ax.spines['top'].set_visible(False)        

    def sub_plot_crf(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')
        do_log_contrast = kwargs.get('do_log_contrast', True)    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        crf_ax = ax
        # Contrast response function at different SFs 
        crf_ax.set_title(f'CRF')    
        crf_ax.set_xlabel('contrast (%)')
        crf_ax.set_ylabel('fMRI response (a.u.)')
        if do_log_contrast:
            contrasts = np.logspace(np.log10(0.1), np.log10(100), 100)
        else:
            contrasts = np.linspace(0,100,100)
        max_R = 0
        for iSF, vSF in enumerate(self.SF_list):
            # Plot the CRF at each SF we sample in the stimulus
            # [1] Get the "Q" aka "C50" aka "semisaturation point"
            # -> i.e., where response=50%
            # -> we get this using the CSF curve
            this_Q = ncsf_info['part_csf_curve'][iSF] 
            # print(f'vSF={vSF}, this_Q={this_Q}')
            this_crf = ncsf_calculate_crf_curve(
                crf_exp=ncsf_info['crf_exp'],
                Q=this_Q, 
                C=contrasts,
                edge_type=self.edge_type,
            )
            crf_ax.plot(
                contrasts, 
                this_crf.squeeze(), 
                alpha=0.8,
                color=self._get_SF_cols(vSF),
                label=f'{vSF:.1f}',
            )
            max_R = max(max_R, np.nanmax(this_crf.squeeze()))
        # bloop
        # Put a grid on the axis (only the major ones)
        crf_ax.grid(which='both', axis='both', linestyle='--', alpha=0.5)
        if do_log_contrast:
            ax.set_xscale('log')
            crf_ax.set_xticks([0.1, 10,100])
            crf_ax.set_xlim([0.1, 100]) # ax.set_xlim([0, 100])
        else:
            crf_ax.set_xticks([0, 50,100])
            crf_ax.set_xlim([0, 100]) # ax.set_xlim([0, 100])            
        ax.set_yticklabels([0, '50%', 'MAX'])
        # Make the axis square
        crf_ax.set_box_aspect(1) 
        # ax.set_title('CRF')
        if max_R < 1:
            crf_ax.set_yticks([0, 0.5, 1.0])
            crf_ax.set_ylim([0, 1])
        else:
            crf_ax.set_yticks([0, max_R/2, max_R])
            crf_ax.set_ylim([0, max_R])
        crf_ax.set_xlabel('contrast (%)')
        crf_ax.set_ylabel('fMRI response (a.u.)')
        # 
        crf_ax.spines['right'].set_visible(False)
        crf_ax.spines['top'].set_visible(False)            
        if len(self.SF_cols) > 10:
            self._add_SF_colorbar(crf_ax)
        else:
            leg = crf_ax.legend(
                handlelength=0, handletextpad=0, fancybox=True,
                bbox_to_anchor=(1.3, 1), loc='upper right',
                )
            # for item in leg.legendHandles:
            #     item.set_visible(False)        
            for color,text in zip(self.SF_cols.values(),leg.get_texts()):
                text.set_color(color)        
    
    def sub_plot_ts(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        ts_ax = ax
        ts_ax.plot(ncsf_info['ts'][0,:time_pt], color='g', marker="*", markersize=2, linewidth=3, alpha=0.8)        
        if self.real_ts is not None:
            ts_ax.plot(self.real_ts[idx,:time_pt], color='k', linestyle=':', marker='^', linewidth=1.5, alpha=0.8)
        ts_ax.set_xlim(0, ncsf_info['ts'].shape[-1])
        ts_ax.set_title('')
        ts_ax.plot((0,ncsf_info['ts'].shape[-1]), (0,0), 'k')   
        # Find the time for 0 stimulation, add grey patches
        id_no_stim = self.prfpy_model.stimulus.SF_seq==0.0
        x = np.arange(len(id_no_stim))
        y1 = np.ones_like(x)*np.nanmin(ncsf_info['ts'])
        y2 = np.ones_like(x)*np.nanmax(ncsf_info['ts'])
        ts_ax.fill_between(x, y1, y2, where=id_no_stim, facecolor='grey', alpha=0.5)    
        if time_pt is not None:
            ts_ax.plot(
                (time_pt, time_pt), (y1[0], y2[0]),
                color=time_pt_col, linewidth=1, alpha=0.8)    
            # also plot a full invisible version, to keep ax dim...
            ts_ax.plot(ncsf_info['ts'][0,:], alpha=0)
            if self.real_ts is not None:
                ts_ax.plot(self.real_ts[idx,:], alpha=0)

    def return_csf_curve(self, idx, **kwargs):
        '''return_csf_curve
        Return the CSF curve for a voxel
        '''
        if isinstance(idx, int):
            idx = [idx]        
        SF_list = kwargs.get('SF_list', self.SF_list)
        csf_curve = asymmetric_parabolic_CSF(
            SF_seq      = SF_list,
            width_r     = self.pd_params['width_r'][idx].to_numpy(), 
            SFp         = self.pd_params['SFp'][idx].to_numpy(), 
            CSp         = self.pd_params['CSp'][idx].to_numpy(), 
            width_l     = self.pd_params['width_l'][idx].to_numpy(),                         
        )
        
        return csf_curve

    def return_csf_mat(self, idx, **kwargs):
        if isinstance(idx, int):
            idx = [idx]
        SF_list = kwargs.get('SF_list', self.SF_list)
        con_list = kwargs.get('con_list', np.logspace(np.log10(.25), np.log10(100), 100))
        csf_mat = nCSF_response_grid(
            SF_list     = SF_list, 
            CON_list    = con_list,
            width_r     = self.pd_params['width_r'][idx].to_numpy(), 
            SFp         = self.pd_params['SFp'][idx].to_numpy(), 
            CSp         = self.pd_params['CSp'][idx].to_numpy(), 
            width_l     = self.pd_params['width_l'][idx].to_numpy(),                         
            crf_exp     = self.pd_params['width_l'][idx].to_numpy(),
            edge_type   = self.edge_type,        
            )
        return csf_mat
    
    def return_crf_curve(self, idx, **kwargs):
        if isinstance(idx, int):
            idx = [idx]
        # if isinstance(idx, )
        ow_Qs = kwargs.get('ow_Qs', False)
        Qs_at_CSp = kwargs.get('Qs_at_CSp', False)
        sf_for_crf = kwargs.get('sf_for_crf', 6.0)
        con_list = kwargs.get('con_list', np.logspace(np.log10(.25), np.log10(100), 100))
        crf_exp_vals = self.pd_params['crf_exp'][idx].to_numpy()
        if ow_Qs:
            # if ncsf_params['crf_exp'].shape[0]==1:
            # return
            if not hasattr(crf_exp_vals, 'shape'):
                # print('if not hasattr(ncsf_params[crf_exp], shape):')
                Qs = 20 * np.array([1])[...,np.newaxis]
            elif crf_exp_vals.shape ==():
                # print('elif ncsf_params[crf_exp].shape ==():')
                Qs = 20 * np.array([1])#[...,np.newaxis]
            else:
                # print('else')
                Qs = 20 * np.ones_like(crf_exp_vals)

        elif Qs_at_CSp:
            # Setelct
            Qs = 100/self.pd_params['CSp'][idx].to_numpy() 
        else:
            sf_grid = [0.5, sf_for_crf] # doesn't matter
            csf_curve = asymmetric_parabolic_CSF(
                SF_seq      = sf_grid, #csenf_stim.log_SF_grid, 
                width_r     = self.pd_params['width_r'][idx].to_numpy(), 
                SFp         = self.pd_params['SFp'][idx].to_numpy(), 
                CSp         = self.pd_params['CSp'][idx].to_numpy(), 
                width_l     = self.pd_params['width_l'][idx].to_numpy(), 
                ).T         
            Qs = 100/csf_curve[1,:]  

        crf_curves = nCSF_apply_crf(
            csenf_values=(100/Qs)[...,np.newaxis], 
            crf_exp=crf_exp_vals[...,np.newaxis], 
            CON_seq = con_list,
            edge_type=self.edge_type,
        )
        return crf_curves


# ************************** 
def update_fig_fontsize(fig, new_font_size, font_multiply=False):
    '''dpu_update_fig_fontsize
    Description:
        Update the font size of a figure
    Input:
        fig             matplotlib figure
        new_font_size   int/float             
    Return:
        None        
    '''
    fig_kids = fig.get_children() # Get the children of the figure, i.e., the axes
    for i_kid in fig_kids: # Loop through the children
        if isinstance(i_kid, mpl.axes.Axes): # If the child is an axes, update the font size of the axes
            update_ax_fontsize(i_kid, new_font_size, font_multiply)
        elif isinstance(i_kid, mpl.text.Text): # If the child is a text, update the font size of the text
            update_this_item_fontsize(i_kid, new_font_size, font_multiply)         
                


def update_ax_fontsize(ax, new_font_size, font_multiply=False, include=None, do_extra_search=True):
    '''dpu_update_ax_fontsize
    Description:
        Update the font size of am axes
    Input:
        ax              matplotlib axes
        new_font_size   int/float
        *Optional*
        include         list of strings     What to update the font size of. 
                                            Options are: 'title', 'xlabel', 'ylabel', 'xticks','yticks'
        do_extra_search bool                Whether to search through the children of the axes, and update the font size of any text
    Return:
        None        
    '''
    if include is None: # If no include is specified, update all the text       
        include = ['title', 'xlabel', 'ylabel', 'xticks','yticks']
    if not isinstance(include, list): # If include is not a list, make it a list
        include = [include]
    incl_list = []
    for i in include: # Loop through the include list, and add the relevant text to the list
        if i=='title': 
            incl_list += [ax.title]
        elif i=='xlabel':
            incl_list += [ax.xaxis.label]
        elif i=='ylabel':
            incl_list += [ax.yaxis.label]
        elif i=='xticks':
            incl_list += ax.get_xticklabels()
        elif i=='yticks':
            incl_list += ax.get_yticklabels()
        elif i=='legend':
            incl_list += ax.get_legend().get_texts()

    for item in (incl_list): # Loop through the text, and update the font size
        update_this_item_fontsize(item, new_font_size, font_multiply)
    if do_extra_search:
        for item in ax.get_children():
            if isinstance(item, mpl.legend.Legend):
                texts = item.get_texts()
                if not isinstance(texts, list):
                    texts = [texts]
                for i_txt in texts:
                    update_this_item_fontsize(i_txt, new_font_size, font_multiply)
            elif isinstance(item, mpl.text.Text):
                update_this_item_fontsize(item, new_font_size, font_multiply)
def update_this_item_fontsize(this_item, new_font_size, font_multiply):
    if font_multiply:
        new_fs = this_item.get_fontsize() * new_font_size
    else:
        new_fs = new_font_size.copy()
    this_item.set_fontsize(new_fs)                                                        