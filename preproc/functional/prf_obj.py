import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from copy import deepcopy
import sys
opj = os.path.join


from dpu_mini.utils import *
from dpu_mini.plot_functions import *

# ********** PRF OBJECTS
class PrfObj(object):
    '''Prf1T1M
    For use with bcoder.

    '''
    def __init__(self, pd_params, **kwargs):
        '''__init__
        Input:
        ----------
        '''
        self.model_labels = list(pd_params.keys())
        self.pd_params_in = pd_params.copy()
        self.pd_params = pd_params.copy()
        self.saved_kwargs = kwargs
        self.n_vox = self.pd_params.shape[0]
        self.model=kwargs.get('model', '')
        self.bcoder_model = kwargs.get('bcoder_model', None)
        self.data = kwargs.get('data', None)
        if self.bcoder_model is not None:
            # self.preds = self.bcoder_model.predict(parameters=self.pd_params_in)
            if self.model == 'csf':
                sfs = np.unique(self.bcoder_model.paradigm['SF'])
                cons = np.unique(self.bcoder_model.paradigm['CON'])
                # sfs = np.logspace(np.log10(0.1), np.log10(50), 100)
                # cons = np.logspace(np.log10(0.1), np.log10(100), 100)
                # sfs,cons=np.meshgrid(sfs,cons)
                # self.sfs,self.cons = sfs.flatten, cons.flatten
                csfs,sfs,cons = self.bcoder_model.get_csf_for_plot(
                    parameters=self.pd_params_in, 
                    SF_grid=sfs, CON_grid=cons, 
                    )
                self.csfs,self.sfs,self.cons=csfs,sfs,cons
        # Calculate extra interesting stuff
        if 'x' in self.pd_params.keys():
            # Ecc, pol
            self.pd_params['ecc'], self.pd_params['pol'] = dag_coord_convert(
                self.pd_params['x'],self.pd_params['y'],'cart2pol')      

        # if self.model in ('norm', 'dog'):
        #     # -> size ratio:
        #     self.pd_params['size_ratio'] = self.pd_params['size_2'] / self.pd_params['size_1']
        #     self.pd_params['amp_ratio'] = self.pd_params['amp_2'] / self.pd_params['amp_1']
        # if self.model == 'norm':
        #     self.pd_params['bd_ratio'] = self.pd_params['b_val'] / self.pd_params['d_val']
        #     # Suppression index 
        #     self.pd_params['sup_idx'] = (self.pd_params['amp_1'] * self.pd_params['size_1']**2) / (self.pd_params['amp_2'] * self.pd_params['size_2']**2)
        if self.model=='csf':
            #     self.pd_params['log10_SFp'] = np.log10(self.pd_params['SFp'])
            #     self.pd_params['log10_CSp'] = np.log10(self.pd_params['CSp'])
            #     self.pd_params['log10_crf_exp'] = np.log10(self.pd_params['crf_exp'])
            self.pd_params['sfmax'] = ncsf_calculate_sfmax(
                self.pd_params['width_r'],
                self.pd_params['SFp'],
                self.pd_params['CSp'],
            )
            #     self.pd_params['log10_sfmax'] = np.log10(self.pd_params['sfmax'])
            self.pd_params['AUC'] = ncsf_calculate_aulcsf(
                self.pd_params['width_r'],
                self.pd_params['SFp'],
                self.pd_params['CSp'],    
                self.pd_params['width_l'],
                normalize_AUC=True,                                          
            )

    def return_vx_mask(self, th={}):
        '''return_vx_mask
        Returns a mask (boolean array) for voxels
        
        Notes: 
        ----------
        th keys must be split into 2 parts
        'comparison-param' : value
        e.g.: to exclude gauss fits with r2 less than 0.1
        th = {'min-r2': 0.1 } 
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
        
    def hist(self, param, th={'min-r2':.1}, ax=None, **kwargs):
        '''hist: Plot a histogram of a parameter, masked by th'''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)        
        ax.hist(self.pd_params[param][vx_mask].to_numpy(), **kwargs)
        ax.set_title(param)

    def visual_field(self, th={}, ax=None, dot_col='k', **kwargs):
        '''visual_field
        Plot the visual field of the voxels, masked by the vx_mask
        and colored by a parameter

        Notes:
        ----------
        Default vx mask is all voxels with r2 > 0.1 and ecc < 5

        Input:
        ----------
        Optional:
        th          dict, threshold for parameters
        ax          matplotlib.axes, if None, then plt.axes() is used
        dot_col     str, color of the dots
        kwargs      dict, kwargs for dag_visual_field_scatter
        '''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        if isinstance(dot_col,str):
            if dot_col in self.pd_params.keys():
                dot_col = self.pd_params[dot_col][vx_mask].to_numpy()

        dag_visual_field_scatter(
            ax=ax, 
            dot_x=self.pd_params['x'][vx_mask].to_numpy(),
            dot_y=self.pd_params['y'][vx_mask].to_numpy(),
            dot_col = dot_col,
            **kwargs
        )        

    def scatter(self, px, py, th={'min-r2':.1}, ax=None, **kwargs):
        '''scatter
        Scatter plot of 2 parameters, masked by the vx_mask
        Can also color by a third parameter

        Notes:
        ----------
        Default vx mask is all voxels with r2 > 0.1

        Input:
        ----------
        px          str, parameter to plot on x axis
        py          str, parameter to plot on y axis
        Optional:
        th          dict, threshold for parameters
        ax          matplotlib.axes, if None, then plt.axes() is used
        dot_col     str, color of the dots
        dot_alpha   float, alpha of the dots
        kwargs      dict, kwargs for dag_scatter

        '''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        pc = kwargs.get('pc', None)        
        if pc is not None:
            kwargs['dot_col'] = self.pd_params[pc][vx_mask]
        dag_scatter(
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

    def multi_scatter(self, px_list, th={'min-r2':.1}, **kwargs):
        '''multi_scatter
        Several scatter plots... multiple comparisons...
        i.e., creates a grid of scatter plots
        '''
        tmp_dict = self.return_th_params(px_list, th, **kwargs)
        fig, ax_list = dag_multi_scatter(tmp_dict, **kwargs)            
        return fig, ax_list

    def prf_plotter(self,idx, do_str=False):
        if self.model.lower()=='csf':
            _ =  self.csf_ts_plotter(idx)
        elif self.model.lower()=='gauss':
            _ =  self.gauss_ts_plotter(idx)
        elif self.model.lower()=='dn':
            _ = self.twoRF_ts_plotter(idx)
        if do_str:
            pstr = self.make_prf_str(idx)
            plt.gca().text(
                1.05, 0.5, pstr, 
                transform=plt.gca().transAxes, 
                fontsize=12, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            plt.tight_layout() # Adjusts layout so the side text isn't cut off
        return _ 
    
    def twoRF_ts_plotter(self, idx):
        from braincoder.utils.visualize import quick_plot2prf
        quick_plot2prf(
            model=self.bcoder_model, 
            parameters=self.pd_params_in.iloc[idx,:],
            data=self.data.iloc[:,idx]
        )
        return plt.gcf()
    
    def gauss_ts_plotter(self, idx):
        from braincoder.utils.visualize import quick_plot
        quick_plot(
            model=self.bcoder_model, 
            parameters=self.pd_params_in.iloc[idx,:],
            data=self.data.iloc[:,idx]
        )
        return plt.gcf()
    def csf_ts_plotter(self, idx):
        fig, ax = plt.subplots(1,2,figsize=(10,4), width_ratios=(2,8))
        # prf_str = self.make_prf_str(idx)
        ax[0].scatter(
            self.sfs,100/self.cons,c=self.csfs[idx,:],
            cmap='jet'
        )
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        # ax[1].text(2, 0.5, prf_str, fontsize=10, va='center', ha='left', wrap=True)
        tpred = self.bcoder_model.predict(
            parameters=self.pd_params_in.iloc[[idx],:]
        )
        ax[1].plot(
            tpred,
            # self.preds.iloc[:,idx], 
            '-g', lw=5, alpha=0.7)
        if self.data is not None:
            ax[1].plot(
                self.data.iloc[:,idx],
                # self.preds.iloc[:,idx], 
                ':k', lw=1, alpha=0.7)
        return fig
    

class PrfMulti(object):
    '''PrfMulti
    Multiple prf obj - same subject (n vx)
    
    Notes:
    ----------
    It is important that there are the same number of voxels in each model/task    
    Create a list of PrfObj objects, and associated labels, which are all collected in
    this class.     
    It will hold all of the original Prf1T1M objects inside a dictionary    
    The idea is that it makes it easier to do comparisons across conditions/models

    Functions:
    ----------
    Data processing:
    return_vx_mask: returns a mask (boolean array) for voxels
    return_th_param: returns the specified parameters, masked by the vx_mask

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
        self.po = {}
        self.n_vox = prf_obj_list[0].n_vox
        for i,this_id in enumerate(id_list):
            self.po[this_id] = deepcopy(prf_obj_list[i])
        total_dict = {}
        for this_id in id_list:
            for p in self.po[this_id].pd_params.keys():
                total_dict[f'{this_id}-{p}'] = self.po[this_id].pd_params[p].to_numpy()
        self.pd_params = pd.DataFrame(total_dict)

    def return_vx_mask(self, th={}):
        '''return_vx_mask
        Returns a mask (boolean array) for voxels
        
        Notes: 
        ----------
        As in Prf1T1M, but with one extra part of the key:        
        th keys must be split into 3 parts
        'id-comparison-param' : value
        th = {'prf1-min-r2': 0.1 } 
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

                    p_available = list(self.po[prf_id].pd_params.keys())
                    if p in p_available:
                        vx_mask &= self.po[prf_id].return_vx_mask({f'{comp}-{p}':th_val})
                    else:
                        print(f'Warning - {p} is not a paramer for {prf_id}, ignoring...')
                continue # now next item in th_key...
            vx_mask &= self.po[id].return_vx_mask({f'{comp}-{p}':th_val})

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
            min_r2 = kwargs.get('min_r2', .1)
            th = {}
            for key in list(set(px_id)):
                th[f'{key}-min-r2'] = min_r2
        # add extra th from the default
        th_plus = kwargs.get('th_plus', {})
        th = {**th, **th_plus}
        # relevant mask 
        vx_mask = self.return_vx_mask(th)
        # create tmp dict with relevant stuff...
        tmp_dict = {}
        for i_px_id,i_px_p in zip(px_id, px_p):
            tmp_dict[f'{i_px_id}-{i_px_p}'] = self.po[i_px_id].pd_params[i_px_p][vx_mask].to_numpy()
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
                shift_dict['shift_mag'], shift_dict['shift_dir'] = dag_coord_convert(
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

        self.po[new_id] = new_prf
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
            self.po[new_id] = PrfDiff(
                self.po[id1], self.po[id2], diff_id=new_id,
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
            th = {f'{px_id}-min-r2':.1}            
        vx_mask = self.return_vx_mask(th)        
        label = kwargs.get('label', f'{px_id}-{px_p}')
        kwargs['label'] = label
        ax.hist(self.po[px_id].pd_params[px_p][vx_mask].to_numpy(), **kwargs)
        ax.set_title(f'{px_id}-{px_p}')

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
            min_r2 = kwargs.get('min_r2', .1)
            th = {
                f'{px_id}-min-r2':min_r2,
                f'{py_id}-min-r2':min_r2,
            }
            if pc is not None:
                th[f'{pc_id}-min-r2'] = min_r2

        th_plus = kwargs.get('th_plus', None)
        if not th_plus is None:
            th = {**th, **th_plus}
        vx_mask = self.return_vx_mask(th)
        if pc is not None:
            kwargs['dot_col'] = self.po[pc_id].pd_params[pc_p][vx_mask]
        if vx_mask.sum()==0:
            print('Warning: no voxels found')
            return
        try:
            dag_scatter(
                ax=ax,
                X=self.po[px_id].pd_params[px_p][vx_mask].to_numpy(),
                Y=self.po[py_id].pd_params[py_p][vx_mask].to_numpy(),
                **kwargs
            )      
        except:
            dag_scatter(
                ax=ax,
                X=self.pd_params[px][vx_mask].to_numpy(),
                Y=self.pd_params[py][vx_mask].to_numpy(),
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
        fig, ax_list = dag_multi_scatter(tmp_dict, **kwargs)            
        return fig, ax_list
    
    def arrow(self, pold, pnew, ax=None, th=None, **kwargs):
        '''arrow: arrows from one prf_obj to another'''
        if ax==None:
            ax = plt.gca()
        if th is None:
            th = {
                f'{pold}-min-r2':kwargs.get('min_r2', 0.1),
                f'{pold}-max-ecc':kwargs.get('max_ecc', 5),
                f'{pnew}-min-r2':kwargs.get('min_r2', 0.1),
                f'{pnew}-max-ecc':kwargs.get('max_ecc', 5),
                }
        th_plus = kwargs.get('th_plus', {})
        th = dict(**th, **th_plus)        
            
        vx_mask = self.return_vx_mask(th)        
        kwargs['title'] = kwargs.get('title', f'{pold}-{pnew}')

        arrow_out = dag_arrow_plot(
            ax, 
            old_x=self.po[pold].pd_params['x'][vx_mask], 
            old_y=self.po[pold].pd_params['y'][vx_mask], 
            new_x=self.po[pnew].pd_params['x'][vx_mask], 
            new_y=self.po[pnew].pd_params['y'][vx_mask], 
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
            min_r2 = kwargs.get('min_r2', 0.1)
            max_ecc = kwargs.get('max_ecc', 5)
            th = {
                f'{vf_obj}-min-r2':min_r2,
                f'{vf_obj}-max-ecc': max_ecc,
                f'{col_obj}-min-r2':min_r2,
                }        
        th_plus = kwargs.get('th_plus', {})
        th = dict(**th, **th_plus)        
        kwargs['title'] = kwargs.get('title', f'vf={vf_obj}: col={col_obj_p}')            
        vx_mask = self.return_vx_mask(th)        
        r2_weight = kwargs.get('r2_weight', False) 
        if r2_weight:
            kwargs['bin_weight'] = self.po[col_obj].pd_params['r2'][vx_mask]
        
        for p in ['dot_size', 'dot_alpha']:
            if p not in kwargs.keys():
                continue
            if isinstance(kwargs[p], str):
                # bloop
                kwargs[p] = self.pd_params[kwargs[p]][vx_mask]

        dag_visual_field_scatter(
            dot_x   = self.po[vf_obj].pd_params['x'][vx_mask],
            dot_y   = self.po[vf_obj].pd_params['y'][vx_mask],
            dot_col = self.po[col_obj].pd_params[col_p][vx_mask],
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
        if ('x' in self.model_labels1) and ('x' in self.model_labels2):
            self.pd_params['shift_mag'], self.pd_params['shift_dir'] = dag_coord_convert(
                self.pd_params['x'], self.pd_params['y'], 'cart2pol'
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