import numpy as np
import os
opj = os.path.join
import matplotlib.image as mpimg
from scipy import io, interpolate
from scipy.ndimage import zoom

opj = os.path.join

def print_p():
    '''
    Easy look up table for prfpy model parameters
    name to index...
    '''
    p_order = {}
    # [1] gauss. Note hrf_1, and hrf_2 are idx 5 and 6, if fit...
    p_order['gauss'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # size
        'amp_1'         :  3, # beta
        'bold_baseline' :  4, # baseline 
        'hrf_deriv'     :  5, # *hrf_1
        'hrf_disp'      :  6, # *hrf_2
        'rsq'           : -1, # ... 
    }    
    # [2] css. Note hrf_1, and hrf_2 are idx 6 and 7, if fit...
    p_order['css'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # size
        'amp_1'         :  3, # beta
        'bold_baseline' :  4, # baseline 
        'n_exp'         :  5, # n
        'hrf_deriv'     :  6, # *hrf_1
        'hrf_disp'      :  7, # *hrf_2        
        'rsq'           : -1, # ... 
    }

    # [3] dog. Note hrf_1, and hrf_2 are idx 7 and 8, if fit...
    p_order['dog'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # prf_size
        'amp_1'         :  3, # prf_amplitude
        'bold_baseline' :  4, # bold_baseline 
        'amp_2'         :  5, # srf_amplitude
        'size_2'        :  6, # srf_size
        'hrf_deriv'     :  7, # *hrf_1
        'hrf_disp'      :  8, # *hrf_2        
        'rsq'           : -1, # ... 
    }

    p_order['norm'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # prf_size
        'amp_1'         :  3, # prf_amplitude
        'bold_baseline' :  4, # bold_baseline 
        'amp_2'         :  5, # srf_amplitude
        'size_2'        :  6, # srf_size
        'b_val'         :  7, # neural_baseline 
        'd_val'         :  8, # surround_baseline
        'hrf_deriv'     :  9, # *hrf_1
        'hrf_disp'      : 10, # *hrf_2        
        'rsq'           : -1, # rsq
    }            

    p_order['csf']  ={
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


def filter_for_nans(array):
    """filter out NaNs from an array"""

    if np.isnan(array).any():
        return np.nan_to_num(array)
    else:
        return array


def resample2d(array: np.ndarray, new_size: int, kind: str = 'linear') -> np.ndarray:
    """
    Resample a 2D or 3D array to (new_size, new_size) using linear interpolation.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (H, W) or (H, W, C).
    new_size : int
        Desired output height and width.
    kind : {'linear'}, optional
        Only 'linear' is supported (default).

    Returns
    -------
    np.ndarray
        Resampled array of shape (new_size, new_size) or
        (new_size, new_size, C) for 3D input.
    """
    if kind != 'linear':
        raise ValueError("Only linear interpolation is supported.")

    # compute zoom factors for height and width
    zoom_y = new_size / array.shape[0]
    zoom_x = new_size / array.shape[1]

    if array.ndim == 2:
        # single-channel
        return zoom(array, (zoom_y, zoom_x), order=1)
    elif array.ndim == 3:
        # preserve channel axis
        return zoom(array, (zoom_y, zoom_x, 1), order=1)
    else:
        raise ValueError("Input must be 2D or 3D.")

def get_prfdesign(screenshot_path, n_pix=100, dm_edges_clipping=[0,0,0,0]):
    """
    get_prfdesign

    Basically Marco's gist, but then incorporated in the repo. It takes the directory of screenshots and creates a vis_design.mat file, telling pRFpy at what point are certain stimulus was presented.

    Parameters
    ----------
    screenshot_path: str
        string describing the path to the directory with png-files
    n_pix: int
        size of the design matrix (basically resolution). The larger the number, the more demanding for the CPU. It's best to have some value which can be divided with 1080, as this is easier to downsample. Default is 40, but 270 seems to be a good trade-off between resolution and CPU-demands
    dm_edges_clipping: list, dict, optional
        people don't always see the entirety of the screen so it's important to check what the subject can actually see by showing them the cross of for instance the BOLD-screen (the matlab one, not the linux one) and clip the image accordingly. This is a list of 4 values, which are the number of pixels to clip from the left, right, top and bottom of the image. Default is [0,0,0,0], which means no clipping. Negative values will be set to 0.

    Returns
    ----------
    numpy.ndarray
        array with shape <n_pix,n_pix,timepoints> representing a binary paradigm

    Example
    ----------
    >>> dm = get_prfdesign('path/to/dir/with/pngs', n_pix=270, dm_edges_clipping=[6,1,0,1])
    """

    image_list = os.listdir(screenshot_path)

    # get first image to get screen size
    img = (255*mpimg.imread(opj(screenshot_path, image_list[0]))).astype('int')

    # there is one more MR image than screenshot
    design_matrix = np.zeros((img.shape[0], img.shape[0], 1+len(image_list)))

    for image_file in image_list:

        # assuming last three numbers before .png are the screenshot number
        img_number = int(image_file[-7:-4])-1

        # subtract one to start from zero
        img = (255*mpimg.imread(opj(screenshot_path, image_file))).astype('int')

        # make it square
        if img.shape[0] != img.shape[1]:
            offset = int((img.shape[1]-img.shape[0])/2)
            img = img[:, offset:(offset+img.shape[0])]
        cross = np.zeros(img.shape[0:2])
        cross[np.where(((img[...,0] == 0) & (
            img[...,1] == 0)) | ((img[...,0] == 255) & (img[...,1] == 255)))] = 1
        cross[np.where(((img[...,0] == img[...,1]) & (
            img[...,1] == img[...,2]) & (img[...,0] != 127)))] = 1
        bar = (np.sum(img, axis=-1) == 384)*1.0
        bar += cross
        design_matrix[...,img_number] = bar == 1.0
        # # binarize image into dm matrix
        # # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
        # design_matrix[...,img_number][np.where(((img[...,0] == 0) & (
        #     img[...,1] == 0)) | ((img[...,0] == 255) & (img[...,1] == 255)))] = 1

        # design_matrix[...,img_number][np.where(((img[...,0] == img[...,1]) & (
        #     img[...,1] == img[...,2]) & (img[...,0] != 127)))] = 1

    #clipping edges; top, bottom, left, right
    if isinstance(dm_edges_clipping, dict):
        dm_edges_clipping = [
            dm_edges_clipping['top'],
            dm_edges_clipping['bottom'],
            dm_edges_clipping['left'],
            dm_edges_clipping['right']]

    # ensure absolute values; should be a list by now anyway
    dm_edges_clipping = [abs(ele) for ele in dm_edges_clipping]

    design_matrix[:dm_edges_clipping[0], :, :] = 0
    design_matrix[(design_matrix.shape[0]-dm_edges_clipping[1]):, :, :] = 0
    design_matrix[:, :dm_edges_clipping[2], :] = 0
    design_matrix[:, (design_matrix.shape[0]-dm_edges_clipping[3]):, :] = 0

    # downsample (resample2d can also deal with 3D input)
    if n_pix != design_matrix.shape[0]:
        dm_resampled = resample2d(design_matrix, n_pix)
        dm_resampled[dm_resampled<0.9] = 0
        return dm_resampled
    else:
        return design_matrix




def raw_ts_to_average_psc(raw_ts, baseline=20):
    '''raw_ts_to_average_psc
    Function to return average, percent signal change data
    Parameters
    -------
    raw_ts      list of np.ndarrays. Where np.ndarray is a run
                Or a single np.ndarray, if there is only 1 run
                Each np.ndarray is n voxels/vertices X number of timepoints
    Returns
    -------
    psc_avg_ts  np.ndarray
    
    '''
    # Convert to psc
    if not isinstance(raw_ts, list):
        raw_ts = [raw_ts]

    psc_ts = []
    for this_run in raw_ts:
        psc_ts.append(percent_change(this_run, baseline=baseline))
    
    psc_avg_ts = np.median(np.array(psc_ts), 0)

    return psc_avg_ts


def percent_change(ts, baseline=20):
    """percent_change
    Function to convert input data to percent signal change. Two options are current supported: the nilearn method (`nilearn=True`), where the mean of the entire timecourse if subtracted from the timecourse, and the baseline method (`nilearn=False`), where the median of `baseline` is subtracted from the timecourse.
    Parameters
    ----------
    ts: numpy.ndarray
        Array representing the data to be converted to percent signal change. Should be of shape (n_voxels, n_timepoints)        
    baseline: int, list, np.ndarray optional
        Use custom method where only the median of the baseline (instead of the full timecourse) is subtracted, by default 20. Length should be in `volumes`, not `seconds`. Can also be a list or numpy array (1d) of indices which are to be considered as baseline. The list of indices should be corrected for any deleted volumes at the beginning.
    Returns
    ----------
    numpy.ndarray
        Array with the same size as `ts` (voxels,time), but with percent signal change.
    Raises
    ----------
    ValueError
        If `ax` > 2
    """
    
    if ts.ndim == 1:
        ts = ts[:,np.newaxis]
    vx_dim = 0 # axis for voxels
    t_dim = 1  # axis for time
    
    # first step of PSC; set NaNs to zero if dividing by 0 (in case of crappy timecourses)
    psc_factor = np.nan_to_num( 100 / np.mean(ts, axis=t_dim)) 
    ts_m = ts*psc_factor[...,np.newaxis]

    if isinstance(baseline, int):
        # Is the baseline an int? 
        # Then use 0:baseline as the timepoints for finding the median baseline        
        median_baseline = np.median(ts_m[:, :baseline], axis=t_dim)    
    elif isinstance(baseline, np.ndarray):        
        # Is the baseline an array? Then convert to list 
        baseline = list(baseline)

    if isinstance(baseline, list):
        # Is the baseline a list?
        # Then use the specified indices as the timepoints for finding the median baseline
        median_baseline = np.median(ts_m[:, baseline], axis=t_dim)    

    # subtract
    psc = ts_m-median_baseline[..., np.newaxis]
    
    return psc

def get_rsq(tc_target, tc_fit):
    ss_res = np.sum((tc_target-tc_fit)**2, axis=-1)
    ss_tot = np.sum(
        (tc_target-tc_target.mean(axis=-1)[...,np.newaxis])**2, 
        axis=-1
        )
    rsq = 1-(ss_res/ss_tot)

    return rsq