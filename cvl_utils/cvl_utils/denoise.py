from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import nibabel as nib
import scipy
from scipy.signal import savgol_filter
    
# CREDIT: Dr Nick Hedger & Dr Roni Maimon Mor 

class PCA_denoiser():
    """
    PCA_denoiser
    Class for preparing nuisance regressors and then running PCA denoising on fMRIprep output using confounds.tsv file  
    
    """
    
    def __init__(self,confounds,lf_filter, ncomps):
        """
        Initialise the PCA_denoiser class.
        
        Input arguments
        ----------
        confounds: pandas dataframe containing noise components
        lf_filter: low frequency filter to apply on the pcs
        ncomps : number of pca components
        Parameters
        ----------        
        """
        self.confound_frame=confounds
        self.lf_filter=lf_filter
        self.ncomps = ncomps
        self.prepare_frame()
        self.pca_comps = []

    def prepare_frame(self):
        """ prepare_frame
        
        Prepares the noise regressors.
        1.   Converts this frame to a numpy array. 
        2.   Converts NAs for each regressor to the median.
        3. Zscores each regressor over time.
        4. Returns the result in self.prepared_array
     
        Returns
        ----------
        self.nuissance_array: a numpy array of nuisance regressors with NANs replaced with medians.
        self.prepared_array: a z-scored numpy array of the nuisance regressors
        """
        self.nuissance_array=np.array(self.confound_frame)
        medians=np.nanmedian(self.nuissance_array,axis=0)
        for c,v in enumerate(medians):
            self.nuissance_array[:,c][np.isnan(self.nuissance_array[:,c])]=medians[c]
        if (np.std(self.nuissance_array, axis=0)==0).any():
            print(["Don't ignore this warning!!!!!"]*100)
            print('Warning: NaNs in prepared array after z-scoring. Check confounds file and preparation steps.')
            print('Warning: NaNs in prepared array after z-scoring. Check confounds file and preparation steps.')
            print('Warning: NaNs in prepared array after z-scoring. Check confounds file and preparation steps.')
            print('Warning: NaNs in prepared array after z-scoring. Check confounds file and preparation steps.')
            print('Warning: NaNs in prepared array after z-scoring. Check confounds file and preparation steps.')
            print('Warning: NaNs in prepared array after z-scoring. Check confounds file and preparation steps.')
            print('Marcus is aware of this - speak to him - he is fixing it')
            print('LIKELY DUE TO 0 MOTION FROM MCFLIRT - CHECK THIS')
            
            self.prepared_array=scipy.stats.zscore(self.nuissance_array, axis=0)
            nan_cols = np.isnan(self.prepared_array).any(axis=0)
            self.prepared_array[:, nan_cols] = 0
        else:
            self.prepared_array=scipy.stats.zscore(self.nuissance_array, axis=0)

    def run_pca(self):
        
        """ PCA_regression
        
        Runs a PCA on the noise components 
        1. Fits a PCA of n components to self.prepared_array
        1.5. Apply filter to the pca_comps
     
        Returns
        ----------
       
        self.pca_comps: PCA output of 
        """
        # Fit PCA to the prepared array
        self.pca = PCA(n_components=self.ncomps)  
        
        self.pca_comps_unfiltered = self.pca.fit_transform(self.prepared_array)
        self.pca_comps = self.lf_filter.filter_data(self.pca_comps_unfiltered.T).T
        
        return self.pca_comps

    def PCA_regression(self,data):
        
        """ PCA_regression
        
        Runs a PCA on the noise components and removes these from the data.
        3. Fits this to self.data
        4. Takes the residuals of the regression.
        5. Adds back in the intercept.
        6. Returns the result in self.denoised_data
     
        Returns
        ----------
       
        betas: Ordinary least squares beta coefficients for respective noise components
        yhat: predictions (dot products) 
        rsq: root mean squared differences
        denoised_data: residuals (now cleaned data)
        """
        
        # -- MD ADDED
        # data = self.lf_filter.filter_data(data.T).T
        # --


        # Add row of 1s to the design matrix.
        self.dm = np.vstack([np.ones(self.pca_comps.shape[0]), self.pca_comps.T]).T
        # Do OLS
        betas = np.linalg.lstsq(self.dm, data.T,rcond=-1)[0]

        # Predictions are dot product of dm and betas.
        yhat = np.dot(self.dm, betas).T

        # # Get model rsq
        # rsq = np.divide(
        #     (data - self.yhat).var(-1), data.var(-1), out=np.zeros_like(self.data.var(-1)), where=self.data.var(-1) != 0
        # )

        # Get residuals
        resid=data-yhat

        # Add back in intercept
        resid+= np.nanmean(data,axis=-1)[:,np.newaxis]
        return resid #, betas
        
class SGFilter:
    
    def __init__(self,polyorder=3, deriv=0, window_length = 347,tr=1):
        """ Applies a savitsky-golay filter to continuous data.

        Fits a savitsky-golay filter to 2D data and subtracts the
        fitted data from the original data to effectively remove low-frequency
        signals.

        polyorder : int (default: 3)
        Order of polynomials to use in filter.
        deriv : int (default: 0)
        Number of derivatives to use in filter.
        window_length : int (default: 347)
        Window length in seconds.

        """
        self.polyorder=polyorder
        self.deriv=deriv
        self.window_length=window_length
        print('****************')
        print(f'sg_window {window_length}')
        self.tr=tr
        
    def filter_data(self,data):
        """
        Parameters
        ----------
        data : numpy arrat of ts data (vertices,time)
       
        Returns
        -------
        out_file : str
            Absolute path to filtered nifti-file.
        """

        # TR must be in seconds
        if self.tr < 0.01:
            self.tr = np.round(self.tr * 1000, decimals=3)
        if self.tr > 20:
            self.tr = self.tr / 1000.0

        window = int(self.window_length / self.tr)

        # Window must be odd
        if window % 2 == 0:
            window += 1

        data_filt = savgol_filter(data, window_length=window, polyorder=self.polyorder,
                                  deriv=self.deriv, axis=-1, mode='nearest')

        data_filt = data - data_filt + data.mean(axis=-1)[:, np.newaxis]

        return data_filt