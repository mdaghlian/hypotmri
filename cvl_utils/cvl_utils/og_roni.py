from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import nibabel as nib
import scipy
from scipy.signal import savgol_filter
    

class PCA_denoiser():
    """
    PCA_denoiser
    Class for preparing nuisance regressors and then running PCA denoising on fMRIprep output using confounds.tsv file  
    
    """
    
    def __init__(self,confounds,data,lf_filter):
        """
        Initialise the PCA_denoiser class.
        
        Input arguments
        ----------
        confounds: pandas dataframe containing noise components
        data: numpy arrat of ts data (vertices,time)
        lf_filter: low frequency filter to apply on the pcs
        
        Parameters
        ----------
        
        self.csvpath: the path to the noise components tsv file.
        self.datapath: the path to the fMRIprepped data.
        self.load_csv: reads the .tsv file using pandas
        self.load_data: loads the fMRIprepped data using nibabel
        
        """
        
        self.data=data
        self.confound_frame=confounds
        self.lf_filter=lf_filter


    def subset_frame(self,vars):
        """ subset_frame
        
        Subsets the noise components to only include variables defined by the user
        
        Returns
        ----------
        self.subsetted_frame: subsetted noise components.
        
        """
        self.subsetted_frame=self.confound_frame[vars]

    def prepare_frame(self):
        """ prepare_frame
        
        Prepares the noise regressors.
        1.   Converts this subsetted frame to a numpy array. 
        2.   Converts NAs for each regressor to the median.
        3. Zscores each regressor over time.
        4. Returns the result in self.prepared_array
     
        Returns
        ----------
        self.nuissance_array: a numpy array of nuisance regressors with NANs replaced with medians.
        self.prepared_array: a z-scored numpy array of the nuisance regressors
        """
        self.nuissance_array=np.array(self.subsetted_frame)
        medians=np.nanmedian(self.nuissance_array,axis=0)
        for c,v in enumerate(medians):
            self.nuissance_array[:,c][np.isnan(self.nuissance_array[:,c])]=medians[c]
        self.prepared_array=scipy.stats.zscore(self.nuissance_array)
    
    def PCA_regression(self,ncomps):
        
        """ PCA_regression
        
        Runs a PCA on the noise components and removes these from the data.
        1. Fits a PCA of n components to self.prepared_array
        1.5. Apply filter to the pca_comps
        2. Creates a design matrix with these regressors and an intercept.
        3. Fits this to self.data
        4. Takes the residuals of the regression.
        5. Adds back in the intercept.
        6. Returns the result in self.denoised_data
     
        Returns
        ----------
       
        self.pca_comps: PCA output of 
        self.dm: design matrix (including intercept)
        self.betas: Ordinary least squares beta coefficients for respective noise components
        self.yhat: predictions (dot products) 
        self.rsq: root mean squared differences
        self.resid: residuals after removing PCA components
        self.denoised_data: residuals (now cleaned data)
        """
        # Fit PCA to the prepared array
        self.pca = PCA(n_components=ncomps)  
        
        self.pca_comps = self.pca.fit_transform(self.prepared_array)
        self.pca_comps = self.lf_filter.filter_data(self.pca_comps.T).T
        
        # Add row of 1s to the design matrix.
        self.dm = np.vstack([np.ones(self.pca_comps.shape[0]), self.pca_comps.T]).T

        # Do OLS
        self.betas = np.linalg.lstsq(self.dm, self.data.T,rcond=-1)[0]

        # Predictions are dot product of dm and betas.
        self.yhat = np.dot(self.dm, self.betas).T

        # Get model rsq
        self.rsq = np.divide(
            (self.data - self.yhat).var(-1), self.data.var(-1), out=np.zeros_like(self.data.var(-1)), where=self.data.var(-1) != 0
        )

        # Get residuals
        self.resid=self.data-self.yhat

        # Add back in intercept
        self.resid+= np.nanmean(self.data,axis=-1)[:,np.newaxis]
        self.denoised_data=self.resid
        
class SGFilter:
    
    def __init__(self,polyorder=3, deriv=0, window_length = 120,tr=1):
        """ Applies a savitsky-golay filter to continuous data.

        Fits a savitsky-golay filter to 2D data and subtracts the
        fitted data from the original data to effectively remove low-frequency
        signals.

        polyorder : int (default: 3)
        Order of polynomials to use in filter.
        deriv : int (default: 0)
        Number of derivatives to use in filter.
        window_length : int (default: 120)
        Window length in seconds.

        """
        self.polyorder=polyorder
        self.deriv=deriv
        self.window_length=window_length
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
            self.tr = np.round(tr * 1000, decimals=3)
        if self.tr > 20:
            self.tr = self.tr / 1000.0

        window = int(self.window_length / self.tr)

        # Window must be odd
        if window % 2 == 0:
            window += 1

        data_filt = savgol_filter(data, window_length=window, polyorder=self.polyorder,
                                  deriv=self.deriv, axis=1, mode='nearest')

        data_filt = data - data_filt + data_filt.mean(axis=-1)[:, np.newaxis]

        return data_filt