# Bug list / to check list
#### ```s02_coreg.py``` 
- for one run & one subject - MCFLIRT does not provide any rotation / translation matrices (thinks that it is perfectly still).
- Probably driven by coregistration failing. Need to introduce manual helpers - and make it easy to do a manual coregistration as a starting point.  
- Likely related to this [issue](https://github.com/nipreps/fmriprep/issues/2360#issuecomment-819898526).
- Potential soluition: change cost function? Check xform / qform headers? 

#### ```s04_confounds.py```
Version without PCA seems to work better for now. Is this because fMRIprep confounds are not adding much? 
- Needs testing to find best implementation: number of regressors; which regressors; filtering;
- Use the .yml to control settings - add notes on what to include etc. 


#### ```s03_cf_prfpy.py```
- How to best implement concatenation & baselining

# Feature list:
---
#### [ ] Logging. 
Run all stages such that everything that is output to command line is stored as a log file

#### [ ] SDC more versions
- Use fsl topup? And use fieldmaps 


#### [ ] automatic submissions. 
Allow jobs to automatically start when another finishes (so whole pipeline can be run through)

---
#### [ ] Quality control for SDC + coregistration
Need to think about what would be most helpful. Carpet plots? Movies, concatenated?

---
#### [ ] QC + reports for all steps? 
Automated .pdf files produced? 

Especially for pRF + CF analyses



---
#### [ ] big picture - sustainability
Think about how easy it is to slot in new steps

(we do actually want to be useful)

