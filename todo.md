# TODO

## 0. Big picture — sustainability
Think about how easy is it to slot in a new pipeline step? Similarly, how easy is it to pull out subcomponents of each step? Should some code be refactored to the cvl_utils package? Consider defining the scope of the pipeline.

## 1. Bugs

Bugs which actively need fixing

### `s02_coreg.py` — MCFLIRT returns no motion matrices
- For one run/subject, MCFLIRT thinks the data is perfectly still and produces
  no rotation/translation matrices.
- Likely downstream of a coregistration failure, not MCFLIRT itself.
- Possibly related to [fmriprep#2360](https://github.com/nipreps/fmriprep/issues/2360#issuecomment-819898526).
- Can be fixed with manual coregistration...
- Next steps:
  - [ ] Flag this error during coreg & crash
  - [ ] Add manual coregistration helper (easy way to set a manual starting point) + explainer

---

## 2. Essential features

Crucial aspects of the pipeline not yet implemented

### **BIDSify**

- Potentially take a zzSTIMNOTES.rtf file and a dcm folder, output BIDS(-ish) niftis?
- Or use more standard tools? 

### **Logging** 
- every stage should write its stdout/stderr to a log file,
  not just print to console.

### **Automatic job chaining** 

- allow a job to auto-start once its dependency finishes, so the full pipeline can run unattended. 
- Should record what ran and when.
- Via cluster

### **QC + reports** 
- automated PDF report per stage.

---

## 3. Decisions - required

Decisions which need to be made for pipeline to be complete.  

### **Confounds (`s04_confounds.py`)** 

- the non-PCA version currently performs better. Is that because fMRIPrep's confounds aren't adding useful signal, or
is the PCA implementation wrong?

- [ ] Test to find the best config: number of regressors, which regressors, filtering
- [ ] Use the `.yml` to control these settings, with notes on what to include/why

### **CF fitting (`s03_cf_prfpy.py`)** 

- how should concatenation and baselining across runs be implemented?

---


## 4. Backlog / to consider

Possible features / wishlists. Need to decide whether they will improve the pipeline 

### SDC - alternatives?
- add an alternative SDC correction path using FSL `topup` + fieldmaps, rather than relying on the current approach alone?

### pRF + Connective fields
- [ ] Batching options across ROIs
- [ ] Option to morph ROIs (for all analyses)

### GLM support
Add a GLM path analogous to the pRF pipeline — decide what inputs it should take & default BIDS-structure? 

### Project-specific config files

Where to locate? Currently kept inside ```${PIPELINE_DIR}```; but could be inside `code/` folder of each BIDS directory?

Option to take "example" configs as default if project not specified? 

