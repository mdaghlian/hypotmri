# Bug list
#### ```s02_coreg.py``` 
- for one run & one subject - MCFLIRT does not provide any rotation / translation matrices (thinks that it is perfectly still). 
- Likely related to this [issue](https://github.com/nipreps/fmriprep/issues/2360#issuecomment-819898526).
- Potential soluition: change cost function? Check xform / qform headers? 

#### ```s04_confounds.py```
- Unclear when to filter: filter data & confounds, then GLM? Filter confounds then GLM, then filter again?...
- Which components to include? acompcor; edge_comp_cor? How many? 
- Check whether motion derivatives calculated from McFlirt are reasonable
- filtering in prfpy? 


# Feature list:
---
#### [ ] automatic submissions. 
Allow jobs to automatically start when another finishes (so whole pipeline can be run through)

---
#### [ ] ```s02_coreg.py```: fallback for no "bref" 

(make from the bold mean, or first volume)

---
#### [ ] Quality control for SDC + coregistration
Need to think about what would be most helpful. Carpet plots? Movies, concatenated?


---
#### [ ] Connective field implementation
(using prfpy)


---
#### [ ] big picture - sustainability
Think about how easy it is to slot in new steps

(we do actually want to be useful)

