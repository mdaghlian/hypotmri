# TOPUP 
Apply SDC, using either fsl or afni

AFNI is based on image registration method so 

```bash
bash s01_topup.sh --bids_dir ~/projects/dp-clean-link/240522NG/hypot/ --output_dir ~/projects/dp-clean-link/240522NG/hypot/derivatives/sf1_topup --sub sub-hp01 --ses ses-01 --task pRFLE 
```

```bash
bash s01B_topup_AFNI.sh --bids_dir ~/projects/dp-clean-link/240522NG/hypot/ --output_dir ~/projects/dp-clean-link/240522NG/hypot/derivatives/sf1_topupAFNI --sub sub-hp01 --ses ses-01 --task pRFLE 
```