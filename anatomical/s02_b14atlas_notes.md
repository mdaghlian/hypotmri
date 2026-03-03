# s03 generate b14 atlas for the freesurfer output
activate correct conda environments, specify bids_dir & subject to run
```bash
conda activate b14
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
FS_DIR=$BIDS_DIR/derivatives/freesurfer
python s02_b14atlas.py sub-hp01 --fs_dir $FS_DIR
```
