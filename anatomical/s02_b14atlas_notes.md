# s03 generate b14 atlas for the freesurfer output

[1] Create conda env with neuropythy installed (if not already created)
```bash
mamba create -n npythflat001 python
conda activate npythflat001
pip install neuropythy
```

[2] activate & run 
```bash
conda activate npythflat001
PROJ_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
SUBJECTS_DIR="${PROJ_DIR}/derivatives/freesurfer"
python s02_b14atlas.py sub-hp01 --fsdir $SUBJECTS_DIR
```
