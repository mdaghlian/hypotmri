proj_path="/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot"
export SUBJECTS_DIR=$proj_path/derivatives/freesurfer

recon-all -subjid sub-hp01  \
    -i $proj_path/sub-hp01/ses-01/anat/sub-hp01_ses-01_acq-MPRAGE_T1w.nii \
    -all -parallel -openmp 8 


# Fmriprep

mamba create -n fmriprep001 python
conda activate fmriprep001 

python -m pip install fmriprep-docker


cd /Users/marcusdaghlian/projects/pilot-clean-link/derivatives/BIDS

# Create JSON files for all BOLD runs (replace 2.0 with your actual TR)
for bold in sub-*/func/*_bold.nii*; do
  json="${bold%.nii*}.json"
  if [ ! -f "$json" ]; then
    echo '{"RepetitionTime": 3.0, "TaskName": "colbw"}' > "$json"
  fi
done

# Run fmriprep
fmriprep-docker \
  /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/ \
  /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/fmriprep \
  participant \
  --participant-label sub-hp01 \
  --fs-subjects-dir  /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/freesurfer \
  --output-spaces func T1w fsnative \
  --fs-license-file /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/code/license.txt \
  --skip-bids-validation \
  --omp-nthreads 8 \
  -w /Users/marcusdaghlian/projects/dp-clean-link/240522NG/BIDSWF