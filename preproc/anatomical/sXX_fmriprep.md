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