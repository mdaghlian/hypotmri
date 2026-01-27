# To install for mac
mamba create --name tf_gpu001 python=3.10
conda activate tf_gpu001
unset DYLD_LIBRARY_PATH

pip install --upgrade pip
pip install --upgrade tensorflow-macos tensorflow-metal
pip install --no-deps tensorflow-probability==0.24.0
pip install tf-keras==2.16.0



cd braincoder 
pip install -e . 


    
pybest  --subject 01 --out-dir derivatives/pybest20cNohighpass/ --space func  \
    --n-cpus 8 derivatives/fmriprep --save-all --verbose DEBUG \
    --n-comps 20 --high-pass 0.0