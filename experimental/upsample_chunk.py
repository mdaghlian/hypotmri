import argparse
import nibabel as nib
import numpy as np
from nilearn.image import resample_img, iter_img
from joblib import Parallel, delayed
from tqdm import tqdm
import os

def resample_single_volume(vol, target_affine):
    """Worker function to resample a single 3D volume."""
    return resample_img(vol, target_affine=target_affine, interpolation='nearest')

def main():
    parser = argparse.ArgumentParser(description="Parallel 4D fMRI Upsampling (NN)")
    parser.add_argument("-i", "--input", required=True, help="Input 4D .nii.gz file")
    parser.add_argument("-o", "--output", required=True, help="Output filename")
    parser.add_argument("-v", "--voxsize", type=float, default=0.5, help="Isotropic voxel size (default 0.5)")
    parser.add_argument("-n", "--n_cpus", type=int, default=-1, help="Number of CPUs (-1 for all)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # 1. Load header and calculate geometry
    print(f"--- Loading {args.input} ---")
    img_4d = nib.load(args.input)
    affine = img_4d.affine
    header = img_4d.header
    
    # Check that it's actually 4D
    if len(img_4d.shape) != 4:
        raise ValueError(f"Expected 4D image, got shape {img_4d.shape}")
    
    n_volumes = img_4d.shape[3]
    
    # Create target affine with more robust handling
    # Preserve the rotation/translation but update voxel size
    target_affine = affine.copy()
    
    # More robust: scale the first 3 columns (which encode voxel directions)
    for i in range(3):
        # Normalize and rescale each direction vector
        current_voxel_size = np.linalg.norm(affine[:3, i])
        target_affine[:3, i] = (affine[:3, i] / current_voxel_size) * args.voxsize
    
    # 2. Run Parallel Processing
    print(f"--- Upsampling {n_volumes} volumes using {args.n_cpus} CPUs ---")
    print(f"    Original voxel size: {nib.affines.voxel_sizes(affine)}")
    print(f"    Target voxel size: ({args.voxsize}, {args.voxsize}, {args.voxsize})")
    
    # We use joblib to parallelize the resample_img task
    results = Parallel(n_jobs=args.n_cpus)(
        delayed(resample_single_volume)(vol, target_affine) 
        for vol in tqdm(iter_img(img_4d), total=n_volumes, desc="Processing")
    )
    
    # 3. Concatenate and Save
    print("--- Concatenating and saving (this requires significant RAM) ---")
    # Grab data from all resampled objects
    upsampled_data = np.stack([res.get_fdata() for res in results], axis=-1)
    
    # Create new NIfTI object with clean header
    new_header = header.copy()
    new_header.set_zooms((args.voxsize, args.voxsize, args.voxsize, header.get_zooms()[3]))
    
    new_img = nib.Nifti1Image(upsampled_data, target_affine, header=new_header)
    
    new_img.to_filename(args.output)
    print(f"--- Successfully saved to {args.output} ---")
    print(f"    Output shape: {new_img.shape}")

if __name__ == "__main__":
    main()