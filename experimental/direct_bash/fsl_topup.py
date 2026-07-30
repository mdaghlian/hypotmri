#!/usr/bin/env python3
"""
FSL Topup Correction Script for BIDS Data (Multi-Run Support)
This script applies topup correction to fMRI data using opposite phase-encoded EPI scans
Supports processing multiple runs in a single execution

Uses fslpy for FSL operations in Python

Usage:
    python apply_topup_correction.py <bids_dir> <output_dir> <subject> <session> <task> [runs]

Examples:
    # Single run
    python apply_topup_correction.py /data/bids /data/derivatives/topup sub-01 ses-01 rest run-01
    
    # Multiple runs
    python apply_topup_correction.py /data/bids /data/derivatives/topup sub-01 ses-01 rest run-01,run-02,run-03
    
    # All runs
    python apply_topup_correction.py /data/bids /data/derivatives/topup sub-01 ses-01 rest all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil

# FSL imports
import fsl.wrappers as fsl
from fsl.data.image import Image
import nibabel as nib


class PhaseEncodingConverter:
    """Convert between BIDS phase encoding direction and FSL vector format"""
    
    PE_MAPPING = {
        'j-': [0, -1, 0],
        'j': [0, 1, 0],
        'i-': [-1, 0, 0],
        'i': [1, 0, 0],
        'k-': [0, 0, -1],
        'k': [0, 0, 1]
    }
    
    @classmethod
    def to_vector(cls, pe_direction: str) -> List[int]:
        """Convert BIDS PE direction to FSL vector"""
        if pe_direction not in cls.PE_MAPPING:
            raise ValueError(f"Unknown phase encoding direction: {pe_direction}")
        return cls.PE_MAPPING[pe_direction]


class BIDSFileManager:
    """Manage BIDS file discovery and naming"""
    
    def __init__(self, bids_dir: Path, subject: str, session: str):
        self.bids_dir = Path(bids_dir)
        self.subject = subject
        self.session = session
        self.func_dir = self.bids_dir / subject / session / "func"
        
    def find_bold_files(self, task: str, run_spec: str = "all") -> List[Path]:
        """Find BOLD files based on task and run specification"""
        if run_spec == "all":
            # Find all runs for this task
            bold_files = sorted(self.func_dir.glob(
                f"{self.subject}_{self.session}_task-{task}_run-*_bold.nii*"
            ))
            
            # If no runs found, try without run label
            if not bold_files:
                bold_files = list(self.func_dir.glob(
                    f"{self.subject}_{self.session}_task-{task}_*bold.nii*"
                ))
                # Filter out files with run label
                bold_files = [f for f in bold_files if 'run-' not in f.name]
            
            return bold_files
        else:
            # Process specific runs
            runs = [r.strip() for r in run_spec.split(',')]
            bold_files = []
            for run in runs:
                pattern = f"{self.subject}_{self.session}_task-{task}_{run}_bold.nii*"
                found = list(self.func_dir.glob(pattern))
                if found:
                    bold_files.extend(found)
                else:
                    print(f"Warning: BOLD file not found for {run}, skipping...")
            
            return bold_files
    
    def find_epi_file(self, task: str, run_label: Optional[str] = None) -> Optional[Path]:
        """Find corresponding EPI file"""
        if run_label:
            pattern = f"{self.subject}_{self.session}_task-{task}_{run_label}_*epi.nii*"
        else:
            pattern = f"{self.subject}_{self.session}_task-{task}_*epi.nii*"
        
        epi_files = list(self.func_dir.glob(pattern))
        if run_label is None:
            # Filter out files with run label
            epi_files = [f for f in epi_files if 'run-' not in f.name]
        
        return epi_files[0] if epi_files else None
    
    def find_sbref_file(self, task: str, run_label: Optional[str] = None) -> Optional[Path]:
        """Find corresponding SBREF file"""
        if run_label:
            pattern = f"{self.subject}_{self.session}_task-{task}_{run_label}_*sbref.nii*"
        else:
            pattern = f"{self.subject}_{self.session}_task-{task}_*sbref.nii*"
        
        sbref_files = list(self.func_dir.glob(pattern))
        if run_label is None:
            # Filter out files with run label
            sbref_files = [f for f in sbref_files if 'run-' not in f.name]
        
        return sbref_files[0] if sbref_files else None
    
    @staticmethod
    def extract_run_label(bold_file: Path) -> Optional[str]:
        """Extract run label from filename"""
        import re
        match = re.search(r'run-(\d+)', bold_file.name)
        return f"run-{match.group(1)}" if match else None
    
    @staticmethod
    def get_base_name(file_path: Path) -> str:
        """Get base filename without extensions"""
        name = file_path.name
        # Remove .nii.gz or .nii
        if name.endswith('.nii.gz'):
            return name[:-7]
        elif name.endswith('.nii'):
            return name[:-4]
        return name


class MetadataReader:
    """Read and parse BIDS JSON metadata"""
    
    @staticmethod
    def read_json(nifti_path: Path) -> Dict:
        """Read JSON sidecar for a NIfTI file"""
        # Try both .nii.gz and .nii extensions
        json_path = Path(str(nifti_path).replace('.nii.gz', '.json').replace('.nii', '.json'))
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON sidecar not found: {json_path}")
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_phase_encoding_info(nifti_path: Path) -> Tuple[str, float]:
        """Extract phase encoding direction and total readout time"""
        metadata = MetadataReader.read_json(nifti_path)
        
        pe_dir = metadata.get('PhaseEncodingDirection')
        trt = metadata.get('TotalReadoutTime')
        
        if pe_dir is None:
            raise ValueError(f"PhaseEncodingDirection not found in {nifti_path}")
        if trt is None:
            raise ValueError(f"TotalReadoutTime not found in {nifti_path}")
        
        return pe_dir, float(trt)


class TopupProcessor:
    """Handle topup correction workflow"""
    
    def __init__(self, work_dir: Path, output_dir: Path):
        self.work_dir = Path(work_dir)
        self.output_dir = Path(output_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_reference_images(
        self,
        bold_file: Path,
        epi_file: Path,
        sbref_file: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """Prepare reference images for topup"""
        print("\nStep 1: Preparing reference images...")
        
        # BOLD reference
        bold_ref = self.work_dir / "bold_ref.nii.gz"
        if sbref_file:
            print(f"  Using SBREF as BOLD reference")
            shutil.copy(sbref_file, bold_ref)
        else:
            print(f"  Extracting first volume from BOLD")
            fsl.fslroi(str(bold_file), str(bold_ref), 0, 1)
        
        # EPI reference
        print(f"  Extracting first volume from EPI")
        epi_ref = self.work_dir / "epi_ref.nii.gz"
        fsl.fslroi(str(epi_file), str(epi_ref), 0, 1)
        
        return bold_ref, epi_ref
    
    def merge_references(self, bold_ref: Path, epi_ref: Path) -> Path:
        """Merge reference images for topup"""
        print("\nStep 2: Merging images for topup...")
        merged = self.work_dir / "merged_b0.nii.gz"
        fsl.fslmerge(str(merged), str(bold_ref), str(epi_ref), t=True)
        return merged
    
    def create_acqparams(
        self,
        bold_pe: str,
        bold_trt: float,
        epi_pe: str,
        epi_trt: float
    ) -> Path:
        """Create acquisition parameters file"""
        print("\nStep 3: Creating acquisition parameters file...")
        
        bold_vec = PhaseEncodingConverter.to_vector(bold_pe)
        epi_vec = PhaseEncodingConverter.to_vector(epi_pe)
        
        acqparams = self.work_dir / "acqparams.txt"
        with open(acqparams, 'w') as f:
            f.write(f"{bold_vec[0]} {bold_vec[1]} {bold_vec[2]} {bold_trt}\n")
            f.write(f"{epi_vec[0]} {epi_vec[1]} {epi_vec[2]} {epi_trt}\n")
        
        print("  Acquisition parameters:")
        with open(acqparams, 'r') as f:
            print(f"    {f.read()}", end='')
        
        return acqparams
    
    def run_topup(self, merged: Path, acqparams: Path) -> Path:
        """Run FSL topup"""
        print("\nStep 4: Running topup (this may take several minutes)...")
        
        topup_basename = self.work_dir / "topup_results"
        
        # Run topup using fslpy wrapper
        fsl.topup(
            imain=str(merged),
            datain=str(acqparams),
            config='b02b0.cnf',
            out=str(topup_basename),
            fout=str(topup_basename) + '_field',
            iout=str(topup_basename) + '_unwarped',
            verbose=True
        )
        
        print("  Topup completed successfully")
        return topup_basename
    
    def apply_topup(
        self,
        bold_file: Path,
        acqparams: Path,
        topup_basename: Path,
        output_name: str
    ) -> Path:
        """Apply topup correction to BOLD data"""
        print("\nStep 5: Applying topup correction to BOLD data...")
        
        corrected_bold = self.output_dir / f"{output_name}_desc-topup_bold.nii.gz"
        
        fsl.applytopup(
            imain=str(bold_file),
            inindex=1,
            datain=str(acqparams),
            topup=str(topup_basename),
            out=str(corrected_bold),
            method='jac',
            verbose=True
        )
        
        print(f"  Applied topup to BOLD: {corrected_bold}")
        return corrected_bold
    
    def save_outputs(
        self,
        topup_basename: Path,
        acqparams: Path,
        output_base_name: str,
        bold_file: Path,
        epi_file: Path,
        bold_pe: str,
        bold_trt: float
    ) -> Dict[str, Path]:
        """Save all output files"""
        print("\nStep 6: Organizing outputs...")
        
        outputs = {}
        
        # Warp field (displacement field)
        warp_output = self.output_dir / f"{output_base_name}_desc-topup_warp.nii.gz"
        shutil.copy(f"{topup_basename}_field.nii.gz", warp_output)
        outputs['warp'] = warp_output
        
        # Field coefficients (spline coefficients)
        coef_output = self.output_dir / f"{output_base_name}_desc-topup_fieldcoef.nii.gz"
        shutil.copy(f"{topup_basename}_fieldcoef.nii.gz", coef_output)
        outputs['fieldcoef'] = coef_output
        
        # Unwarped reference image
        unwarp_output = self.output_dir / f"{output_base_name}_desc-topup_unwarped.nii.gz"
        shutil.copy(f"{topup_basename}_unwarped.nii.gz", unwarp_output)
        outputs['unwarped'] = unwarp_output
        
        # Movement parameters (transformation matrix)
        movpar_source = Path(f"{topup_basename}_movpar.txt")
        if movpar_source.exists():
            movpar_output = self.output_dir / f"{output_base_name}_desc-topup_movpar.txt"
            shutil.copy(movpar_source, movpar_output)
            outputs['movpar'] = movpar_output
            print(f"  Movement parameters: {movpar_output}")
        else:
            print(f"  Warning: Movement parameters file not found")
            outputs['movpar'] = None
        
        # Acquisition parameters
        acqparam_output = self.output_dir / f"{output_base_name}_desc-topup_acqparams.txt"
        shutil.copy(acqparams, acqparam_output)
        outputs['acqparams'] = acqparam_output
        
        return outputs
    
    def create_json_sidecar(
        self,
        corrected_bold: Path,
        bold_file: Path,
        epi_file: Path,
        bold_pe: str,
        bold_trt: float,
        outputs: Dict[str, Path]
    ):
        """Create comprehensive JSON sidecar"""
        metadata = {
            "Description": "BOLD data corrected for susceptibility-induced distortions using FSL topup",
            "Sources": [
                bold_file.name,
                epi_file.name
            ],
            "TopupConfig": "b02b0.cnf",
            "ApplytopupMethod": "jac",
            "PhaseEncodingDirection": bold_pe,
            "TotalReadoutTime": bold_trt,
            "ProcessingOutputs": {
                "CorrectedBOLD": corrected_bold.name,
                "WarpField": outputs['warp'].name,
                "FieldCoefficients": outputs['fieldcoef'].name,
                "UnwarpedReference": outputs['unwarped'].name,
                "MovementParameters": outputs['movpar'].name if outputs['movpar'] else None,
                "AcquisitionParameters": outputs['acqparams'].name
            }
        }
        
        json_path = corrected_bold.parent / f"{corrected_bold.stem.replace('.nii', '')}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)


def process_run(
    bold_file: Path,
    epi_file: Path,
    sbref_file: Optional[Path],
    work_dir: Path,
    output_dir: Path,
    run_number: int,
    total_runs: int
) -> bool:
    """Process a single run"""
    print("\n" + "=" * 42)
    print(f"Processing run {run_number}/{total_runs}")
    print("=" * 42)
    
    try:
        # Get run label
        run_label = BIDSFileManager.extract_run_label(bold_file)
        if run_label:
            print(f"Run: {run_label}")
        else:
            print("Run: (no run label)")
        
        # Get base name
        base_name = BIDSFileManager.get_base_name(bold_file)
        
        print("\nFound files:")
        print(f"  BOLD: {bold_file}")
        print(f"  EPI: {epi_file}")
        if sbref_file:
            print(f"  SBREF: {sbref_file}")
        
        # Read metadata
        bold_pe, bold_trt = MetadataReader.get_phase_encoding_info(bold_file)
        epi_pe, epi_trt = MetadataReader.get_phase_encoding_info(epi_file)
        
        print("\nPhase encoding parameters:")
        print(f"  BOLD PE direction: {bold_pe}")
        print(f"  EPI PE direction: {epi_pe}")
        print(f"  BOLD Total Readout Time: {bold_trt}")
        print(f"  EPI Total Readout Time: {epi_trt}")
        
        # Initialize processor
        processor = TopupProcessor(work_dir, output_dir)
        
        # Process
        bold_ref, epi_ref = processor.prepare_reference_images(bold_file, epi_file, sbref_file)
        merged = processor.merge_references(bold_ref, epi_ref)
        acqparams = processor.create_acqparams(bold_pe, bold_trt, epi_pe, epi_trt)
        topup_basename = processor.run_topup(merged, acqparams)
        corrected_bold = processor.apply_topup(bold_file, acqparams, topup_basename, base_name)
        outputs = processor.save_outputs(
            topup_basename, acqparams, base_name, bold_file, epi_file, bold_pe, bold_trt
        )
        processor.create_json_sidecar(
            corrected_bold, bold_file, epi_file, bold_pe, bold_trt, outputs
        )
        
        print("\n" + "=" * 42)
        print(f"Run {run_number} completed successfully!")
        print("=" * 42)
        print("\nOutput files:")
        print(f"  Corrected BOLD: {corrected_bold}")
        print(f"  Warp field: {outputs['warp']}")
        print(f"  Field coefficients: {outputs['fieldcoef']}")
        print(f"  Unwarped reference: {outputs['unwarped']}")
        if outputs['movpar']:
            print(f"  Movement parameters: {outputs['movpar']}")
        print()
        
        return True
        
    except Exception as e:
        print(f"\nError processing run: {e}")
        print("Skipping this run...\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Apply FSL topup correction to BIDS fMRI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single run
  %(prog)s /data/bids /data/derivatives/topup sub-01 ses-01 rest run-01
  
  # Process multiple specific runs
  %(prog)s /data/bids /data/derivatives/topup sub-01 ses-01 rest run-01,run-02,run-03
  
  # Process all runs
  %(prog)s /data/bids /data/derivatives/topup sub-01 ses-01 rest all
        """
    )
    
    parser.add_argument('bids_dir', type=str, help='BIDS dataset directory')
    parser.add_argument('output_dir', type=str, help='Output directory for derivatives')
    parser.add_argument('subject', type=str, help='Subject ID (e.g., sub-01)')
    parser.add_argument('session', type=str, help='Session ID (e.g., ses-01)')
    parser.add_argument('task', type=str, help='Task name (e.g., rest)')
    parser.add_argument('runs', type=str, nargs='?', default='all',
                       help='Run specification: run-01, run-01,run-02, or "all" (default: all)')
    parser.add_argument('--keep-work', action='store_true',
                       help='Keep working directory (default: remove)')
    
    args = parser.parse_args()
    
    print("=" * 42)
    print("FSL Topup Correction (Multi-Run)")
    print(f"Subject: {args.subject}")
    print(f"Session: {args.session}")
    print(f"Task: {args.task}")
    print("=" * 42)
    
    # Initialize file manager
    file_manager = BIDSFileManager(args.bids_dir, args.subject, args.session)
    
    # Find BOLD files
    bold_files = file_manager.find_bold_files(args.task, args.runs)
    
    if not bold_files:
        print(f"Error: No BOLD files found for {args.subject}_{args.session}_task-{args.task}")
        sys.exit(1)
    
    print(f"\nFound {len(bold_files)} run(s) to process")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.subject / args.session / "func"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each run
    successful = 0
    for i, bold_file in enumerate(bold_files, 1):
        run_label = file_manager.extract_run_label(bold_file)
        epi_file = file_manager.find_epi_file(args.task, run_label)
        
        if not epi_file:
            print(f"\nError: EPI file not found for run {i}")
            print("Skipping this run...")
            continue
        
        sbref_file = file_manager.find_sbref_file(args.task, run_label)
        
        # Setup work directory
        if run_label:
            work_dir = output_dir / f"work_{run_label}"
        else:
            work_dir = output_dir / "work"
        
        # Process
        if process_run(bold_file, epi_file, sbref_file, work_dir, output_dir, i, len(bold_files)):
            successful += 1
        
        # Clean up work directory if requested
        if not args.keep_work and work_dir.exists():
            shutil.rmtree(work_dir)
    
    print("\n" + "=" * 42)
    print("All runs completed!")
    print("=" * 42)
    print(f"Processed {successful}/{len(bold_files)} run(s) successfully")
    print(f"Output directory: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()