#!/usr/bin/env python3
"""
Add IntendedFor field to fieldmap JSON sidecars in BIDS dataset.
Matches fmap files to func files based on subject, session, task, and run.
"""

import json
import os
from pathlib import Path
import argparse


def find_matching_func(fmap_path, bids_root):
    """
    Find the functional file that matches a given fieldmap.
    
    Parameters:
    -----------
    fmap_path : Path
        Path to the fieldmap file
    bids_root : Path
        Root directory of the BIDS dataset
    
    Returns:
    --------
    str or None
        Relative path to matching functional file, or None if not found
    """
    # Parse fieldmap filename - handle both .nii and .nii.gz
    fmap_name = fmap_path.name
    if fmap_name.endswith('.nii.gz'):
        fmap_name = fmap_name.replace('.nii.gz', '')
    elif fmap_name.endswith('.nii'):
        fmap_name = fmap_name.replace('.nii', '')
    
    parts = fmap_name.split('_')
    
    # Extract BIDS entities
    entities = {}
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)
            entities[key] = value
    
    # Build expected functional filename base (without extension)
    func_parts = []
    for entity in ['sub', 'ses', 'task', 'run']:
        if entity in entities:
            func_parts.append(f"{entity}-{entities[entity]}")
    func_parts.append('bold')
    func_filename_base = '_'.join(func_parts)
    
    # Construct functional file path - try both .nii.gz and .nii
    sub = entities.get('sub')
    ses = entities.get('ses')
    
    if ses:
        func_dir = bids_root / f"sub-{sub}" / f"ses-{ses}" / "func"
        relative_dir = f"ses-{ses}/func"
    else:
        func_dir = bids_root / f"sub-{sub}" / "func"
        relative_dir = f"func"
    
    # Try both extensions
    for ext in ['.nii.gz', '.nii']:
        func_filename = func_filename_base + ext
        func_path = func_dir / func_filename
        relative_path = f"{relative_dir}/{func_filename}"
        
        if func_path.exists():
            return relative_path
    
    # If neither exists, warn user
    print(f"  WARNING: Expected functional file not found: {func_dir}/{func_filename_base}{{.nii.gz,.nii}}")
    return None


def add_intendedfor(bids_root, subject, session, dry_run=False):
    """
    Add IntendedFor field to all fieldmap JSON sidecars for a specific subject and session.
    
    Parameters:
    -----------
    bids_root : str or Path
        Root directory of the BIDS dataset
    subject : str
        Subject ID (without 'sub-' prefix)
    session : str
        Session ID (without 'ses-' prefix)
    dry_run : bool
        If True, only print what would be done without modifying files
    """
    bids_root = Path(bids_root)
    
    if not bids_root.exists():
        raise ValueError(f"BIDS root directory does not exist: {bids_root}")
    
    # Construct path to subject/session fmap directory
    sub_ses_path = bids_root / f"sub-{subject}" / f"ses-{session}"
    fmap_dir = sub_ses_path / "fmap"
    
    if not fmap_dir.exists():
        raise ValueError(f"Fieldmap directory does not exist: {fmap_dir}")
    
    # Find all fmap JSON files for this subject/session
    fmap_jsons = list(fmap_dir.glob('*_epi.json'))
    
    if not fmap_jsons:
        print(f"No fieldmap JSON files found in {fmap_dir}")
        return
    
    print(f"Processing subject: sub-{subject}, session: ses-{session}")
    print(f"Found {len(fmap_jsons)} fieldmap JSON file(s)")
    print()
    
    for json_path in sorted(fmap_jsons):
        print(f"Processing: {json_path.name}")
        
        # Find matching functional file
        matching_func = find_matching_func(json_path, bids_root)
        
        if matching_func is None:
            print("  Skipping (no matching functional file found)")
            print()
            continue
        
        # Load existing JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Add or update IntendedFor field
        data['IntendedFor'] = [matching_func]
        
        print(f"  IntendedFor: {matching_func}")
        
        if not dry_run:
            # Write updated JSON
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print("  ✓ Updated")
        else:
            print("  (dry run - not saved)")
        
        print()
    
    if dry_run:
        print("DRY RUN complete. Run without --dry-run to apply changes.")
    else:
        print("All fieldmap JSON files updated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add IntendedFor field to fieldmap JSON sidecars in BIDS dataset"
    )
    parser.add_argument(
        'bids_root',
        type=str,
        help="Path to BIDS dataset root directory"
    )
    parser.add_argument(
        '--subject',
        type=str,
        required=True,
        help="Subject ID (without 'sub-' prefix, e.g., 'hp01')"
    )
    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help="Session ID (without 'ses-' prefix, e.g., '01')"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be done without modifying files"
    )
    
    args = parser.parse_args()
    
    try:
        add_intendedfor(args.bids_root, args.subject, args.session, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)