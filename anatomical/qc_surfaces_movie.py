#!/usr/bin/env python3
"""
qc_surfaces_movie.py

Uses freeview to screenshot FreeSurfer segmentation slices (sagittal),
then combines them into an mp4 movie for easy QC.
Assumes SUBJECTS_DIR is set in the environment.
"""

import subprocess
import os
import sys
import argparse


FREEVIEW_CMD = "freeview -cmd {cmd}"
CMD_TXT = (
    " -v {anatomy}:grayscale=10,100"
    " -f {lh_wm}:color=red:edgecolor=red"
    " -f {rh_wm}:color=red:edgecolor=red"
    " -f {lh_pial}:color=white:edgecolor=white"
    " -f {rh_pial}:color=white:edgecolor=white"
    "\n -viewport sagittal\n"
)
SLICE_ADDITION = " -slice {xpos} 127 127 \n -ss {opfn} \n"


def usage():
    print("Usage: qc_surfaces_movie.py <sub>")
    print("Takes sagittal screenshots of FreeSurfer surfaces and saves an mp4.")
    print("Assumes SUBJECTS_DIR is set in the environment.")
    sys.exit(1)


def parse_args():
    if len(sys.argv) < 2:
        usage()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("sub")
    parser.add_argument("--slices-start", type=int, default=90)
    parser.add_argument("--slices-end",   type=int, default=240)
    parser.add_argument("--framerate",    type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Normalise subject label
    subject = args.sub
    subject = "sub-" + subject.removeprefix("sub-")

    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if not subjects_dir:
        print("Error: SUBJECTS_DIR is not set in the environment.")
        sys.exit(1)

    fs_folder = os.path.join(subjects_dir, subject)
    if not os.path.isdir(fs_folder):
        print(f"Error: Subject directory not found for {subject} in {subjects_dir}")
        sys.exit(1)

    # --- Build freeview command file ---
    target_dir = os.path.join(fs_folder, "movie")
    os.makedirs(target_dir, exist_ok=True)
    cmd_file = os.path.join(target_dir, "cmd.txt")

    sj_cmd = CMD_TXT.format(
        anatomy  = os.path.join(fs_folder, "mri",  "T1.mgz"),
        lh_wm    = os.path.join(fs_folder, "surf", "lh.white"),
        rh_wm    = os.path.join(fs_folder, "surf", "rh.white"),
        lh_pial  = os.path.join(fs_folder, "surf", "lh.pial"),
        rh_pial  = os.path.join(fs_folder, "surf", "rh.pial"),
    )

    slices = range(args.slices_start, args.slices_end)
    for sag_slice in slices:
        sj_cmd += SLICE_ADDITION.format(
            xpos  = sag_slice,
            opfn  = os.path.join(target_dir, str(sag_slice).zfill(3) + ".png"),
        )
    sj_cmd += " -quit\n"

    with open(cmd_file, "w") as f:
        f.write(sj_cmd)

    # --- Run freeview ---
    print(f"Taking screenshots for {subject}...")
    subprocess.call(FREEVIEW_CMD.format(cmd=cmd_file), shell=True)

    # --- Stitch into movie ---
    mp4_out = os.path.join(target_dir, f"{subject}.mp4")
    convert_cmd = (
        f'ffmpeg -framerate {args.framerate}'
        f' -pattern_type glob -i "{target_dir}/*.png"'
        f' -b:v 2M -c:v mpeg4 {mp4_out}'
    )
    print(f"Building movie -> {mp4_out}")
    subprocess.call(convert_cmd, shell=True)


if __name__ == "__main__":
    main()