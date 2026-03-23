#!/usr/bin/env python
"""
unWarpEPIfloat.py
=================
Unwarp EPI data using blip-up/blip-down calibration data.

Usage
-----
    unWarpEPIfloat.py -f bold+orig'[348..351]' -r reverse+orig'[0..3]' \\
                      -d bold -s TS -w /path/to/workdir

Options
-------
    -f / --forward   Forward calibration data (subset of BOLD, with [idx] selector)
    -r / --reverse   Reverse-PE calibration data (with [idx] selector)
    -d / --data      Dataset(s) to correct, comma-separated, no +orig suffix
    -s / --subjID    Subject ID prefix (default: TS)
    -w / --workdir   Working directory containing input datasets (default: cwd)
    -a / --anat4warp Optional anatomical dataset
    -g / --giant_move  Pass giant_move to align_epi_anat.py
"""

import os
import sys
import subprocess
from optparse import OptionParser


class unWarpWithBlipUpBlipDownEPI:

    def __init__(self, options, parser):

        print("Unwarp function initialized")

        self.subjectID    = options.subjID if options.subjID else 'TS'
        self.workDir      = options.workdir if options.workdir else None
        self.tshift_ignore = '0'
        self.interp_mode  = 'quintic'
        self.compress     = '.gz'
        self.template     = 'NONE'
        self.template_minpatch = 17

        if not options.forward:
            print("!!! Required forward calibration data missing - exiting !!!")
            parser.print_usage()
            sys.exit(-1)
        self.forwardCalibrationData = options.forward
        print("Forward calibration data is %s" % self.forwardCalibrationData)

        if not options.reverse:
            print("!!! Required reverse calibration data missing - exiting !!!")
            parser.print_usage()
            sys.exit(-1)
        self.reverseCalibrationData = options.reverse
        print("Reverse calibration data is %s" % self.reverseCalibrationData)

        if not options.data:
            print("!!! Required data to be corrected missing - exiting !!!")
            parser.print_usage()
            sys.exit(-1)
        self.dataToCorrect = [ds for ds in options.data.split(',')]
        print("Data set(s) to be corrected is(are) %s" % self.dataToCorrect)

        if not options.anat4warp:
            self.dataAnat = 'NONE'
            print("Anatomical data not provided - continuing without it.")
        else:
            self.dataAnat = options.anat4warp
            print("Anatomical data set is %s" % self.dataAnat)

        self.giant_move = bool(options.giant_move)

    def unWarpData(self):

        # Change to explicit working directory first so all relative paths
        # and the subsequent os.chdir(outputDir) resolve correctly regardless
        # of how/where the script was invoked (shell, Docker, subprocess).
        if self.workDir:
            os.chdir(self.workDir)

        outputDir        = 'unWarpOutput_' + self.subjectID
        calibForwardName = self.subjectID + '_calibForwardData'
        calibReverseName = self.subjectID + '_calibReverseData'

        dataToCorrectList = [self.subjectID + '_' + ds for ds in self.dataToCorrect]

        # ------------------------------------------------------------------
        # Create output directory
        # ------------------------------------------------------------------
        if os.path.isdir(outputDir):
            print("!!! Output directory %s already exists !!!  Script ends now !!!"
                  % outputDir)
        else:
            try:
                subprocess.call(['mkdir', '-m', '755', outputDir],
                                stdout=subprocess.PIPE)
            except Exception:
                try:
                    subprocess.call(['mkdir', outputDir],
                                    stdout=subprocess.PIPE)
                except Exception:
                    print("!!! Could not create output directory %s !!!" % outputDir)

        # ------------------------------------------------------------------
        # Copy input data into outputDir (still in workDir / cwd at this point)
        # ------------------------------------------------------------------
        if self.dataAnat != 'NONE':
            anatBaseName = self.subjectID + '_anat.nii' + self.compress
            subprocess.call(['3dcopy', self.dataAnat,
                             outputDir + '/' + anatBaseName],
                            stdout=subprocess.PIPE)

        # 3dcalc instead of 3dcopy to allow sub-brick selection on calibration data
        subprocess.call(['3dcalc',
                         '-a', self.forwardCalibrationData,
                         '-expr', 'a',
                         '-prefix', outputDir + '/' + calibForwardName],
                        stdout=subprocess.PIPE)

        subprocess.call(['3dcalc',
                         '-a', self.reverseCalibrationData,
                         '-expr', 'a',
                         '-prefix', outputDir + '/' + calibReverseName],
                        stdout=subprocess.PIPE)

        for dataSet in self.dataToCorrect:
            subprocess.call(['3dcopy', dataSet,
                             outputDir + '/' + self.subjectID + '_' + dataSet],
                            stdout=subprocess.PIPE)

        # ------------------------------------------------------------------
        # Move into outputDir for all subsequent processing
        # ------------------------------------------------------------------
        os.chdir(outputDir)

        # ------------------------------------------------------------------
        # Compute median of each calibration dataset
        # ------------------------------------------------------------------
        processedCalibDataName01R = '01_' + calibReverseName + 'Median.nii' + self.compress
        subprocess.call(['3dTstat',
                         '-prefix', processedCalibDataName01R,
                         '-median', calibReverseName + '+orig'],
                        stdout=subprocess.PIPE)

        processedCalibDataName01F = '01_' + calibForwardName + 'Median.nii' + self.compress
        subprocess.call(['3dTstat',
                         '-prefix', processedCalibDataName01F,
                         '-median', calibForwardName + '+orig'],
                        stdout=subprocess.PIPE)

        # ------------------------------------------------------------------
        # Skull-strip each median calibration dataset
        # ------------------------------------------------------------------
        processedCalibDataName02R = ('02_' + calibReverseName
                                     + 'SkullStripped.nii' + self.compress)
        subprocess.call(['3dAutomask',
                         '-apply_prefix', processedCalibDataName02R,
                         processedCalibDataName01R])

        processedCalibDataName02F = ('02_' + calibForwardName
                                     + 'SkullStripped.nii' + self.compress)
        subprocess.call(['3dAutomask',
                         '-apply_prefix', processedCalibDataName02F,
                         processedCalibDataName01F])

        # ------------------------------------------------------------------
        # 3dQwarp: compute blip-up/blip-down displacement field
        # ------------------------------------------------------------------
        processedCalibDataName03 = ('03_' + self.subjectID
                                    + '_MidWarped.nii' + self.compress)
        subprocess.call(['3dQwarp',
                         '-plusminus',
                         '-pmNAMES', 'Reverse', 'Forward',
                         '-pblur', '0.05', '0.05',
                         '-blur', '-1', '-1',
                         '-noweight', '-minpatch', '9',
                         '-source', processedCalibDataName02R,
                         '-base',   processedCalibDataName02F,
                         '-prefix', processedCalibDataName03])

        # ------------------------------------------------------------------
        # Apply warp to each dataset to be corrected
        # ------------------------------------------------------------------
        for dataToCorrectName in dataToCorrectList:
            print("dataToCorrectName is %s" % dataToCorrectName)

            # (1) Time-shift
            processedData04 = dataToCorrectName + '_H'
            subprocess.call(['3dTshift',
                             '-tzero', '0', '-quintic',
                             '-ignore', self.tshift_ignore,
                             '-prefix', '04_' + processedData04 + '.nii' + self.compress,
                             dataToCorrectName + '+orig'],
                            stdout=subprocess.PIPE)

            # (2) Apply displacement warp
            processedData05 = processedData04 + 'W'
            subprocess.call(['3dNwarpApply',
                             '-nwarp',  ('03_' + self.subjectID
                                         + '_MidWarped_Forward_WARP.nii' + self.compress),
                             '-source', '04_' + processedData04 + '.nii' + self.compress,
                             '-prefix', '05_' + processedData05 + '.nii' + self.compress,
                             '-' + self.interp_mode])

            # (3) Volume registration via 3dAllineate (save matrix only)
            subprocess.call(['3dAllineate',
                             '-base',   ('03_' + self.subjectID
                                         + '_MidWarped_Forward.nii' + self.compress),
                             '-source', '05_' + processedData05 + '.nii' + self.compress,
                             '-prefix', 'NULL',
                             '-1Dmatrix_save', processedData04 + 'WV.aff12.1D',
                             '-1Dparam_save',  processedData04 + 'WV.motion.1D',
                             '-warp', 'shift_rotate', '-onepass',
                             '-fineblur', '2', '-lpa', '-norefinal',
                             '-final', 'quintic', '-automask+2', '-quiet'])

            # (4) Apply combined affine + nonlinear warp in one step
            warpMatrices = (processedData04 + 'WV.aff12.1D' + ' '
                            + '03_' + self.subjectID
                            + '_MidWarped_Forward_WARP.nii' + self.compress)
            subprocess.call(['3dNwarpApply',
                             '-nwarp',  warpMatrices,
                             '-source', '04_' + processedData04 + '.nii' + self.compress,
                             '-prefix', '06_' + processedData05 + 'V.nii' + self.compress,
                             '-' + self.interp_mode])

            # Preserve orientation / obliquity info
            subprocess.call(['3drefit',
                             '-atrcopy', calibForwardName + '+orig',
                             'IJK_TO_DICOM_REAL',
                             '06_' + processedData05 + 'V.nii' + self.compress])

        # ------------------------------------------------------------------
        # Optional: align anatomy to corrected EPI
        # ------------------------------------------------------------------
        if self.dataAnat != 'NONE':
            print("Aligning anat dataset to %s" % dataToCorrectList[0])
            dataToCorrectName = dataToCorrectList[0]

            subprocess.call(['3dUnifize', '-GM', '-clfrac', '0.22',
                             '-prefix', ('07_' + dataToCorrectName
                                         + '_unif.nii' + self.compress),
                             '06_' + processedData05 + 'V.nii' + self.compress + '[1]'])

            subprocess.call(['3dUnifize', '-GM', '-clfrac', '0.22',
                             '-prefix', ('08_' + self.subjectID
                                         + '_anat_unif.nii' + self.compress),
                             '-input',  self.subjectID + '_anat.nii' + self.compress])

            subprocess.call(['3dcalc',
                             '-a', ('08_' + self.subjectID
                                    + '_anat_unif.nii' + self.compress),
                             '-prefix', ('08_' + self.subjectID
                                         + '_anat_unif_short.nii' + self.compress),
                             '-datum', 'short', '-nscale', '-expr', 'a'])

            subprocess.call(['3dWarp',
                             '-card2oblique', ('07_' + dataToCorrectName
                                               + '_unif.nii' + self.compress),
                             '-prefix', ('08_' + self.subjectID
                                         + '_anat_ob_temp.nii' + self.compress),
                             ('08_' + self.subjectID
                              + '_anat_unif_short.nii' + self.compress)])

            alignCmds = ['align_epi_anat.py',
                         '-anat',       ('08_' + self.subjectID
                                         + '_anat_unif_short.nii' + self.compress),
                         '-epi',        ('07_' + dataToCorrectName
                                         + '_unif.nii' + self.compress),
                         '-epi_base',   '0',
                         '-epi_strip',  '3dAutomask',
                         '-suffix',     '_aligned',
                         '-cost',       'lpc+ZZ',
                         '-master_anat', ('08_' + self.subjectID
                                          + '_anat_ob_temp.nii' + self.compress)]
            if self.giant_move:
                alignCmds.append('-giant_move')
            subprocess.call(alignCmds)

        sys.exit(0)


def main():
    usage       = ("  %prog -f bold+orig'[0..5]' -r reverse+orig'[0..4]' "
                   "-d bold -s TS -w /workdir")
    description = ("Unwarp EPI data using blip-up/blip-down calibration data.")
    epilog      = ("Questions: Vinai Roopchansingh, Daniel Glen")

    parser = OptionParser(usage=usage, description=description, epilog=epilog)

    parser.add_option('-f', '--forward',   action='store',
                      help='Forward calibration data (with optional [idx] selector)')
    parser.add_option('-r', '--reverse',   action='store',
                      help='Reverse-PE calibration data (with optional [idx] selector)')
    parser.add_option('-d', '--data',      action='store',
                      help='Dataset(s) to correct, comma-separated, no +orig suffix')
    parser.add_option('-s', '--subjID',    action='store',
                      help='Subject ID prefix (default: TS)')
    parser.add_option('-w', '--workdir',   action='store',
                      help='Working directory containing input datasets (default: cwd)')
    parser.add_option('-a', '--anat4warp', action='store',
                      help='Optional anatomical dataset')
    parser.add_option('-g', '--giant_move', action='store_true',
                      help='Pass -giant_move to align_epi_anat.py')

    options, _ = parser.parse_args()

    job = unWarpWithBlipUpBlipDownEPI(options, parser)
    job.unWarpData()


if __name__ == '__main__':
    sys.exit(main())