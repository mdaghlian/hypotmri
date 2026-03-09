#!/usr/bin/env python
"""
run_mp2rage_preproc.py
======================
Run the MP2RAGE preprocessing pipeline:

    Step 0  – SPM bias-field correction of the INV2 image
    Step 1  – MPRAGEise the UNI image using the bias-corrected INV2
    Step 1b – Nighres MP2RAGE skull stripping (masks T1w + T1map)
    Step 2  – Nighres MGDM segmentation using skull-stripped inputs

The MPRAGEised UNI and MGDM outputs feed directly into the FreeSurfer
autorecon pipeline with no additional skull-strip.

Usage examples
--------------
# SPM standalone mode
python run_mp2rage_preproc.py \\
    --uni         /data/sub-01/ses-01/anat/sub-01_ses-01_UNI.nii.gz \\
    --inv2        /data/sub-01/ses-01/anat/sub-01_ses-01_INV2.nii.gz \\
    --t1map       /data/sub-01/ses-01/anat/sub-01_ses-01_T1map.nii.gz \\
    --outdir      /out/sub-01/ses-01/anat \\
    --subject     sub-01 \\
    --session     ses-01 \\
    --workdir     /tmp/mp2rage_work \\
    --mp2rage-script-dir /opt/mp2rage_scripts \\
    --spm-standalone     /opt/spm12/run_spm12.sh \\
    --mcr-path           /opt/mcr/v99

# MATLAB mode (omit --spm-standalone and --mcr-path)
python run_mp2rage_preproc.py \\
    --uni         /data/sub-01/ses-01/anat/sub-01_ses-01_UNI.nii.gz \\
    --inv2        /data/sub-01/ses-01/anat/sub-01_ses-01_INV2.nii.gz \\
    --outdir      /out \\
    --subject     sub-01 \\
    --mp2rage-script-dir /opt/mp2rage_scripts
"""

import os
import argparse
from pathlib import Path

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.io import DataSink

from preproc_utils import SPMBiasCorrect, MPRAGEise, NighresSkullStrip, NighresMGDM


# ---------------------------------------------------------------------------
# Output filename helper
# ---------------------------------------------------------------------------

def build_output_name(outputdir: str, subject: str, session: str,
                      suffix: str, extension: str = '.nii.gz') -> str:
    """
    Build a BIDS-style output filename from subject, session, and a suffix.

    Examples
    --------
    >>> build_output_name('/out', 'sub-01', 'ses-01', 'T1w-mpragised')
    '/out/sub-01_ses-01_T1w-mpragised.nii.gz'
    >>> build_output_name('/out', 'sub-01', None, 'T1w-mpragised')
    '/out/sub-01_T1w-mpragised.nii.gz'
    """
    tokens = [subject]
    if session:
        tokens.append(session)
    tokens.append(suffix)
    return os.path.join(outputdir, '_'.join(tokens) + extension)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(uni: str, inv2: str, outdir: str,
                   subject: str, session: str, workdir: str,
                   mp2rage_script_dir: str,
                   t1map: str = None,
                   spm_script: str = 's01_spmbc',
                   spm_standalone: str = None,
                   mcr_path: str = None,
                   nighres_docker: str = 'nighres/nighres:latest',
                   mgdm_contrast: str = 'Mp2rage7T',
                   mgdm_atlas: str = None,
                   ) -> pe.Workflow:
    """
    Construct and return the MP2RAGE preprocessing nipype workflow.

    Parameters
    ----------
    uni                : Path to UNI image
    inv2               : Path to INV2 image
    outdir             : Directory for final outputs
    subject            : BIDS subject label (e.g. 'sub-01')
    session            : BIDS session label (e.g. 'ses-01'), or None
    workdir            : Nipype working/cache directory
    mp2rage_script_dir : Directory containing the SPM m-script
    t1map              : Path to T1 map image (optional, recommended for 7T)
    spm_script         : SPM m-script name (default: 's01_spmbc')
    spm_standalone     : Path to SPM standalone executable (optional)
    mcr_path           : Path to MATLAB MCR directory (optional)
    nighres_docker     : Nighres Docker image tag
    mgdm_contrast      : MGDM contrast type (default: Mp2rage7T)
    mgdm_atlas         : MGDM atlas file (optional; uses nighres default when unset)

    Returns
    -------
    wf : pe.Workflow
    """
    os.makedirs(outdir,  exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    wf = pe.Workflow(name='mp2rage_preproc', base_dir=workdir)

    # ------------------------------------------------------------------
    # Input node
    # ------------------------------------------------------------------
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['uni', 'inv2', 't1map']),
        name='inputnode',
    )
    inputnode.inputs.uni  = str(Path(uni).resolve())
    inputnode.inputs.inv2 = str(Path(inv2).resolve())
    if t1map:
        inputnode.inputs.t1map = str(Path(t1map).resolve())

    # ------------------------------------------------------------------
    # Step 0 – SPM bias-field correction of INV2
    # ------------------------------------------------------------------
    spm_kwargs = dict(
        spm_script=spm_script,
        mp2rage_script_dir=mp2rage_script_dir,
    )
    if spm_standalone and mcr_path:
        spm_kwargs['spm_standalone'] = spm_standalone
        spm_kwargs['mcr_path']       = mcr_path

    biascorrect = pe.Node(
        SPMBiasCorrect(**spm_kwargs),
        name='spm_biascorrect',
    )
    wf.connect(inputnode, 'inv2', biascorrect, 'input_image')

    # ------------------------------------------------------------------
    # Step 1 – MPRAGEise the UNI image using bias-corrected INV2
    # ------------------------------------------------------------------
    mpragise = pe.Node(MPRAGEise(), name='mpragise')
    wf.connect(inputnode,   'uni',          mpragise, 'uni_image')
    wf.connect(biascorrect, 'output_image', mpragise, 'inv2_image')

    # ------------------------------------------------------------------
    # Step 1b – Nighres skull stripping
    #
    # Uses bias-corrected INV2 as the mandatory second_inversion input.
    # Passes MPRAGEised UNI as t1w so nighres returns a skull-stripped
    # version ready for MGDM. T1map is also masked here if provided.
    # ------------------------------------------------------------------
    skullstrip = pe.Node(
        NighresSkullStrip(docker_image=nighres_docker),
        name='skullstrip',
    )
    wf.connect(biascorrect, 'output_image', skullstrip, 'inv2_image')
    wf.connect(mpragise,    'output_image', skullstrip, 't1w_image')
    wf.connect(inputnode,    'uni', skullstrip, 't1w_image')
    if t1map:
        wf.connect(inputnode, 't1map', skullstrip, 't1map_image')

    # ------------------------------------------------------------------
    # Step 2 – Nighres MGDM segmentation
    #
    # Receives skull-stripped T1w (and T1map if available) from Step 1b.
    # Using skull-stripped inputs is the workflow shown in the official
    # nighres tissue classification example.
    # ------------------------------------------------------------------
    mgdm_kwargs = dict(
        docker_image=nighres_docker,
        contrast_type=mgdm_contrast,
    )
    if mgdm_atlas:
        mgdm_kwargs['atlas'] = mgdm_atlas

    mgdm = pe.Node(NighresMGDM(**mgdm_kwargs), name='mgdm')

    wf.connect(skullstrip, 't1w_masked', mgdm, 'input_image')
    # wf.connect(inputnode, 'uni', mgdm, 'input_image')
    if t1map:
        wf.connect(skullstrip, 't1map_masked', mgdm, 't1map_image')

    # ------------------------------------------------------------------
    # Output node
    # ------------------------------------------------------------------
    output_fields = [
        'inv2_biascorrected',
        'uni_mpragised',
        'brain_mask',
        'mgdm_segmentation',
        'mgdm_distance',
        'mgdm_labels',
        'mgdm_memberships',
    ]
    outputnode = pe.Node(
        niu.IdentityInterface(fields=output_fields),
        name='outputnode',
    )
    wf.connect(biascorrect, 'output_image', outputnode, 'inv2_biascorrected')
    wf.connect(mpragise,    'output_image', outputnode, 'uni_mpragised')
    wf.connect(skullstrip,  'brain_mask',   outputnode, 'brain_mask')
    wf.connect(mgdm,        'segmentation', outputnode, 'mgdm_segmentation')
    wf.connect(mgdm,        'distance',     outputnode, 'mgdm_distance')
    wf.connect(mgdm,        'labels',       outputnode, 'mgdm_labels')
    wf.connect(mgdm,        'memberships',  outputnode, 'mgdm_memberships')

    # ------------------------------------------------------------------
    # DataSink – copy final outputs to outdir with BIDS-style names
    # ------------------------------------------------------------------
    prefix = subject + ('_' + session if session else '')

    sink = pe.Node(DataSink(base_directory=outdir), name='datasink')
    sink.inputs.substitutions = [
        ('inv2_biascorrected', '{}_INV2-spmbc'.format(prefix)),
        ('uni_mpragised',      '{}_UNI-mpragised'.format(prefix)),
        ('brain_mask',         '{}_brain-mask'.format(prefix)),
        ('mgdm_segmentation',  '{}_mgdm-seg'.format(prefix)),
        ('mgdm_distance',      '{}_mgdm-dist'.format(prefix)),
        ('mgdm_labels',        '{}_mgdm-lbls'.format(prefix)),
        ('mgdm_memberships',   '{}_mgdm-mems'.format(prefix)),
    ]

    wf.connect(outputnode, 'inv2_biascorrected', sink, '@inv2_biascorrected')
    wf.connect(outputnode, 'uni_mpragised',      sink, '@uni_mpragised')
    wf.connect(outputnode, 'brain_mask',         sink, '@brain_mask')
    wf.connect(outputnode, 'mgdm_segmentation',  sink, '@mgdm_segmentation')
    wf.connect(outputnode, 'mgdm_distance',      sink, '@mgdm_distance')
    wf.connect(outputnode, 'mgdm_labels',        sink, '@mgdm_labels')
    wf.connect(outputnode, 'mgdm_memberships',   sink, '@mgdm_memberships')

    return wf


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='MP2RAGE → FreeSurfer preprocessing pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--uni',    required=True, help='UNI image (.nii/.nii.gz)')
    p.add_argument('--inv2',   required=True, help='INV2 image (.nii/.nii.gz)')
    p.add_argument('--t1map',  default=None,  help='T1 map (.nii/.nii.gz) — strongly recommended for 7T')
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--subject', required=True, help='BIDS subject label e.g. sub-01')
    p.add_argument('--session', default=None,  help='BIDS session label e.g. ses-01')
    p.add_argument('--workdir', default='/tmp/mp2rage_work',
                   help='Nipype cache directory')
    p.add_argument('--mp2rage-script-dir', required=True,
                   help='Directory containing SPM m-script')
    p.add_argument('--spm-script', default='sZ0_spmbc',
                   help='SPM m-script name')
    p.add_argument('--spm-standalone', default=None,
                   help='Path to SPM standalone executable')
    p.add_argument('--mcr-path', default=None,
                   help='Path to MATLAB MCR (required if --spm-standalone set)')
    p.add_argument('--nighres-docker', default='nighres/nighres:latest',
                   help='Nighres Docker image tag')
    p.add_argument('--mgdm-contrast', default='Mp2rage7T',
                   help='MGDM contrast type')
    p.add_argument('--mgdm-atlas', default=None,
                   help='MGDM atlas file (uses nighres default if unset)')
    return p


def main():
    args = _build_parser().parse_args()

    wf = build_pipeline(
        uni=args.uni,
        inv2=args.inv2,
        outdir=args.outdir,
        subject=args.subject,
        session=args.session,
        workdir=args.workdir,
        mp2rage_script_dir=args.mp2rage_script_dir,
        t1map=args.t1map,
        spm_script=args.spm_script,
        spm_standalone=args.spm_standalone,
        mcr_path=args.mcr_path,
        nighres_docker=args.nighres_docker,
        mgdm_contrast=args.mgdm_contrast,
        mgdm_atlas=args.mgdm_atlas,
    )

    wf.run()


if __name__ == '__main__':
    main()