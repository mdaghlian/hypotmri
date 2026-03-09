import os
import json
import shutil
import subprocess
import threading
from pathlib import Path

import nibabel as nib
import numpy as np

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    isdefined,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_stem(path: Path) -> str:
    """Strip both .nii.gz and .nii extensions to return the bare file stem."""
    return Path(path.stem).stem if path.suffix == '.gz' else path.stem


def _stage_inputs(work_dir: str, *paths: str) -> None:
    """Copy files into work_dir if not already there."""
    for src in paths:
        dst = os.path.join(work_dir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)


def _check_result(result, tool_name: str) -> None:
    """Raise RuntimeError with full stdout/stderr if a subprocess failed."""
    if result.returncode != 0:
        raise RuntimeError(
            '{} failed (exit {}).\n'
            '--- stdout ---\n{}\n'
            '--- stderr ---\n{}'.format(
                tool_name, result.returncode, result.stdout, result.stderr)
        )


def _run_docker(work_dir: str, docker_image: str, cmd: list,
                env_vars: dict = None, verbose: bool = True) -> None:
    """
    Run *cmd* (a list of strings) inside *docker_image*, mounting *work_dir*
    as ``/data``.

    Optional *env_vars* dict is passed as ``-e KEY=VALUE`` flags.

    If *verbose* is True, container stdout/stderr are streamed to the terminal
    in real time via two reader threads (avoids deadlock on full pipe buffers).

    Raises ``RuntimeError`` with full stdout/stderr on non-zero exit.
    """
    env_flags = []
    for k, v in (env_vars or {}).items():
        env_flags += ['-e', '{}={}'.format(k, v)]

    proc = subprocess.Popen(
        ['docker', 'run', '--rm',
         *env_flags,
         '-v', '{}:/data'.format(work_dir),
         docker_image] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_lines = []
    stderr_lines = []

    def _reader(stream, store, label):
        for line in stream:
            store.append(line)
            if verbose:
                print('[docker {}] {}'.format(label, line), end='', flush=True)

    t_out = threading.Thread(target=_reader,
                             args=(proc.stdout, stdout_lines, 'stdout'))
    t_err = threading.Thread(target=_reader,
                             args=(proc.stderr, stderr_lines, 'stderr'))
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    proc.wait()

    # Reconstruct a result-like object for _check_result
    class _Result:
        returncode = proc.returncode
        stdout     = ''.join(stdout_lines)
        stderr     = ''.join(stderr_lines)

    _check_result(_Result(), 'Docker container ({})'.format(docker_image))


# ---------------------------------------------------------------------------
# Interface 0 – SPM bias-field correction
# ---------------------------------------------------------------------------

class SPMBiasCorrectInputSpec(BaseInterfaceInputSpec):
    input_image        = File(mandatory=True, exists=True,
                              desc='Input NIfTI image (.nii or .nii.gz)')
    spm_script         = traits.Str('s01_spmbc', usedefault=True,
                                    desc='SPM m-script name (default: s01_spmbc)')
    spm_standalone     = traits.Str(desc='Path to SPM standalone executable '
                                         '(omit to use MATLAB)')
    mcr_path           = traits.Str(desc='Path to MATLAB MCR directory '
                                         '(required when spm_standalone is set)')
    mp2rage_script_dir = traits.Str(mandatory=True,
                                    desc='Directory containing the SPM m-script')


class SPMBiasCorrectOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc='Bias-corrected NIfTI image (.nii.gz)')


class SPMBiasCorrect(BaseInterface):
    """
    Run SPM bias-field correction via SPM standalone or MATLAB.

    Wraps the ``s01_spmbc`` m-script (or a user-specified alternative).
    SPM output is converted to .nii.gz and written to the node working
    directory as ``<stem>_spmbc.nii.gz``.

    Mode selection
    --------------
    * **SPM standalone** – set both ``spm_standalone`` and ``mcr_path``.
    * **MATLAB**         – leave both unset; ``matlab`` must be on ``$PATH``.
    """

    input_spec  = SPMBiasCorrectInputSpec
    output_spec = SPMBiasCorrectOutputSpec

    _output_file: Path = None

    def _run_interface(self, runtime):
        input_path = Path(self.inputs.input_image).resolve()
        stem       = _get_stem(input_path)

        spm_out_dir       = Path(runtime.cwd) / '{}_spm_biascorrect'.format(stem)
        spm_out_dir.mkdir(parents=True, exist_ok=True)
        spm_biascorrected = spm_out_dir / '{}_biascorrected.nii'.format(stem)

        script = self.inputs.spm_script

        staged_input = Path(runtime.cwd) / input_path.name
        if staged_input.resolve() != input_path.resolve():
            shutil.copy(str(input_path), str(staged_input))

        matlab_expr = "{script}('{input}', '{outdir}');".format(
            script=script,
            input=str(staged_input),
            outdir=str(spm_out_dir),
        )

        if isdefined(self.inputs.spm_standalone) and isdefined(self.inputs.mcr_path):
            cmd = [
                self.inputs.spm_standalone,
                self.inputs.mcr_path,
                'script',
                matlab_expr,
            ]
        else:
            cmd = [
                'matlab', '-nodisplay', '-nosplash', '-nodesktop',
                '-batch', matlab_expr,
            ]

        result = subprocess.run(
            cmd,
            shell=False,
            cwd=self.inputs.mp2rage_script_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,
        )
        _check_result(result, 'SPM bias correction')

        if not spm_biascorrected.exists():
            raise FileNotFoundError(
                'SPM bias correction completed but expected output not found:\n'
                '  {}'.format(spm_biascorrected)
            )

        out_path = Path(runtime.cwd) / '{}_spmbc.nii.gz'.format(stem)
        img = nib.load(str(spm_biascorrected))
        nib.save(img, str(out_path))

        self._output_file = out_path
        return runtime

    def _list_outputs(self):
        if self._output_file is None:
            raise RuntimeError('_list_outputs called before _run_interface')
        return {'output_image': str(self._output_file.resolve())}


# ---------------------------------------------------------------------------
# Interface 1 – MPRAGEise UNI image
# ---------------------------------------------------------------------------

def _mprage_ise(uni_file: str, inv2_file: str, out_file: str) -> None:
    """
    Suppress MP2RAGE background noise by multiplying UNI by normalised INV2.

    The INV2 image is normalised to its 99th percentile (computed over
    positive voxels only) before multiplication, so the resulting image
    retains the UNI contrast profile but with background noise suppressed.
    """
    uni_img  = nib.load(uni_file)
    inv2_img = nib.load(inv2_file)

    uni_data  = uni_img.get_fdata()
    inv2_data = inv2_img.get_fdata()

    positive_voxels = inv2_data[inv2_data > 0]
    if positive_voxels.size == 0:
        raise ValueError(
            'INV2 image contains no positive voxels — '
            'check that the correct image was supplied: {}'.format(inv2_file)
        )

    norm_factor = np.percentile(positive_voxels, 99)
    if norm_factor == 0:
        raise ValueError(
            '99th-percentile of INV2 positive voxels is zero — '
            'normalisation would produce NaN/Inf values.'
        )

    uni_clean_data = (inv2_data / norm_factor) * uni_data

    nib.save(
        nib.Nifti1Image(uni_clean_data, uni_img.affine, uni_img.header),
        out_file,
    )


class MPRAGEiseInputSpec(BaseInterfaceInputSpec):
    uni_image  = File(mandatory=True, exists=True,
                      desc='UNI image from MP2RAGE acquisition (.nii or .nii.gz)')
    inv2_image = File(mandatory=True, exists=True,
                      desc='Bias-corrected INV2 image (.nii or .nii.gz)')


class MPRAGEiseOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc='MPRAGEised UNI image (.nii.gz)')


class MPRAGEise(BaseInterface):
    """
    Suppress MP2RAGE background noise (MPRAGEising).

    Multiplies the UNI image by the INV2 image normalised to its 99th
    percentile, effectively driving background noise toward zero while
    preserving grey/white contrast.

    The INV2 input should already be bias-field corrected (e.g. via
    :class:`SPMBiasCorrect`) for best results.

    Output is written to the nipype node working directory as
    ``<uni_stem>_mpragised.nii.gz``.
    """

    input_spec  = MPRAGEiseInputSpec
    output_spec = MPRAGEiseOutputSpec

    _output_file: Path = None

    def _run_interface(self, runtime):
        uni_path = Path(self.inputs.uni_image).resolve()
        stem     = _get_stem(uni_path)

        out_path = Path(runtime.cwd) / '{}_mpragised.nii.gz'.format(stem)

        _mprage_ise(
            uni_file=str(uni_path),
            inv2_file=str(Path(self.inputs.inv2_image).resolve()),
            out_file=str(out_path),
        )

        self._output_file = out_path
        return runtime

    def _list_outputs(self):
        if self._output_file is None:
            raise RuntimeError('_list_outputs called before _run_interface')
        return {'output_image': str(self._output_file.resolve())}


# ---------------------------------------------------------------------------
# Interface 1b – Nighres MP2RAGE skull stripping (via Docker)
# ---------------------------------------------------------------------------

class NighresSkullStripInputSpec(BaseInterfaceInputSpec):
    inv2_image   = File(mandatory=True, exists=True,
                        desc='INV2 image — only mandatory input for brain mask')
    t1w_image    = File(desc='T1-weighted (MPRAGEised UNI) image to be masked')
    t1map_image  = File(desc='Quantitative T1 map to be masked')
    docker_image = traits.Str('nighres/nighres:latest', usedefault=True,
                              desc='Nighres Docker image tag')


class NighresSkullStripOutputSpec(TraitedSpec):
    brain_mask   = File(exists=True, desc='Binary brain mask')
    t1w_masked   = File(desc='T1w image with skull removed (only if t1w_image provided)')
    t1map_masked = File(desc='T1map with skull removed (only if t1map_image provided)')


class NighresSkullStrip(BaseInterface):
    """
    Skull-strip MP2RAGE data using ``nighres.brain.mp2rage_skullstripping``
    inside a Docker container.

    Only ``inv2_image`` is mandatory. When ``t1w_image`` and/or
    ``t1map_image`` are provided, nighres will mask them and return the
    skull-stripped versions alongside the brain mask — these are the images
    that should be passed to MGDM.

    Outputs are read back from a JSON file written by the container so that
    filename detection is robust to nighres version changes.
    """

    input_spec  = NighresSkullStripInputSpec
    output_spec = NighresSkullStripOutputSpec

    _brain_mask_file  : Path = None
    _t1w_masked_file  : Path = None
    _t1map_masked_file: Path = None

    def _run_interface(self, runtime):
        cwd = Path(runtime.cwd)

        inv2_path = Path(self.inputs.inv2_image).resolve()
        _stage_inputs(str(cwd), str(inv2_path))
        stem = _get_stem(inv2_path)

        # Build optional kwargs for t1w and t1map
        optional_kwargs = ''
        if isdefined(self.inputs.t1w_image) and self.inputs.t1w_image:
            t1w_path = Path(self.inputs.t1w_image).resolve()
            _stage_inputs(str(cwd), str(t1w_path))
            optional_kwargs += (
                '    t1_weighted="/data/{}", '.format(t1w_path.name)
            )
        if isdefined(self.inputs.t1map_image) and self.inputs.t1map_image:
            t1map_path = Path(self.inputs.t1map_image).resolve()
            _stage_inputs(str(cwd), str(t1map_path))
            optional_kwargs += (
                '    t1_map="/data/{}", '.format(t1map_path.name)
            )

        python_script = (
            'import nighres, json; '
            'r = nighres.brain.mp2rage_skullstripping('
            '    second_inversion="/data/{inv2}", '
            + optional_kwargs +
            '    save_data=True, '
            '    output_dir="/data", '
            '    file_name="{stem}"); '
            'paths = {{k: str(v) for k, v in r.items()}}; '
            'open("/data/skullstrip_outputs.json", "w").write(json.dumps(paths)); '
        ).format(
            inv2=inv2_path.name,
            stem=stem,
        )

        _run_docker(
            work_dir=str(cwd),
            docker_image=self.inputs.docker_image,
            cmd=['python3', '-c', python_script],
            verbose=True,
        )

        json_path = cwd / 'skullstrip_outputs.json'
        if not json_path.exists():
            raise FileNotFoundError(
                'Skull stripping completed but output JSON not found: {}'.format(json_path)
            )

        with open(json_path) as f:
            out_paths = json.load(f)

        def _remap(docker_path):
            """Replace /data prefix with actual cwd."""
            return Path(str(docker_path).replace('/data', str(cwd)))

        self._brain_mask_file = _remap(out_paths['brain_mask'])

        if 't1w_masked' in out_paths and out_paths['t1w_masked']:
            self._t1w_masked_file = _remap(out_paths['t1w_masked'])

        if 't1map_masked' in out_paths and out_paths['t1map_masked']:
            self._t1map_masked_file = _remap(out_paths['t1map_masked'])

        # Verify mandatory output
        if not self._brain_mask_file.exists():
            raise FileNotFoundError(
                'Expected brain mask not found: {}'.format(self._brain_mask_file)
            )

        return runtime

    def _list_outputs(self):
        if self._brain_mask_file is None:
            raise RuntimeError('_list_outputs called before _run_interface')
        outputs = {'brain_mask': str(self._brain_mask_file.resolve())}
        if self._t1w_masked_file is not None:
            outputs['t1w_masked']   = str(self._t1w_masked_file.resolve())
        if self._t1map_masked_file is not None:
            outputs['t1map_masked'] = str(self._t1map_masked_file.resolve())
        return outputs


# ---------------------------------------------------------------------------
# Interface 2 – Nighres MGDM brain segmentation (via Docker)
# ---------------------------------------------------------------------------

class NighresMGDMInputSpec(BaseInterfaceInputSpec):
    input_image   = File(mandatory=True, exists=True,
                         desc='Skull-stripped T1w (MPRAGEised UNI) image')
    docker_image  = traits.Str('nighres/nighres:latest', usedefault=True,
                               desc='Nighres Docker image tag')
    contrast_type = traits.Str('Mp2rage7T', usedefault=True,
                               desc='MGDM contrast type (default: Mp2rage7T)')
    t1map_image   = File(desc='Skull-stripped T1 map (.nii or .nii.gz). '
                              'When provided, passed as contrast_image2/T1map7T '
                              'for improved tissue boundary delineation at 7T.')
    atlas         = traits.Str(desc='MGDM atlas file (optional; uses nighres '
                                    'default when unset)')


class NighresMGDMOutputSpec(TraitedSpec):
    segmentation = File(exists=True, desc='Hard tissue segmentation label image')
    memberships  = File(exists=True, desc='Tissue membership / probability image')
    labels       = File(exists=True, desc='Label definition image')
    distance     = File(exists=True, desc='Distance to nearest border image')


class NighresMGDM(BaseInterface):
    """
    Run nighres MGDM brain segmentation inside a Docker container.

    Expects skull-stripped inputs (from :class:`NighresSkullStrip`).
    Output paths are read back from a JSON sidecar written by the container,
    making filename detection robust to nighres version changes.

    Docker mount
    ------------
    The node working directory is mounted as ``/data`` inside the container.
    All paths passed to the Python script inside Docker are ``/data``-relative.
    """

    input_spec  = NighresMGDMInputSpec
    output_spec = NighresMGDMOutputSpec

    _seg_file : Path = None
    _mem_file : Path = None
    _lab_file : Path = None
    _dist_file: Path = None

    def _run_interface(self, runtime):
        input_path = Path(self.inputs.input_image).resolve()
        stem       = _get_stem(input_path)
        cwd        = Path(runtime.cwd)

        _stage_inputs(str(cwd), str(input_path))

        # Optional T1map
        contrast2_kwargs = ''
        if isdefined(self.inputs.t1map_image) and self.inputs.t1map_image:
            t1map_path = Path(self.inputs.t1map_image).resolve()
            _stage_inputs(str(cwd), str(t1map_path))
            contrast2_kwargs = (
                '    contrast_image2="/data/{t1map}", '
                '    contrast_type2="T1map7T", '
            ).format(t1map=t1map_path.name)

        # Optional atlas
        atlas_kwarg = ''
        if isdefined(self.inputs.atlas) and self.inputs.atlas:
            atlas_kwarg = '    atlas_file="{}", '.format(self.inputs.atlas)

        python_script = (
            'import nighres, json; '
            'r = nighres.brain.mgdm_segmentation('
            '    contrast_image1="/data/{input}", '
            '    contrast_type1="{contrast}", '
            + contrast2_kwargs
            + atlas_kwarg +
            '    save_data=True, '
            '    output_dir="/data", '
            '    file_name="{stem}"); '
            # Write result paths to JSON for robust host-side detection
            'paths = {{k: str(v) for k, v in r.items()}}; '
            'open("/data/mgdm_outputs.json", "w").write(json.dumps(paths)); '
        ).format(
            input=input_path.name,
            contrast=self.inputs.contrast_type,
            stem=stem,
        )

        _run_docker(
            work_dir=str(cwd),
            docker_image=self.inputs.docker_image,
            cmd=['python3', '-c', python_script],
            verbose=True,
        )

        json_path = cwd / 'mgdm_outputs.json'
        if not json_path.exists():
            raise FileNotFoundError(
                'MGDM completed but output JSON not found: {}'.format(json_path)
            )

        with open(json_path) as f:
            out_paths = json.load(f)

        def _remap(docker_path):
            """Replace /data prefix with actual cwd."""
            return Path(str(docker_path).replace('/data', str(cwd)))

        self._seg_file  = _remap(out_paths['segmentation'])
        self._dist_file = _remap(out_paths['distance'])
        self._mem_file  = _remap(out_paths['memberships'])
        self._lab_file  = _remap(out_paths['labels'])

        for attr, label in [
            (self._seg_file,  'segmentation'),
            (self._dist_file, 'distance'),
            (self._mem_file,  'memberships'),
            (self._lab_file,  'labels'),
        ]:
            if not attr.exists():
                raise FileNotFoundError(
                    'MGDM completed but expected {} output not found:\n'
                    '  {}'.format(label, attr)
                )

        return runtime

    def _list_outputs(self):
        if self._seg_file is None:
            raise RuntimeError('_list_outputs called before _run_interface')
        return {
            'segmentation': str(self._seg_file.resolve()),
            'distance':     str(self._dist_file.resolve()),
            'memberships':  str(self._mem_file.resolve()),
            'labels':       str(self._lab_file.resolve()),
        }