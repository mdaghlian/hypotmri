# How this pipeline is written - and how you can implement your own stages 

The aim of this pipeline is to be modular, reproducible, efficient and readable.

The bash scripts run as in any pipeline.  

The python scrips are a little more involved...


## Simple mental model for python scripts:

If you remember nothing else, remember this:

> **This is essentially just a bash pipeline, but:**
> * `called from python`
> * `create one folder (workdir) per run`
> * `check for stage outputs in workdir`
> * `if present - skip this stage`
> * `else run command this stage`

Why run it like this? Why not just use BASH? 

1) The `workdir` + python combination is very useful for making the script portable. Because all paths are defined relative to `workdir` you can run the same command on the cluster, locally, or inside a container. This is organized by the `run_cmd` function.

> It doesn’t matter whether the commands are: AFNI; FSL; FreeSurfer or your own tools

> The structure is always the same.

2) Easier to write and implement little helper functions + mathematical operations in python than in bash.   

I'll explain it is a bit more depth here. 

---

### [1] The working directory

Each run/processing stage gets its own folder:

```text
derivatives/<output-file>/<sub>/<ses>/<task_run>/
```

Inside that folder:

* inputs are copied in
* commands are run
* temporary files live there

Everything happens inside `work_dir`.

This avoids path confusion (especially with containers and running on the cluster vs locally)

The functions `_stage` and `_container_path`, are used to update paths to work in this way.

---

### [2] Check outputs
Each script is divided into steps. These are given at the top of the script:

```bash
# e.g., 
STEP_KEYS = [
    'convert_bold',
    'convert_reverse',
    'unwarp',
    'apply_warp_sbref',
]
```

Each key has an associated output file which it will look for inside the workdir. If it is not present, the command will run, otherwise it will skip. 

Note - you can also specify specific stages to re-run: e.g., `--overwrite unwarp` would re-run overwrite unwarp stage (and all the downstream stages). 

This makes it easier to run again if something breaks, or you need to do additional runs, without rerunning everything. 

### [3] `run_cmd`

All of the tools (AFNI; FSL; Freesurfer) are called with the function `run_cmd`. 

```python
afni_docker="vnmd/afni_26.0.07"
nifti = "path/to/nifti.nii"

# if afni_docker is none, keeps the same path
# else, replace work_dir with /data/ for the container
src = _container_path(work_dir, os.path.basename(nifti), afni_docker)
dst = _container_path(work_dir, 'afni_copy', afni_docker)

# Run afni 3dcopy
# if docker is none, run locally 
# else run in docker
run_cmd(
    work_dir=workdir, 
    docker_image=afni_docker,
    cmd=['3dcopy', src, dst]
    )
```

All of this boils down to:
```bash
3dcopy path/to/nifti.nii afni_copy
```
*Wouldn't it just be easier to do this straightaway?*

Doing it this way makes it easier for us to pick and choose from many different packages, without worrying about complex installation path interference etc. 


## How to think about adding your own step

You don’t need to understand everything.

Just follow this pattern.

---

### Step 1 — think in bash first

Example:

```bash
my_tool input.nii output.nii.gz
```

---

### Step 2 — wrap it

```python
def my_step(input_file, work_dir, docker):
    run_cmd(
        work_dir=work_dir,
        docker_image=docker,
        cmd=['my_tool', input_file, 'output.nii.gz']
    )
    return os.path.join(work_dir, 'output.nii.gz')
```

---

### Step 3 — plug into pipeline

Inside the run loop:

```python
output = final_output_path

if not check_skip({'output': output}, ow['my_step']):
    result = my_step(...)
    copy result → output
```

---

### Step 4 — register the step

Add to:

```python
STEP_KEYS = [..., 'my_step']
```

Now you can do:

```bash
--overwrite my_step
```

---

## The only rules you need to follow

### ✔ Always write outputs to the working directory

```text
good: work_dir/output.nii.gz
bad: random/other/place/output.nii.gz
```

---

### ✔ Always use `run_cmd`

So your code works:

* locally
* in Docker
* in clusters

---

### ✔ Always check if output exists

```python
if not check_skip(...):
```

---

### ✔ Keep steps independent

Each step should:

* take inputs
* produce outputs
* not rely on hidden state

---

