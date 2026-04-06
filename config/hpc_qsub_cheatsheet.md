# Sun Grid Engine (SGE) / Son of Grid Engine (SOGE) Cheat Sheet
Written by Marcus + chatgpt

To run a "job" (i.e., a stage of the pipeline for a single subject) you need to send it to the "queue". Myriad uses "SGE" to manage the queues. basically they are there to make sure that access to the cluster is fair (i.e., not one person submits to many jobs). Things you need to think about are - how long will the 
Myriad (ucl HPC) 

This guide covers the essential commands and common directives for submitting, monitoring, and controlling jobs on an SGE/SOGE cluster.

## 1. Core Commands (CLI)

|Command|Purpose|Common Options|Description|
|---|---|---|---|
|`qsub <script.sh>`|**Submit a Job**|`-N <name>`|Assign a descriptive name to the job.|
|||`-cwd`|Execute the job from the current working directory (highly recommended).|
|||`-V`|Export all environment variables from the current shell to the job.|
|||`-q <queue_name>`|Select a specific queue (e.g., `all.q`).|
|||`-pe <pe> <slots>`|Request a Parallel Environment (`pe`) and a number of slots (cores).|
|||`-l h_rt=HH:MM:SS`|Hard runtime limit (wall clock time). The job is killed if this time is exceeded.|
|||`-l h_vmem=<size>`|Hard virtual memory limit (e.g., `4G` for 4GB). This is usually **per slot**.|
|||`-t 1-100`|Submit a job array (100 identical tasks indexed 1 through 100).|
|`qstat`|**Check Job Status**|`-u <user>`|Show jobs for a specific user (e.g., `-u $USER` or `-u \*` for all users).|
|||`-f`|Full output: shows queue status, resource usage, and job list.|
|||`-j <job_id>`|Display detailed information about a single job ID.|
|||`-t`|Show task-level detail for job arrays.|
|`qdel <job_id>`|**Delete/Kill a Job**|`-u <user>`|Delete all jobs submitted by a specific user.|
|||`<job_id>.1`|Delete a single task (`1`) from a job array (`job_id`).|
|||`-h`|Hold a pending job.|
|||`-U`|Release a held job.|
|`qacct`|**View Accounting**|`-j <job_id>`|Show detailed resource usage and execution metrics for a _completed_ job.|
|`qrsh`|**Interactive Shell**||Get an interactive shell on a compute node (useful for debugging).|

## 2. Job Status Codes (qstat output)

|Code|State|Description|
|---|---|---|
|**qw**|Queued Waiting|Job is submitted and waiting for resources (cores, memory, time).|
|**h**|Held|Job is manually held or held due to dependency/error (`hqw`, `hR`).|
|**r**|Running|Job is currently executing on a compute node.|
|**t**|Transferring|Job is being spooled to the execution host.|
|**s**|Suspended|Job is running but temporarily suspended.|
|**E**|Error|Job encountered an error, often due to configuration issues (`Eqw`).|
|**dr**|Deleting Running|Job is running, but deletion has been requested.|

## 3. QSUB Script Directives (`#$`)

In your job submission script (e.g., `my_job.sh`), lines starting with `#$` are read by `qsub` as command-line options.

|Directive|Command Line Equivalent|Purpose|
|---|---|---|
|`#$ -N my_job_name`|`qsub -N my_job_name`|Set the job name.|
|`#$ -cwd`|`qsub -cwd`|Run job in the current working directory.|
|`#$ -V`|`qsub -V`|Pass all environment variables.|
|`#$ -o logs/$JOB_NAME.$JOB_ID.out`|`qsub -o logs/...`|Redirect Standard Output (STDOUT) to a file.|
|`#$ -e logs/$JOB_NAME.$JOB_ID.err`|`qsub -e logs/...`|Redirect Standard Error (STDERR) to a file.|
|`#$ -j y`|`qsub -j y`|Join STDOUT and STDERR into a single output file (using the file specified by `-o`).|
|`#$ -q highmem.q`|`qsub -q highmem.q`|Specify the queue name.|
|`#$ -l h_rt=12:00:00`|`qsub -l h_rt=...`|Request a 12-hour maximum run time.|
|`#$ -l h_vmem=8G`|`qsub -l h_vmem=...`|Request 8GB of memory (often per core).|
|`#$ -pe smp 8`|`qsub -pe smp 8`|Request 8 cores (slots) using the 'smp' parallel environment.|
|`#$ -t 1-20:1`|`qsub -t 1-20:1`|Create a job array with 20 tasks, indexed from 1 to 20.|
|`#$ -tc 5`|`qsub -tc 5`|Limit the number of concurrently running tasks in a job array to 5.|
|`#$ -m be`|`qsub -m be`|Send email notification on **B**egin and **E**nd of the job.|
|`#$ -M your_email@domain.com`|`qsub -M ...`|Specify the email address for notifications.|
|`#$ -hold_jid 12345`|`qsub -hold_jid 12345`|Hold this job until job ID 12345 completes successfully.|

### Example Job Script (`run_analysis.sh`)

```
#!/bin/bash
# 
# --- SGE Directives ---
# Set the job name
#$ -N my_analysis_job
# Use the current working directory
#$ -cwd
# Request 4 cores for an OpenMP/Shared Memory job
#$ -pe smp 4
# Set hard runtime limit to 6 hours
#$ -l h_rt=06:00:00
# Request 4GB of memory per slot (16GB total)
#$ -l h_vmem=4G
# Merge stdout and stderr into the output file
#$ -j y
# Set output file path (using SGE variables)
#$ -o output/$JOB_NAME.$JOB_ID.out
# 
# --- Job Commands ---
echo "Starting job $JOB_ID on $(hostname)"
# Load any required modules
module load python/3.9
# Run your main program
python script/process_data.py --threads $NSLOTS --input data.csv
echo "Job finished at $(date)"
```

### Submitting the Example Job

```
qsub run_analysis.sh 
```