# Trouble shooting 
If you have a problem and then you find a solution, please contact me (m.daghlian@ucl.ac.uk). I will try to incorporate it into the pipeline to make sure it doesn't happen again. Alternatively, I will list it here, as somewhere people can go for reference. 

---
## ssh problems:

- **Permission denied:** Make sure `~/.ssh/config` has correct permissions — `chmod 600 ~/.ssh/config` and `chmod 700 ~/.ssh/`.
- **Still asked for password:** Check that `ssh-copy-id` succeeded for *both* the gateway and `ucl-work`. The cluster step is often missed.

- **Another potential cause:** SSH key not loaded into the agent. If you run `ssh-add -l` and get ```no identities``` as output, this confirms it.
**Fix:**
```bash
ssh-add ~/.ssh/id_ed25519
```
Then retry `ssh ucl-work` as normal.

- **Still not working:** Maybe you need to be connected to ucl vpn see [https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn]


---
## rsync: `unexpected end of file` / child exit status 11

**Symptom**
```
rsync(XXXXX): error: unexpected end of file
rsync(XXXXX): warning: child XXXXX exited with status 11
```
**Cause**
macOS ships with `openrsync` (protocol 29), a BSD reimplementation that is incompatible with GNU rsync (protocol 31) running on Myriad. The mismatch causes a segfault on the remote end.
**Fix**
Install GNU rsync via Homebrew:
```bash
brew install rsync
```
Verify:
```bash
which rsync           # should be /opt/homebrew/bin/rsync
rsync --version | head -1  # should say rsync version 3.x.x
```
**Note**
The `WARNING: connection is not using a post-quantum key exchange algorithm` message that appears alongside this error is unrelated — it is a cosmetic SSH warning from UCL's login nodes and can be ignored.



---
## s02_coreg:```stdbuf not found on macOS```
**Fix**
Install coreutils to mac
```bash
brew install coreutils
```


---
## cvl_utils: edits don't seem to take effect

**Symptom**
You fix something in `cvl_utils/cvl_utils/preproc_func.py` (or another `cvl_utils` module), but a pipeline script keeps behaving like the *old* code.

**Cause**
If the repo lives on a Dropbox-synced volume, Dropbox can rewrite a file's mtime independently of its content. This confuses Python's `__pycache__` staleness check, so Python may keep running a stale compiled `.pyc` for that module instead of the edited source.

**Fix**
```bash
rm -rf cvl_utils/cvl_utils/__pycache__
```
Then re-run — Python recompiles automatically on the next import.


---
## s01_sdc_AFNI (Docker): `error while creating mount source path ... no such file or directory`

**Symptom**
```
docker: Error response from daemon: error while creating mount source path '/host_mnt/Users/.../<run_dir>': mkdir ... no such file or directory
```
raised from `run_docker` in `cvl_utils/cvl_utils/preproc_func.py`, even though the work directory exists on disk.

**Cause (tentative — wait and see if this recurs before trusting this fully)**
Looks like a Docker Desktop / VirtioFS race when mounting a work directory that was just created moments earlier by `os.makedirs`, especially when the path is reached through a symlink (e.g. a no-space symlink such as `dp-clean-link` pointing into a Dropbox folder). Not fully confirmed — a manual reproduction with a freshly created directory did *not* fail.

**Fix**
Simply re-running the command resolved it both times this was seen (2026-06-15), with no code change needed. If it becomes a frequent/blocking issue, consider adding a short retry-with-delay around the `docker run` call in `run_docker`, gated on this specific stderr message.