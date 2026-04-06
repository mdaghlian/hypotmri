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

