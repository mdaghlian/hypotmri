# Setting up HPC Access (myriad)
If you are new to HPC computing, it is a good idea to take some time to learn the basics, you should be familiar with some basic terms and understand what they mean: queue; HPC; node; job; ssh; rsync... etc. You can learn about this here: [https://www.rc.ucl.ac.uk/docs/New_Users/], and also [https://github-pages.arc.ucl.ac.uk/hpc-intro/]

To get request access go here: [https://www.rc.ucl.ac.uk/docs/Account_Services/#apply-for-an-account]

General information [https://www.rc.ucl.ac.uk/docs/Clusters/]

Assuming you have read some of the above and got the general gist, Myriad is UCL's primary HPC cluster. Access is via SSH through a gateway node (`ssh-gateway.ucl.ac.uk`), since Myriad itself isn't directly reachable from outside UCL's network. The `ProxyJump` directive in the SSH config handles this automatically so you only need one command to log in.

---
## 1. Install OpenSSH & rsync

Both macOS & Linux should come with `ssh` already. Double check by running:
```bash
ssh 
rsync
```
If you get eith `command not found` for any either of these you need to install them. If not skip to [**section 2.**](#2-generate-ssh-keys-if-you-dont-have-them) 

For macOS use homebrew to do this. Also check it is installed: 
```bash
brew
```
If brew is not installed, install it with the following lines (& follow the instructions in cli)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Now we can actually install SSH
```bash
# macOS
brew install openssh
# Linux - same but use whatever package manager you normally do: e.g.,
apt-get install openssh
```

Make sure you also have rsync
```bash
rsync
# if not... same again
# For macOS
brew install rsync
# or for linux
apt-get install rsync 
```
---

## 2. Generate SSH Keys (if you don't have them)
This is like a password — it lets the server know you are safe and means you shouldn't have to type your password every time you connect.

First check whether you already have keys:
```bash
ls ~/.ssh/id_*.pub
```

If `.pub` files are listed, you already have keypairs — skip to step 3.

If not, generate them. **You need two keys**: one for the gateway (which supports modern key types) and one specifically for Myriad (which runs an older version of OpenSSH that only accepts RSA keys).

```bash
# Key for the gateway
ssh-keygen -t ed25519 -C "your_email@example.com"
# Accept the default file location (~/.ssh/id_ed25519)
# You do not need a passphrase — press enter to leave blank

# Key for Myriad (requires RSA due to old OpenSSH 7.4)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_ucl
# Press enter to leave passphrase blank
```

---

## 3. Configure SSH

Add the following to your `~/.ssh/config` file (create it if it doesn't exist). This sets up aliases so you can connect with a single short command, and routes traffic through the gateway automatically via `ProxyJump`.

```text
# The Gateway (UCL's external-facing SSH jump host)
Host ucl-gateway
    HostName ssh-gateway.ucl.ac.uk
    User <userid>
    IdentityFile ~/.ssh/id_ed25519
    AddKeysToAgent yes

# Myriad cluster (connected via gateway)
Host ucl-work
    HostName myriad.rc.ucl.ac.uk
    User <userid>
    ProxyJump ucl-gateway
    IdentityFile ~/.ssh/id_rsa_ucl
    ForwardAgent yes
```

Replace `<userid>` with your UCL user ID (e.g. `ucjvabc`).

> **Why two different keys?** Myriad runs OpenSSH 7.4, which is too old to accept ed25519 keys — it silently rejects them and falls back to asking for your password. The gateway runs a newer version and handles ed25519 fine. Using `ForwardAgent yes` means your Mac's ssh-agent passes the right key through to Myriad automatically.

---

## 4. Copy Your SSH Keys to the Cluster

This installs your public keys on the remote machines so you can log in without a password. You need to do this for both the gateway and Myriad separately.

```bash
# [1] Copy ed25519 key to the gateway
ssh-copy-id -i ~/.ssh/id_ed25519.pub <userid>@ssh-gateway.ucl.ac.uk

# [2] Copy RSA key to Myriad (routed through the gateway via your config)
ssh-copy-id -i ~/.ssh/id_rsa_ucl.pub ucl-work

# Enter your UCL password when prompted for each
```

Then fix permissions on Myriad (log in with your password one last time):
```bash
ssh ucl-work
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
exit
```

> **What this does:** `ssh-copy-id` appends your public key to `~/.ssh/authorized_keys` on the remote machine, enabling passwordless login going forward.

---

## 5. Log In
```bash
ssh ucl-work
```

This connects to Myriad via the gateway in one step. You should not be prompted for a password if key copying succeeded!

Now have a play around on the cluster:
```bash
ls          # see your folders 
ml avail    # see all the modules available on the cluster
```

--- 

## 6. Setup pipeline on cluster
**ASSUMING YOU ALREADY SETUP LOCALLY**:
Copy your local code to the cluster
```bash
rsync_code.sh
# which runs...
# rsync -avz /local/path/to/pipeline/ ucl-work:~/pipeline
```

Log in to add conda module:
```bash 
ssh ucl-work
ml python/miniconda3/24.3.0-0
conda init # This adds conda to .bashrc file 
# exit and re-enter to activate conda
exit
ssh ucl-work
which conda
```

Next add the important commands to your `~/.bash_profile`:

```bash
# To open & edit the bash profile run 
nano ~/.bash_profile
```
Go down to the bottom and copy paste the following:
```bash
# Set the following variables to be cluster friendly (i.e., not docker)
export PC_LOCATION="HPC"
export CONTAINER_TYPE="apptainer" # docker or singularity
export PYPACKAGE_MANAGER="conda"
export PIPELINE_DIR="/home/<ucl-id>/pipeline"
source "${PIPELINE_DIR}/config/config_pipeline.sh"

ml apptainer/1.2.4-1
# Apptainer build and cache directories
export APPTAINER_TMPDIR="$HOME/Scratch/.apptainer/tmp"
[[ ! -d "$APPTAINER_TMPDIR" ]] && mkdir -p "$APPTAINER_TMPDIR"

export APPTAINER_CACHEDIR="$HOME/Scratch/.apptainer"
[[ ! -d "$APPTAINER_CACHEDIR" ]] && mkdir -p "$APPTAINER_CACHEDIR"

REMOTE_PROJECT_DIRS="$HOME/Scratch/projects/"
[[ ! -d "$REMOTE_PROJECT_DIRS" ]] && mkdir -p "$REMOTE_PROJECT_DIRS"
```

To save press `ctrl+x`.

To apply the changes run `source ~/.bash_profile`. You do not need to do this every time you login — it will be done automatically.

## 7. Add python environments on the cluster
Just as you ran locally you can install the python environments with conda:
```bash
cd $PIPELINE_DIR/config
bash s00_python_environments.sh
```

## 8. Add the singularity images 
```bash
cd $PIPELINE_DIR/config
bash s00_containers.sh 
```

---

# WORK IN PROGRESS 

---

# How the pipeline (will) integrate with the cluster 
[1] Setup cluster access (see above)
[2] Install pipeline to cluster
[3] Per stage, add a flag for running on the cluster. This will trigger
- rsync -> relevant files from local to cluster
- creation of a 
- Second it will be necessary to install this package on the cluster
- Finally for each stage 
For each stage in the pipeline an additional  

# TODO
- [ ] Get the installation stuff all in one line script? 
- [ ] Pull out submission stuff from s01_hpc_submit.sh and make it more general purpose