# Make sure that you have set up the correct conda environments & docker files
- change any settings in /config/config_pipeline.sh to match your system
- add the following to your ~/.bash_profile file
```bash
# Change it so that the path is correct
source "/path/to/this/repository/config/config_pipeline.sh"
```
- make sure the license.txt file for freesurfer is in your config folder
- Open a new terminal
- cd to the 'config' folder 
- run ```bash s00_docker.sh``` and ```bash s00_pythone_environments.sh``` to load the python + docker stuff you need 
