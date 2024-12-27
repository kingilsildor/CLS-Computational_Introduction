# Introduction to Computational Science
git repo based on the second assignment for the course Introduction to Computational Science.  

# Setup
## Setup the git repository localy
```sh
git clone https://github.com/kingilsildor/CLS-Epidemic_Models.git
cd CLS-Epidemic_Models
git init
git remote add origin https://github.com/kingilsildor/CLS-Epidemic_Models
'Make some changes'
git checkout -b <your-branch>
git add .
git commit -m "Your message"
git push origin <your-branch>
```

## Setup the conda environment
```sh
conda create -n introcomp python=3.12
conda activate introcomp
conda install numpy=1.26.4 pandas=2.2.2 matplotlib=3.9.2 scipy=1.13.1 notebook nbstripout
```
## Alternative
Run the config.sh file to both install the correct dependencies and initialize git. 
```sh
sh config.sh
```

# Editing files
When you want to push a notebook to the repo, clean the file first using:
```sh
nbstripout FILE.ipynb
```
Based on https://github.com/kynan/nbstripout. Packe will be already installed when doing the setup.

# Contributers
Tycho Stam (13303147)
Zainab  () 
