#!/bin/bash

# Define the name of the conda environment and the packages to install
ENV_NAME="introcomp"
PACKAGES="numpy=1.26.4 pandas=2.2.2 matplotlib=3.9.2 scipy=1.13.1 networkx=3.3 notebook nbstripout pip"
REPO_URL="https://github.com/kingilsildor/CLS-Epidemic_Models.git"

# Function to create a conda environment and install packages
create_conda_env() {
    eval "$(conda shell.bash hook)"
    # source ~/anaconda3/etc/profile.d/conda.sh	
    conda config --add channels defaults
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME --yes python=3.12
    echo "Activating conda environment: $ENV_NAME"
    conda activate $ENV_NAME
    echo "Installing packages: $PACKAGES"
    conda install $PACKAGES --yes
    # This package can't be downloaded another way
    pip install ndlib
}

# Function to clone the repository
clone_repo() {
    if [ -z "$1" ]; then
        # If no location is provided, ask for confirmation
        read -r -p "No location provided. Clone in the current directory? (y/n): " response
	case "$response" in
            [yY][eE][sS]|[yY])	
            echo "Cloning repository in the current directory..."
            git clone $REPO_URL
	    ;;
        *)
            echo "Clone operation aborted."
            exit 1
	    ;;
        esac
    else
        echo "Cloning repository at: $1"
        git clone $REPO_URL $1
    fi
    git init
    git remote add origin $REPO_URL
    git status
}

# Main script execution
create_conda_env
clone_repo "$1"

echo "Setup complete!"
