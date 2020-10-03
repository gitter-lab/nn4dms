#!/usr/bin/env bash

# directory containing arg files called 0.txt, 1.txt ... (x-1).txt
args_path="args/"

mkdir -p output/htcondor_logs

# echo some HTCondor job information
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "System: $(uname -spo)"
echo "_CONDOR_JOB_IWD: $_CONDOR_JOB_IWD"
echo "Cluster: $CLUSTER"
echo "Process: $PROCESS"
echo "RunningOn: $RUNNINGON"

# un-tar code and other data, etc
ls ./*.tar.gz | xargs -n1 tar -xzf

# this makes it easier to set up the environments, since the PWD we are running in is not $HOME
export HOME=$PWD
# set up anaconda python environment
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 > /dev/null
bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -p ~/miniconda3
export PATH=~/miniconda3/bin:$PATH

# set up appropriate cuda/cudnn/tensorflow environment
source ~/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# Install packages specified in the environment file
conda env create -f env.yml

# activate environment and list out packages
conda activate nn4dms
conda list

python code/regression.py @${args_path}"${PROCESS}.txt" --cluster="$CLUSTER" --process="$PROCESS"


