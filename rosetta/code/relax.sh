#!/bin/bash

# this script might be called from the root directory, but it does all its work in the rosetta working directory
cd ~/rosetta_working_dir

./3_10_rosetta_scripts.static.linuxgccrelease @flags_mutate -out:level 200
./3_10_residue_energy_breakdown.static.linuxgccrelease @flags_score -out:level 200
