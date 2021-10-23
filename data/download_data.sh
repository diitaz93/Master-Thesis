#!/bin/bash

# download_data.sh
#---------------------------------------------------------------------------- #
# This script downloads the network data used in DECAGON from their           #
# official website and saves it in the folder orig_data. It also creates the  #
# folders for the different data structures used in DECAGON.	              #
#                                                                             #
# Author: Juan Sebastian Diaz, April 2020                                     #
#---------------------------------------------------------------------------- #

if [ ! -d "original_data" ]; then
    mkdir original_data
fi

cd original_data
wget http://snap.stanford.edu/decagon/bio-decagon-ppi.tar.gz
tar -xf bio-decagon-ppi.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-targets.tar.gz
tar -xf bio-decagon-targets.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-targets-all.tar.gz
tar -xf bio-decagon-targets-all.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz
tar -xf bio-decagon-combo.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-mono.tar.gz
tar -xf bio-decagon-mono.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-effectcategories.tar.gz
tar -xf bio-decagon-effectcategories.tar.gz
rm *.tar.gz
echo 'Data already downloaded'

# Directories for data structures
cd ..
if [ ! -d "data_structures" ]; then
    mkdir data_structures
    cd data_structures
    mkdir DS
    mkdir DECAGON
    mkdir MINIBATCH
    mkdir BDM
fi

exit 0



