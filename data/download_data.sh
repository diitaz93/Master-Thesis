#!/bin/bash

# download_data.sh
#---------------------------------------------------------------------------- #
# This script downloads the network data used in DECAGON from their           #
# official website and saves it in the folder orig_data. If the folder        #
# already exists, it is assumed that the data is already downloaded.          #
#                                                                             #
# Author: Juan Sebastian Diaz, April 2020                                     #
#---------------------------------------------------------------------------- #

if [ ! -d "original_data" ]; then
    mkdir original_data
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
    wget https://raw.githubusercontent.com/diitaz93/data/master/proteins.csv?token=AC2US3RH3JASLN626UFGJUK6X2DCO
    rm *.tar.gz
    
else
    echo 'Data already downloaded'
    exit 0
fi
