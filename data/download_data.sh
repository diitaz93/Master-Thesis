#!/bin/bash

# download_data.sh
#---------------------------------------------------------------------------- #
# This script downloads the network data used in DECAGON from their           #
# official website and saves it in the folder orig_data. If the folder        #
# already exists, it is assumed that the data is already downloaded.          #
#                                                                             #
# Author: Juan Sebastian Diaz, April 2020                                     #
#---------------------------------------------------------------------------- #

if [ ! -d "orig_data" ]; then
    mkdir orig_data
    cd orig_data
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
    
else
    echo 'Data already downloaded'
    exit 0
fi
