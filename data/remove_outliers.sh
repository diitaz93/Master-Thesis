#!/bin/bash

# remove_outliers.sh
#--------------------------------------------------------------------------- #
# Script for detecting and elimintating the genes in DTI database that are   #
# not present in PPI database and the drugs in the DDI database that are not #
# present in the SSE database. In this fashion, all databases have the same  #
# elements (nodes).                                                          #
# The scripts gets as parameter the name of the datafile containing the DTI  #
# interactions.                                                              #
#                                                                            #
# Author: Juan Sebastian Diaz, April 2020                                    #
#----------------------------------------------------------------------------#

#---------------------------------- PREPARATION -----------------------------------------------#
TARGET=$1
if [ ! -d "orig_data" ]; then
    echo "Download original data and place it in ./orig_data/"
    exit 1
fi
if [ ! -d "modif_data" ]; then
    mkdir modif_data
else
    rm modif_data/new* 2> /dev/null
fi
rm outliers* 2> /dev/null
tail -n +2 $TARGET > modif_data/new-decagon-targets.csv
tail -n +2 orig_data/bio-decagon-combo.csv > modif_data/new-decagon-combo.csv
echo 'Original number of drug-target interactions'
wc -l modif_data/new-decagon-targets.csv
echo "Original number of drug interactions"
wc -l modif_data/new-decagon-combo.csv
#------------------------------- DETECTION OF SSE OUTLIERS ------------------------------------#
awk -F "\"*,\"*" '{print $1}' modif_data/new-decagon-combo.csv > col_drugs.csv
awk -F "\"*,\"*" '{print $2}' modif_data/new-decagon-combo.csv >> col_drugs.csv
sort col_drugs.csv | uniq > temp_different_drugs_combo.csv
echo 'Total number of drugs in DDI interactions'
wc -l temp_different_drugs_combo.csv
while read p; do
    if grep -Fqw $p orig_data/bio-decagon-mono.csv
    then
	:
    else
	echo "$p" >> outliers_se.csv
    fi
done < temp_different_drugs_combo.csv
echo 'SSE Outliers detected. Number of outliers:'
wc -l outliers_se.csv
# ------------------------------ DELETION OF SSE OUTLIERS -------------------------------------#
while read p; do
    sed -in "/$p/d" modif_data/new-decagon-combo.csv
done < outliers_se.csv
echo 'Outliers deleted. New number of lines:'
wc -l modif_data/new-decagon-combo.csv
#------------------------------- DETECTION OF DTI OUTLIERS ------------------------------------#
awk -F "\"*,\"*" '{print $2}' modif_data/new-decagon-targets.csv | sort | uniq > col_genes.csv
echo 'Total number of target genes'
wc -l col_genes.csv
while read p; do
    if grep -Fqw $p orig_data/bio-decagon-ppi.csv
    then
	:
    else
	echo "$p" >> outliers_dt.csv
    fi
done < col_genes.csv
echo 'Outliers detected. Number of outliers:'
wc -l outliers_dt.csv
# ------------------------------ DELETION OF DTI OUTLIERS -------------------------------------#
while read p; do
    sed -in "/$p/d" modif_data/new-decagon-targets.csv
done < outliers_dt.csv
echo 'Outliers deleted. New number of lines:'
wc -l modif_data/new-decagon-targets.csv
#---------------------------- TERMINATION -----------------------------------------------------#
rm col* temp* outliers*
