#!/bin/bash

# remove_outliers.sh
#---------------------------------------------------------------------------- #
# This script selects a compatible subset of the DECAGON data removing        #
# datapoints considered as outliers. This scripts does 3 dataset depurations: #
# 1) Detecting and eliminating the drugs in the DDI database that are not     #
# present in the SSE database.                                                #
# 2) Detecting and elimintating the genes in DTI database that are not        #
# present in PPI database.                                                    #
# 3) Detect and eliminate the genes in the protein feature databases that are #
# not present in the PPI database.                                            #
# In this fashion, We form a fully connected database without outliers        #
# Note: The feature databases may be incomplete for now. This means that      #
# there may be genes or drugs in DECAGON databases without feature vectors    #
# The scripts gets as parameter the name of the datafile containing the DTI   #
# interactions.                                                               #
#                                                                             #
# col files: unique list of nodes of certain type                             #
# outlier files: lines of certain node not present in other datasets          #
# temp files: unimportant                                                     #
#                                                                             #
# Author: Juan Sebastian Diaz, April 2020                                     #
#---------------------------------------------------------------------------- #

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
#------------------------------- 1. DETECTION OF SSE OUTLIERS ------------------------------------#
awk -F "\"*,\"*" '{print $1}' modif_data/new-decagon-combo.csv > temp_different_drugs_combo.csv
awk -F "\"*,\"*" '{print $2}' modif_data/new-decagon-combo.csv >> temp_different_drugs_combo.csv
sort temp_different_drugs_combo.csv | uniq > col_drugs.csv
echo 'Total number of drugs in DDI interactions'
wc -l col_drugs.csv
while read p; do
    if grep -Fqw $p orig_data/bio-decagon-mono.csv
    then
	:
    else
	echo "$p" >> outliers_se.csv
    fi
done < col_drugs.csv
echo 'SSE Outliers detected. Number of outliers:'
wc -l outliers_se.csv
# ------------------------------ DELETION OF SSE OUTLIERS -------------------------------------#
while read p; do
    sed -in "/$p/d" modif_data/new-decagon-combo.csv
done < outliers_se.csv
echo 'SSE outliers deleted. New number of DDI interactions:'
wc -l modif_data/new-decagon-combo.csv
#------------------------------- 2. DETECTION OF DTI OUTLIERS ------------------------------------#
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
echo 'PPI Outliers detected. Number of outliers:'
wc -l outliers_dt.csv
# ------------------------------ DELETION OF DTI OUTLIERS -------------------------------------#
while read p; do
    sed -in "/$p/d" modif_data/new-decagon-targets.csv
done < outliers_dt.csv
echo 'Outliers deleted. New number of lines:'
wc -l modif_data/new-decagon-targets.csv
#------------------------------- 3. DETECTION OF PF OUTLIERS ------------------------------------#
awk -F "\"*,\"*" '{print $1}' modif_data/ppi_mini.csv > temp_genes.csv
awk -F "\"*,\"*" '{print $2}' modif_data/ppi_mini.csv >> temp_genes.csv
sort temp_genes.csv | uniq > col_prot.csv
echo "Total number of genes in PPI interactions"
wc -l col_prot.csv
while read p; do
    tail -n +2 orig_data/proteins.csv | grep -w $p >> modif_data/new_proteins.csv
done < col_prot.csv
echo "Number of proteins with features"
wc -l modif_data/new_proteins.csv
#---------------------------- TERMINATION -----------------------------------------------------#
rm col* temp* outliers*
