#!/bin/bash

# drug_dataset.sh
#--------------------------------------------------------------------------------#
# Script for selecting a small subset of the database for testing the code       #
# and the model.It needs the number of drug-target interactions as parameter     #
# and it will generate a database with all drugs and proteins directly involved  #
# in the given interactions.                                                     #
#                                                                                #
# Author: Juan Sebastian Diaz, April 2020                                        #
#--------------------------------------------------------------------------------#

# ------------------------------------PREPARATION ---------------------------------------------#
# Write here the original target file. Choose between bio-decagon-targets and bio-decagon-targets-all
./remove_outliers.sh orig_data/bio-decagon-targets-all.csv
rm modif_data/*mini.csv se.csv unique* temp* col* 2> /dev/null
echo "Cleared space from old mini files"
N_se=$1 # Number of drug interactions that we want
# -------------------------------- 1.CHOOSING OF SIDE EFFECTS -----------------------------------#
# Choosing from combo
awk -F "\"*,\"*" '{print $3}' orig_data/bio-decagon-combo.csv| tail -n +2 | sort | uniq | shuf -n $N_se > se.csv
echo 'Side effects chosen'
wc -l se.csv
#------------------------------- 2.CHOOSING OF DRUG INTERACTIONS --------------------------------#
while read p; do
    grep $p -w modif_data/new-decagon-combo.csv >> modif_data/combo_mini.csv
    # It is possible to include a head statement between the previous line
    # to select a number of interaction of each type
done < se.csv
echo 'Drug-drug interactions chosen'
wc -l modif_data/combo_mini.csv
#---------------------------- 3.DETERMINATION OF NUMBER OF DRUGS --------------------------------#
awk -F "\"*,\"*" '{print $1}' modif_data/combo_mini.csv > col_drugs.csv
awk -F "\"*,\"*" '{print $2}' modif_data/combo_mini.csv >> col_drugs.csv
sed 's/\x0D$//' col_drugs.csv | sort | uniq > unique_drugs.csv
echo 'Unique drugs chosen'
wc -l unique_drugs.csv
#---------------------------- 4.CHOOSING INDIVIDUAL SIDE EFFECTS --------------------------------#
while read p; do
    grep $p -Fw orig_data/bio-decagon-mono.csv | head -5 >> modif_data/mono_mini.csv
done < unique_drugs.csv
echo "Individual side effects chosen"
wc -l modif_data/mono_mini.csv
#-------------------------------- 5.CHOOSING OF DTIs --------------------------------------------#
while read p; do
    grep $p -w modif_data/new-decagon-targets.csv >> modif_data/target_mini.csv
done < unique_drugs.csv
echo 'Drug-target interactions chosen'
wc -l modif_data/target_mini.csv
#--------------------------------- 6.CHOOSING OF PPIs -------------------------------------------#
awk -F "\"*,\"*" '{print $2}' modif_data/target_mini.csv | sed 's/\x0D$//' | sort | uniq > col_genes.csv
echo 'DTI genes'
wc -l col_genes.csv
while read p; do
    grep $p -w orig_data/bio-decagon-ppi.csv | head -2 >> temp_ppi.csv
done < col_genes.csv
sort temp_ppi.csv | uniq > modif_data/ppi_mini.csv
echo 'Protein-protein interactions chosen'
wc -l modif_data/ppi_mini.csv
#---------------------------- 7.DETERMINATION OF NUMBER OF GENES -------------------------------#
awk -F "\"*,\"*" '{print $1}' modif_data/ppi_mini.csv > col_genes.csv
awk -F "\"*,\"*" '{print $2}' modif_data/ppi_mini.csv >> col_genes.csv
sed 's/\x0D$//' col_genes.csv | sort | uniq > unique_genes.csv
echo 'Unique genes'
wc -l unique_genes.csv
#--------------------------- TERMINATION -----------------------------------------------------#
rm se.csv unique* temp* col* 2> /dev/null
