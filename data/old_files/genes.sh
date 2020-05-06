#!/bin/bash

############################# PROTEINS ######################################
tail -n +2 orig_data/bio-decagon-ppi.csv | awk -F "\"*,\"*" '{print $1}' > temp_genes.csv
tail -n +2 orig_data/bio-decagon-ppi.csv | awk -F "\"*,\"*" '{print $2}' >> temp_genes.csv
wc -l temp_genes.csv
sort -u temp_genes.csv > col_prot.csv
echo "Total number of proteins in PPI interactions"
wc -l col_prot.csv

tail -n +2 orig_data/bio-decagon-targets-all.csv | awk -F "\"*,\"*" '{print $2}' | sort | uniq > col_pti.csv
echo "Total number of proteins in DTI"
wc -l col_pti.csv
echo "Number of proteins with features"
tail -n +2 orig_data/proteins.csv | wc -l 

while read p; do
    if grep -Fqw $p col_prot.csv
    then
	:
    else
	echo "$p" >> outliers_prot_dti.csv
    fi
done < col_pti.csv
echo "There are $(sed -n '$=' outliers_prot_dti.csv) proteins in DTI that are not in PPI"

######################## NUMBER OF DRUGS ###########################
tail -n +2 orig_data/bio-decagon-targets-all.csv | awk -F "\"*,\"*" '{print $1}' | sort | uniq > col_dti.csv
echo "Total number of drugs in DTI"
wc -l col_dti.csv

tail -n +2 orig_data/bio-decagon-combo.csv | awk -F "\"*,\"*" '{print $1}' > temp_different_drugs_combo.csv
tail -n +2 orig_data/bio-decagon-combo.csv | awk -F "\"*,\"*" '{print $2}' >> temp_different_drugs_combo.csv
wc -l temp_different_drugs_combo.csv
sort temp_different_drugs_combo.csv | uniq > col_drugs.csv
echo 'Total number of drugs in DDI interactions'
wc -l col_drugs.csv

tail -n +2 orig_data/bio-decagon-mono.csv | awk -F "\"*,\"*" '{print $1}' | sort | uniq > col_dse.csv
echo "Total number of drugs with SE"
wc -l col_dse.csv

rm col* temp* outliers*
