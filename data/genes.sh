#!/bin/bash

awk -F "\"*,\"*" '{print $1}' orig_data/bio-decagon-combo.csv > temp_different_drugs_combo.csv
awk -F "\"*,\"*" '{print $2}' orig_data/bio-decagon-combo.csv >> temp_different_drugs_combo.csv
sort temp_different_drugs_combo.csv | uniq > col_drugs.csv
echo 'Total number of drugs in DDI interactions'
wc -l col_drugs.csv
rm temp*


awk -F "\"*,\"*" '{print $1}' orig_data/bio-decagon-targets-all.csv | sort | uniq > col_dti.csv
echo "Total number of drugs in DTI"
wc -l col_dti.csv

awk -F "\"*,\"*" '{print $2}' orig_data/bio-decagon-targets-all.csv | sort | uniq > col_pti.csv
echo "Total number of proteins in DTI"
wc -l col_pti.csv

awk -F "\"*,\"*" '{print $1}' orig_data/bio-decagon-mono.csv | sort | uniq > col_dse.csv
echo "Total number of drugs with SE"
wc -l col_dse.csv

awk -F "\"*,\"*" '{print $1}' orig_data/bio-decagon-ppi.csv > temp_genes.csv
awk -F "\"*,\"*" '{print $2}' orig_data/bio-decagon-ppi.csv >> temp_genes.csv
sort temp_genes.csv | uniq > col_prot.csv
echo "Total number of genes in PPI interactions"
wc -l col_prot.csv
echo "Number of proteins with features"
wc -l orig_data/proteins.csv
#while read p; do
    
#done <

rm col* temp*
