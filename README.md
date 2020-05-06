# Master-Thesis

The code and the model in this repository is a modified version of [DECAGON](https://github.com/mims-harvard/decagon), a convolutional neural network created by Marinka Zitnik and colaborators.<br>
To run the code:
1. Run the script `data/download_data.sh` to download the original data.<br>
2. run the bash script `data/drug_dataset.sh` with the number of desired side effects as a parameter. For example<br>
```./data/drug_dataset.sh 6```
3. If necesary, create an environment according to the list of requierements stated by the original authors in the file `requierements.txt`. A docker container with the requirements and a copy of the latest data can be found in this [Docker repository](https://hub.docker.com/repository/docker/diitaz93/python-decagon). *Note: the conatiner does not support jupyter notebook yet*.<br>
4. Run the jupyter notebook `test.ipynb` to use real data. The python file `main.py` is the original version, which uses a toy dataset as an example.
