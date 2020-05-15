# Master-Thesis

The code and the model in this repository is a modified version of [DECAGON](https://github.com/mims-harvard/decagon), a convolutional neural network created by Marinka Zitnik and colaborators.<br>
To run the code:
1. Run the bash script `download_data.sh` in the folder `data` to download the original data.<br>
2. Run the python script `remove_outliers.py` in the folder `data`. This generates a closed and consistent dataset, removing unlinked nodes from the network.<br>
3. If you want to run the code with a simpler version of the dataset, a small (closed and consistent) subset of the data can be generated running the python script `small_dataset.py`, giving the desired number of **drug-drug interactions** as a parameter. For example<br>
```./small_dataset.py 6```
4. If necesary, create an environment according to the list of requierements stated in the file `requierements.txt`. A docker container with the environment and requirements to run the code can be found in this [Docker repository](https://hub.docker.com/repository/docker/diitaz93/python-decagon). *Note: the container does not support jupyter notebook*.<br>
5. Run the jupyter notebook `test.ipynb` or the python script `main_decagon.py` to use real data. The python file `toy_example.py` is the original version, which uses a toy dataset as an example.
