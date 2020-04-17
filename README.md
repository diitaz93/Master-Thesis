# Master-Thesis

The code and the model in this repository is a modified version of [DECAGON](https://github.com/mims-harvard/decagon), a convolutional neural network created by Marinka Zitnik and colaborators.<br>
To run the code:
1.  Download the database used in DECAGON from [here](http://snap.stanford.edu/decagon/) and save it in `/data/orig_data/`.<br>
2. run the bash script `data/drug_dataset.sh` with the number of desired side effects as a parameter. For example<br>
```./data/drug_dataset.sh 6```
3. If necesary, create an environment according to the list of [requirements](https://github.com/mims-harvard/decagon/blob/master/requirements.txt) stated by the original authors.<br>
4. Run the jupyter notebook `test.ipynb` to use real data. The python file `main.py` is the original version, which uses a toy dataset as an example.
