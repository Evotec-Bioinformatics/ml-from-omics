# Machine learning from omics data

This repository contains the code required to reproduce all results and figures presented in the book chapter "Machine learning from omics data" (Springer Protocols, 2021).

## Instructions

First, install Python 3.10.10 (newer versions may also work) and then clone this repository. Next, install the dependencies using pip (preferably in a new virtualenv):

````shell
pip install -r requirements.txt
````

Optionally, set the number of processes to use during the hyper-parameter search. The default is 4. If you use SLURM, this will be done automatically.

````shell
export SLURM_CPUS_ON_NODE=20
````

Now, you can `cd src/` and train the models:
````shell
python main.py
````

This will take a while. Note, that this script will also download and extract all required datasets and thus needs about 41 GB of disk space. Afterwards generate all four figures by calling:

````shell
python create_plots.py
````

Neither the SVM nor UMAP are fully deterministic methods. Hence, minor deviations from the published results are to be expected.