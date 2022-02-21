# CRC-query
This is a demo of the experiment of the work "Reliable Community Search in Dynamic Networks"

Data are collected from https://networkrepository.com and https://snap.stanford.edu/data/

## Data generation
Each graph instance is obtained by dividing the data into |T| partitions, and the weight is assigned according to the quantile of the frequency of the interaction between edges.

## Customize data
To use other dataset, simply organize your data as a list of graphs (nx.Graph()) with weight information ([0,1]). The length of the list is the number of timestamps.

## Experiment
Notebook shows the demo of experiment including varying parameters for EEF and WCF query, along with index construction, update and compression. The algorithm is in the python script reliable.py file.
