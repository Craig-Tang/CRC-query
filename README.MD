# Code for WCF-Index
This repository is a reference implementation of the querying algorithms proposed in "Reliable Community Search in Dynamic Networks".

## Data Source
Popular Temporal/Dynamic Graph Data are Available at:
* [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
* [Network Data Repository](https://networkrepository.com/networks.php)

## Data preparation
Data is organized in .gml format where each file represents a one-timestamp graph instance and each edge is attached with a "weight" attribute.

## Requirements
* Python 3
* networkx == 2.6.3
* click == 8.0.3


## Run the Code
Input the dataset name, parameters and choose the algorithm to run the code
```
python run.py
```
## Parameters
* Dataset name: name of the dataset folder, string 
* $\theta$ (Theta): parameter of the edge weight threshold, float number in [0,1].
* $k$ (K): parameter of the k-core constraint, integer
* $q$ (Query): query vertex, string format
* $\alpha$ (Alpha): parameter to balance the importance of community size and duration, positive float
* $T_s$ (T_s): starting timestamp of the query interval (included), integer
* $T_e$ (T_e): ending timestamp of the query interval (excluded), integer

Input:
```
Dataset name(str): Bitcoin_otc
Theta(float): 0.1
K(int): 2
Query(str): 1
Alpha(float): 5
T_s(int): 0
T_e(int): 10
Type one number to chose the algorithm (int): [1]EEF; [2]WCF;: 2
```
Output:
```
Index construction time: 3.3389127254486084
Running time of WCF query: 0.4596281051635742
CRC identified with size 16 and time interval [3,4]
CRC output at:  Bitcoin_otc.output-0.1-2-1-5.0_WCF
```

## Visualization Example
demo of (0.5, 4)-core community querying "funny" on dataset [Reddit](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

blue dash lines represent the
edges whose weights are less than 0.5, and black lines represent
the edges whose weights are no less than 0.5.

![](/img/output0.png)
![](/img/output1.png)
![](/img/output2.png)
![](/img/output3.png)