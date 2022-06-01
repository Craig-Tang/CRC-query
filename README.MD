# Code for WCF-Index
This repository is a reference implementation of the querying algorithms proposed in "Reliable Community Search in Dynamic Networks".

## Data prep
Data is organized in .gml format where each file represents a one-timestamp graph instance and each edge is attached with a "weight" attribute.

## Run the Code
Input the dataset, parameters and choose the algorithm to run the code

```
Dataset name(str): Bitcoin_otc
Theta(float): 0.1
K(int): 2
Query(str): 1
Alpha(float): 5
T_s(int): 0
T_e(int): 10
Type one number to chose the algorithm (int): [1]EEF; [2]WCF;: 2
Index construction time: 3.3389127254486084
Running time of WCF query: 0.4596281051635742
CRC identified with size 16 and time interval [3,4]
CRC output at:  Bitcoin_otc.output-0.1-2-1-5.0_WCF
```

## Example
![](/img/output0.png)
![](/img/output1.png)
![](/img/output2.png)
![](/img/output3.png)