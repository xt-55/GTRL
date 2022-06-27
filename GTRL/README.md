# GTRL

This is the source code for paper [GTRL: An Entity Group-Aware Temporal Knowledge Graph Representation Learning Method]


## Data

We processed the ICEWS [1] and GDELT [2] and got three country based datasets:

GDELT18

ICEWS18

ICEWS14


## Prerequisites

- Python 3.7.7

- PyTorch 1.6.0

- dgl 0.5.0

- Sklearn 0.23.2

- Pandas 1.1.1


## Training and testing

Please run following commands for training and testing. We take the dataset `example` as the example.

**Event prediction**

python:

python train_event_predictor.py --runs 5 --dp ../data/ --gpu 1  -d example --seq-len 7


## Acknowledgements

This repo is based on Glean [3]. Great thanks to the original authors for their work!


## Cite

Please cite our paper if you find this code useful for your research.



## References

[1]	Kalev Leetaru and Philip A. Schrodt. 2013. GDELT: Global data on events, location, and tone, 1979-2012. ISA Annual Convention, 2(4): 1-49.

[2]	Elizabeth Boschee, Jennifer Lautenschlager, Sean O’Brien, Steve Shell-man, James Starz, and Michael Ward. 2015. ICEWS coded event data. Harvard Dataverse.

[3]	Songgaojun Deng, Rangwala Huzefa, and Ning Yue. 2020. Dynamic knowledge graph based multi-event forecasting. In KDD, 1585-1595.