>📋  A template README.md for code accompanying our paper. (The complete code and data will be updated after the publication of the paper!)

# Graph Mask Network

This repository is the official implementation of [Graph Mask Network]. 

>📋  In this study, we propose a graph mask network (GMN) method that utilizes a self-generative approach to improve the quality of graphs. The main principle of GMN is the activation of the masked edges of the input graph structure and the reconstruction of the new edges by means of a self-supervised strategy.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training and Evaluation

To train the model(s) in the paper, run this command:

```train
python mian.py --dataset <path_to_data>
```
