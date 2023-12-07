# Bandana: Masked Graph Autoencoder with Non-discrete Bandwidths

# Update: supplementary figures

Please see the `supp` directory for the additional figures.

# Requirements

See `requirements.txt`.

# Reproduction

See `run.ipynb` for our experiment results. 
You can either run the model by this Jupyter file or by commands below in the terminal:

## Link prediction

```shell
python train_link.py --dataset=<dataset_name> --use_cfg --device=<gpu_id>
```
`<dataset_name>`: Cora, Citeseer, Pubmed, Photo, Computers, CS, Physics

```shell
python train_link_ogb.py --dataset=<dataset_name> --use_cfg --device=<gpu_id>
```
`<dataset_name>`: ogbl-collab, ogbl-ppa

## Node classification

```shell
python train_node.py --dataset=<dataset_name> --use_cfg --device=<gpu_id>
```
`<dataset_name>`: Cora, Citeseer, Pubmed, Photo, Computers, CS, Physics, ogbn-arxiv, ogbn-mag
