# Bandana: Masked Graph Autoencoder with Non-discrete Bandwidths

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

For ogbl-collab:
```shell
python train_link_ogb.py --use_cfg --device=<gpu_id>
```

## Node classification

```shell
python train_node.py --dataset=<dataset_name> --use_cfg --device=<gpu_id>
```
`<dataset_name>`: Cora, Citeseer, Pubmed, Photo, Computers, CS, Physics, ogbn-arxiv
