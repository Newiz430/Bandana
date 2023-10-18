# Bandana: Masked Graph Autoencoder with Non-discrete Bandwidths

# Requirements

See `requirements.txt`.

# Reproduction

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
