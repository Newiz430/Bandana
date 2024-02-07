<h1><img align="center" height="50" src="https://cdn.wikirby.com/thumb/4/4c/KRtDLD_Bandana_Waddle_Dee.png/525px-KRtDLD_Bandana_Waddle_Dee.png">Bandana: Masked Graph Autoencoder with Non-discrete Bandwidths</h1>

This is the official source code repo of paper "Masked Graph Autoencoder with Non-discrete Bandwidths" in *TheWebConf(WWW) 2024*.

We explore a new paradigm of topological masked graph autoencoders with non-discrete masking strategies, named "bandwidths". We verify its effectiveness in learning network topology by both theory and experiment.

## Links

| :page_facing_up: [**Preprint version** (full version)](https://arxiv.org/abs/2402.03814) | :book: ~~Published version~~ | :eye_speech_bubble: [OpenReview](https://openreview.net/forum?id=0iwNrRRIiZ) | :speech_balloon: [Blog](https://zhuanlan.zhihu.com/p/681841195) |

## Requirements

See `requirements.txt`.

## Reproduction

See `run.ipynb` for our experiment results. 
You can either run the model by this Jupyter file or by commands below in the terminal:

### Link prediction

```shell
python train_link.py --dataset=<dataset_name> --use_cfg --device=<gpu_id>
```
> `<dataset_name>`: Cora, Citeseer, Pubmed, Photo, Computers, CS, Physics
>
> By `--use_cfg`, the best hyperparameters in the `config/<dataset_name>.yml` file are used by default.

```shell
python train_link_ogb.py --dataset=<dataset_name> --use_cfg --device=<gpu_id>
```
> `<dataset_name>`: ogbl-collab, ogbl-ppa

### Node classification

```shell
python train_node.py --dataset=<dataset_name> --use_cfg --device=<gpu_id>
```
> `<dataset_name>`: Cora, Citeseer, Pubmed, Photo, Computers, CS, Physics, Wiki-CS, ogbn-arxiv, ogbn-mag

## Citing

Please cite our paper for your research if our paper helps:

```
@inproceedings{bandana,
  title={Masked Graph Autoencoder with Non-discrete Bandwidths}, 
  author={Ziwen, Zhao and Yuhua, Li and Yixiong, Zou and Jiliang, Tang and Ruixuan, Li},
  booktitle={Proceedings of the 33rd ACM Web Conference},
  year={2024},
  month={May},
  publisher={Association for Computing Machinery},
  address={Singapore, Singapore},
}
```

## Special thanks

* [MaskGAE](https://github.com/EdisonLeeeee/MaskGAE)
