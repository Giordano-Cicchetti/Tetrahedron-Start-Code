<h2 align="center"> <a href="https://openreview.net/forum?id=ftGnpZrW7P">Tetrehedron StartCode</a></h2>



## Building Environment
GRAM is implemented based on Pytorch. We use Python-3.9 and Cuda-11.7. Other version could be also compatible. Other needed packages are listed in preinstall.sh.

```
conda create -n gram python=3.9
conda activate gram
sh preinstall.sh
```

## Download basic encoder's pretrained checkpoints
Make a dir named pretrained_weights under the main work dir.

1. Download evaclip weight:
```
wget -P pretrained_weights/clip/ https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt
```
2. Download beats weight from https://github.com/microsoft/unilm/tree/master/beats

3. Download bert weight:
```python
from transformers import BertModel, BertTokenizer
bert = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert.save_pretrained('pretrained_weights/bert/bert-base-uncased')
bert_tokenizer.save_pretrained('pretrained_weights/bert/bert-base-uncased')
```


The processed  pretrained_weights path should be as follows:
```
    ├── pretrained_weights
    │   ├── beats
    │   │   └── BEATs_iter3_plus_AS2M.pt
    │   ├── bert
    │   │   └── bert-base-uncased
    │   ├── clip
    │   │   └── EVA01_CLIP_g_14_psz14_s11B.pt
```

## DATASET

Download dataset and unzip it in your desired location.
Change paths to file locations in the configuration file you want to use in pretraining.
E.G. ./config/gram/finetune_cfg/retrieval-msrvtt.json

## TRAIN FROM SRATCH
If you want to pretrain a model from scratch on the MSRVTT dataset you should run the following command.

```
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/gram/finetune_cfg/retrieval-msrvtt.json \
--output_dir /PATH/where/to/save/ckpt \
```

TEST, add these lines
```
--mode 'testing' \
--checkpoint /PATH/TO/SAVED_CHECKPOINT.pt
```


## More on the code:
All the losses are stored in the ./utils/volume.py python file.

They are then used in ./model/gram.py from line 485 up to 586.

The evaluation pipeline is on ./evaluation/evaluation_mm.py

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@inproceedings{cicchetti2025gramian,
title={Gramian Multimodal Representation Learning and Alignment},
author={Giordano Cicchetti and Eleonora Grassucci and Luigi Sigillo and Danilo Comminiello},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=ftGnpZrW7P}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ispamm/GRAM&type=Date)](https://star-history.com/#ispamm/GRAM&Date)


## Third-Party Licenses

For the full list of third-party licenses used in this project, please see the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file.
