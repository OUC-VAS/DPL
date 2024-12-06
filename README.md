# DPL: Cross-quality DeepFake Detection via Dual Progressive Learning

![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch)

This repo contains an official PyTorch implementation of our paper: [DPL: Cross-quality DeepFake Detection via Dual Progressive Learning.](https://openaccess.thecvf.com/content/ACCV2024/html/Zhang_DPL_Cross-quality_DeepFake_Detection_via_Dual_Progressive_Learning_ACCV_2024_paper.html)

[[pre-trained weights](https://drive.google.com/drive/folders/1nTfqa5ptvOBx6Fkixv98PSVTFjPKN4XV?usp=sharing)]

## 🌏Overview

<img src=".\images\overview.png" style="zoom: 85%;" />

## 🛠️ Setup

### ⚙️ Installation

You can run the following script to configure the necessary environment:

```sh
python -m venv venv/dpl
source venv/dpl/bin/activate
pip install -r requirements.txt
```

###  📑Data Preparation

Please refer to [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench/tree/ba02c47058062ae2dbea5450df277cba9bed7bc1/preprocessing) to generate the JSON file for each dataset for the unified data loading in the training and testing process, then put json files into `./dataset_json` folder.

The json file format is like below (see more details in `./dataset_json/demo.json`):

```json
{
    "FF-NT":{
        "FF-real":{
            "train":{
                "c40":{
                    "899":{
                        "label": "FF-real",
                        "frames":[
                            "FaceForensics++/original_sequences/youtube/c40/frames/899/258.png",
                            "FaceForensics++/original_sequences/youtube/c40/frames/899/352.png",
                            ...
                        ]
                    },
                    ...
                },
                "c23":{
                    ...
                }
            },
            ...
        },
        "FF-NT":{
            "train":{
                "c40":{
                    ...
                },
                "c23":{
                    ...
                }
            },
            ...
        }
    }
}
```

> ⭐️ You can directly download the processed data from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#2-download-data), but in order to load the data correctly, you will need to modify `abstract_dataset.py` and `pair_dataset.py`

## 🚀 Training and Testing

In the training phase, the model is trained on FF++(**c23**) dataset.

Run training script:

```sh
python train.py --detector_path ./config/dpl.yaml
```

When you want to train Stage I, simply modify the `train_stage` to 1 in `./config/dpl.yaml`. 

To train Stage II, change `train_stage` to 2 in `./config/dpl.yaml` and set the `checkpoint_path` to the optimal weight path from Stage I 「selected by running the `test_1.py`」.

> 🪧 **Regarding the selection of Stage I checkpoint weight for training Stage II:** <br/>
> Typically, the weight that achieves the best performance on the FF++c40 validation set during Stage I is considered the optimal weight. The optimal weight in Stage I is then used as the pretrained weight for Stage II.

Running the following script to evaluate the performance of the trained weights on FF++c40, and use the results to select the optimal weights:

```sh
python test_1.py
```

Run the following script to evaluate the performance of the selected weight on the dataset after applying four types of random JPEG compression:

```sh
python test.py
```



##  Citation
If you find our repo useful for your research, please consider citing our paper:
```latex
@InProceedings{li2024dpl,
    author    = {Zhang, Dongliang and Li, Yunfei and Zhou, Jiaran and Li, Yuezun},
    title     = {DPL: Cross-quality DeepFake Detection via Dual Progressive Learning},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    year      = {2024},
}
```
