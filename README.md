# Vision-fused Attack: Advancing Aggressive and Stealthy Adversarial Text against Neural Machine Translation

Welcome to the official implementation for the IJCAI 2024 paper *Vision-fused Attack: Advancing Aggressive and Stealthy Adversarial Text against Neural Machine Translation*.

In this paper, we proposed a vision-fused attack (VFA) framework for generating powerful adversarial text. Our VFA uses the vision-merged solution space enhancement and perception-retained adversarial text selection strategy, producing more aggressive and stealthy adversarial text against NMT models. Extensive experiments demonstrated that VFA outperforms comparisons by significant margins both in attacking ability and imperceptibility enhancements. 

![framework](imgs/framework.png "framework")

## Requirements
Please configure the environment as follows:

```
conda create -n VFA python=3.8
conda activate VFA
pip install -r requirements.txt
```

## Model

Please download the following models from Huggingface:
- sentence-transformers/all-MiniLM-L6-v2
- hgl/chinese-bert-wwm-ext
- google-bert/bert-base-multilingual-cased
- Helsinki-NLP/opus-mt-en-zh
- Helsinki-NLP/opus-mt-zh-en
- Helsinki-NLP/opus-tatoeba-en-ja
- Helsinki-NLP/opus-mt-ja-en

Then only keep the model name folder and place it in `./model/`

## Running
Please execute the following command to complete the preparation work:
```
python pixel.py
python radical.py
python TIT.py
```
Then execute the following command to obtain the adversarial texts:
```
python main.py --src zh --tgt en --dataset wmt19 --device 0 --vision_constraint
```

Results will be saved in `./result/`

## Citation

If our work or this repo is useful for your research, please cite our paper as follows:

```
@inproceedings{ijcai2024p730,
  title     = {Vision-fused Attack: Advancing Aggressive and Stealthy Adversarial Text against Neural Machine Translation},
  author    = {Xue, Yanni and Hao, Haojie and Wang, Jiakai and Sheng, Qiang and Tao, Renshuai and Liang, Yu and Feng, Pu and Liu, Xianglong},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {6606--6614},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/730},
  url       = {https://doi.org/10.24963/ijcai.2024/730},
}
```