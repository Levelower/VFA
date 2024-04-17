# VFA

This is the official implementation of the paper "Vision-fused Attack: Advancing Aggressive and Stealthy Adversarial Text against Neural Machine Translation".

## Requirements
- pytorch==1.12.0
- transformers
- pypinyin
- pandas
- jieba
- scikit-learn
- sentencepiece
- nltk
- lpips
- faiss-cpu/faiss-gpu
- scikit-image
- mecab-python3
- fairseq
- sacrebleu
- datasets
- subword-nmt

## Model

Please download the following models from Huggingface:
- sentence-transformers/all-MiniLM-L6-v2
- hgl/chinese-bert-wwm-ext
- Helsinki-NLP/opus-mt-en-zh
- Helsinki-NLP/opus-mt-zh-en
- Helsinki-NLP/opus-tatoeba-en-ja
- Helsinki-NLP/opus-mt-ja-en

## Running
Please execute the following command to complete the preparation work:
```
python pixel.py
python radical.py
python TIT.py
```
Then execute the following command:
```
python main.py --dataset wmt19 --device 0 --vision_constraint
```