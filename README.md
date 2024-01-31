# Multimodal-Sentiment-Analysis

## Repository structure

```
│-- data
│-- train.txt
│-- test_without_label.txt
│-- requirements.txt
│-- README.md
│-- final_concat.py
│-- result.txt 
```

## Setup

- numpy==1.26.0
- pandas==2.1.4
- pillow==9.0.1
- torch==2.1.1+cu118
- transformers==4.34.0
- scikit-learn==1.0.2

You can simply run

pip install -r requirements.txt

## Run

python final_concat.py --model pre_concat --lr 1e-5 --epochs 2

## Result

|  Model    |  accuracy   |
| ---- | ---- |
|   concat   |  70.5%    |
|   encode_concat    |    71.25%  |
|   pre_concat    |   71.88%   |
|   only_text    |   68.5%   |
|   only_img    |   61.5%   |

## Reference

Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Ethan Perez, Davide Testuggine. "Supervised Multimodal Bitransformers for Classifying Images and Text" (https://arxiv.org/abs/1909.02950)

https://github.com/facebookresearch/mmbt/blob/master/mmbt/models/mmbt.py
