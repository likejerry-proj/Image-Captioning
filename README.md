# Image-Captioning
A tutorial on image captioning running on latest PyTorch version.

## Introduction
### Dataset
MSCOCO 2014.
### Encoder
Use pre-trained ResNet-101.
### Decoder
Use LSTM with Attention encoded image.

## How to use
### Configuration 
First clone the repository locally and enter the current directory.
```
git clone https://github.com/likejerry-proj/Image-Captioning.git
cd .\Image-Captioning\
```

Then install related packages.
```
conda create --name img-captioning python=3.9
conda activate img-captioning
pip install -r requirements.txt
```

### Dataset
Download [MSCOCO 2014 Training/13GB](https://cocodataset.org/#download), [MSCOCO 2014 Validation/6GB](https://cocodataset.org/#download).

I would suggest creating a new  'caption_data'  folder and putting the dataset in it. Be careful to the file path and remember to change it if necessary.

Then run datasets.py to create several .json and .hd5f files.
```
python datasets.py
```

### Train
I use 3000 batches for training which is up to 17702, change it if you want.

```
python train.py
```


### Caption a image with beam search
After training we can caption one image for instance. Change the img path if necessary.
```
python caption.py --img='caption_data/val2014/COCO_val2014_000000000285.jpg' --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='caption_data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5
```


