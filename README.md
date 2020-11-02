# Running steps
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hao-wang/yolo3_rebuild/blob/master/driver.ipynb)

**Main steps being reliable, the detailed parameters may be outdated. See above jupyter-notebook for the 
latest version.**

Simplified version of https://github.com/zzh8829/yolov3-tf2
## Convert raw data to yolo-PASCAL style 
1. annotation xmls' converted & copied to Annotations
1. images preprocessed & copied to JPEGImages
```test
python fc2voc.py 
    --annot_source_dir FC_1.0_offline_extension/DB/annotation
    --annot_output_dir FC_offline/Annotations
    --image_source_dir FC_1.0_offline_extension/DB/images
    --image_output_dir FC_offline/JPEGImages
```

## Augmentation
1. Augment with imgaug
```aug
python 
```

## Generate training / validation / test dataset
Better to make sure the sizes' are multiples of *batch_size*.
```generate sets
python tools/gen_image_sets.py
```

## Turn data into .tfrecord files
```images/annotations --> .tfrecord
python voc2tfrecord.py
```

## Test everything is alright
```visualize
!python tools/visualize_dataset.py \
  --root_dir $data_root \
  --spec_dir FC_offline
```

## Train the model
```
!python train.py \
    --root_dir $data_root \
    --spec_dir FC_offline \
    --mode fit --transfer darknet \
    --batch_size 8 \
    --epochs 20 \
    --weights_num_classes 80 \
    --num_classes 7
```

## Serve the model
