# 31st Place Solution for the 2021 Global Wheat Challenge

This solution finetunes a Mask R-CNN model with a Swin Transformer encoder, pretrained in a self-supervised manner using the [MoBY](https://arxiv.org/abs/2105.04553) framework, on the 2021 Global Wheat Head Detection (GWHD) dataset.

## Steps to Reproduce

1. Run the [EDA](EDA.ipynb) notebook to download the data, split the dataset and convert annotations to the COCO format.
2. Train the model with following command:

```
tools/dist_train.sh configs/gwhd/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_gwhd.py 1 --work-dir experiments/moby_swin_t_imnet_mask_rcnn_3x --cfg-options model.pretrained=https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_swin_t_300ep_pretrained.pth
```

3. Run the [inference](inference.ipynb) notebook to generate submission file and submit to the competition.

## References

[Transformer-SSL](https://github.com/SwinTransformer/Transformer-SSL)