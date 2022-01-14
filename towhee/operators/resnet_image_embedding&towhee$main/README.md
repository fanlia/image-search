# Resnet Operator

Authors: derekdqc, shiyu22

## Overview

This Operator generates feature vectors from the pytorch pretrained **Resnet** model[1], which is trained on [imagenet dataset](https://image-net.org/download.php).

**Resnet** models were proposed in “[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)”[2], this model was the winner of ImageNet challenge in 2015. "The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks were difficult due to the problem of vanishing gradients"[3].

## Interface

```python
__init__(self, model_name: str, framework: str = 'pytorch')
```

**Args:**

- model_name:
  - the model name for embedding
  - supported types: `str`, for example 'resnet50' or 'resnet101'
- framework:
  - the framework of the model
  - supported types: `str`, default is 'pytorch'

```python
__call__(self,  image: 'towhee.types.Image')
```

**Args:**

 image:
  - the input image
  - supported types: `towhee.types.Image`


**Returns:**

The Operator returns a tuple `Tuple[('feature_vector', numpy.ndarray)]` containing following fields:

- feature_vector:
  - the embedding of the image
  - data type: `numpy.ndarray`
  - shape: (dim,)

## Requirements

You can get the required python package by [requirements.txt](./requirements.txt).

## How it works

The `towhee/resnet-image-embedding` Operator implements the function of image embedding, which can add to the pipeline. For example, it's the key Operator named embedding_model within [image-embedding-resnet50](https://hub.towhee.io/towhee/image-embedding-resnet50) pipeline and [image-embedding-resnet101](https://hub.towhee.io/towhee/image-embedding-resnet101).

## Reference

[1].https://pytorch.org/hub/pytorch_vision_resnet/

[2].https://arxiv.org/abs/1512.03385

[3].https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
