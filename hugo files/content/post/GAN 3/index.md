---
title: 'GAN 3'
subtitle: 'MNIST Linear GAN'
summary: 'MNIST Linear GAN'
authors: 
- admin
tags:
- Academic- ## Supervised learning
categories: ['Deep Learning', 'Python', 'PyTorch']
date: "2019-05-21T03:43:00Z"
lastmod: ""
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
image:
  caption: 'Image credit: [**Gimages**](https://unsplash.com/photos/CpkOjOcXdUY)'
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []


---

We saw an [Intro to GANs](https://shangeth.github.io/post/gan-1/) and the [Theory of Game between Generator and Discriminator](https://shangeth.github.io/post/gna-2/) in the previous posts. In this post we are going to implement and learn about how to train GANs in PyTorch. We will start with MNIST dataset and in the future posts we will implement different applications of GANs and also my research paper on one of the application of GANs.

So the task is to use the MNIST dataset to generate new MNIST alike data samples with GANs.
![](https://cdn-images-1.medium.com/max/1200/1*M2Er7hbryb2y0RP1UOz5Rw.png)

# Let's Code GAN

## Get the Data
Import all the necessary libraries like Numpy, Matplotlib, torch, torchvision.
```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms
```

Now lets get the MNIST data from the torchvision datasets.
```python
transform = transforms.ToTensor()
data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=512)
```

![](data:image/png;base64,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)

## The Model
As we have already seen in [Theory of Game between Generator and Discriminator](https://shangeth.github.io/post/gna-2/), the GAN models generally have 2 networks Discriminator D and Generator G.
We will code both of these network as seperate classes in PyTorch.
![](https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/gan-mnist/assets/gan_network.png)

### Discriminator
The discriminator is a just a classifier , which takes input images and classifies the images as real or fake generated images. So lets make a classifier network in PyTorch. 

```python
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(D, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)      
        
    def forward(self, x):
        # flatten image
        x = x.view(-1, 28*28)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = F.log_softmax(self.fc4(x))
        return out
```

The D network has 4 linear layers with leaky relu and dropout layers in between.

Here the input size will be 28*28*1 (size of MNIST image)\\
hidden dim can be anything of your choice.\\
output_size = 2 (real or fake)

I am also adding a log softmax in the end for computation purpose.

Lets make a Discriminator object
```python
D_network = D(28*28*1, 50, 2)
print(D_network)
```
output :
```
D(
  (fc1): Linear(in_features=784, out_features=200, bias=True)
  (fc2): Linear(in_features=200, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=50, bias=True)
  (fc4): Linear(in_features=50, out_features=2, bias=True)
  (dropout): Dropout(p=0.3)
)
```

### Generator
The Generator takes a random vector(z)(also called latent vector) and generates a sample image with a distribution close to the training data distribution. We want to upsample z to an image of size 1*28*28. Tanh was used as activation in the output layer(as used in the original paper) , but feel free to tru other activations and check which gives good result.

```python
class G(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(G, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.fc4 = nn.Linear(hidden_dim*4, output_size) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = F.tanh(self.fc4(x))
        return out
```
The G network architecture is same as D's architecture except now we upsample the z to 28*28*1 size image.
```python
G_network = G(100, 50, 1*28*28)
print(G_network)
```

```
G(
  (fc1): Linear(in_features=100, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=200, bias=True)
  (fc4): Linear(in_features=200, out_features=784, bias=True)
  (dropout): Dropout(p=0.3)
)
```

## Loss

The discriminator wants the probability of fake images close to 0 and the generator wants the probability of the fake images generated by it to be close to 1.

So we define 2 losses

* Real Loss (loss btw p and 1)
* Fake loss (loss btw p and 0)

p is the probability of image to be real.

* For Generator :
minimize real_loss(p) or p to be closer to 1. ie: fool generator by making realistic images.

* For Discriminator :
minimize real_loss + fake loss. ie: p of real image close to 1 and p of fake image close to 0.

```python
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    criterion = nn.NLLLoss()
    loss = criterion(D_out.squeeze(), labels.long().cuda())
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.NLLLoss()
    loss = criterion(D_out.squeeze(), labels.long().cuda())
    return loss
```

[label smoothing](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b) is also done for better convergence.

## Training

