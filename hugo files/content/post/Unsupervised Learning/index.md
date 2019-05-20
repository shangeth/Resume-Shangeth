---
title: 'Unsupervised Learning 101'
subtitle: 'Intro to Unsupervised Learning'
summary: 'Intro to Unsupervised Learning'
authors: 
- admin
tags:
- Academic- ## Supervised learning
categories: ['Deep Learning']
date: "2019-05-20T00:00:00Z"
lastmod: "2019-05-21T00:00:00Z"
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

# Machine Learning is broadly divided into 3 types:


- ## Supervised learning

- ## Unsupervised learning

- ## Reinforcement learning

\
\
# Supervised Learning
  The task in supervised learning is to learn a function to map a data X to a label y. All the classification, regression, object detection/recognition/segmentation generally comes under supervised learning.

  ![](https://corochann.com/wp-content/uploads/2017/02/mnist_plot-800x600.png)

  ![](https://appliedmachinelearning.files.wordpress.com/2018/03/cifar2.jpg)
  In supervised learning we have a dataset which contains data X and label y and we need to learn how to find y given X.

# Unsupervised Learning
  In unsupervised learning, we only have X and not the respective y. The goal is to lean the underlying structure/features of the dataset without any lable.

Some examples of Unsupervised Leanring are 

- ## Clustering
  Clustering is dividing the data into groups through some distance metric like kmeans clustering.
  ![](https://www.imperva.com/blog/wp-content/uploads/sites/9/2017/07/k-means-clustering-on-spherical-data-1v2.png)

- ## Feature Learning
  As the name suggest, learning the features of each of the given data, without its label. This is generally done with a help of a model called Autoencoders.\\
  Autoencoders take the data X as the label y , it try to recreate the data X given data X and learns some underlying features int hat process. We generally take the one of the middle layer as the encoded feature.
  ![](https://cdn-images-1.medium.com/max/1200/1*j_y0bNZLP1yzqtyF48Z3Ug.png)

- ## Dimensionality Reduction
  As we know data can be multi dimensional which can extent even to millions. Computation and visualization of multi dimensioanl data is difficult and thus we want to reduce the dimension of the data (to pick the dimensions which can represent the data more).\\
  Dimensionality reduction is done by choosing the axis in the data space along which variance of the data is high.
  ![](https://static1.squarespace.com/static/5a316dfecf81e0076f50dae2/t/5ac35d702b6a284b3fde6131/1522753187751/PCA.png)

- and many other examples like data compression(using auto encoders), Generative models, density estimation, etc. 


# Why Unsupervised Learning?

- Unsupervised learning doen't need labels.
- Making the training data for supervised learning is not easy. Its expensive, time consuming, labour consuming.
- The world has a lot of unlabelled data, which can be used directly or with a little pre processing for unsupervised learning.

Unsupervised Learning is still an ameature area of research, which has a lot of potential. Unsupervised learning is less expensive and can accelerate the AI field so much.


