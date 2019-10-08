---
layout: post
title: Paper Summary on Distance Metric, Representation Learning
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

:+1: means being highly related to my personal research interest. 
{:.message}

## [Few-shot Learning](../../my_docs/few-shot.md) 
* ICLR 2018-Meta-Learning for Semi-Supervised Few-Shot Classification
* NeurIPS 2019-Unsupervised Meta Learning for Few-Show Image Classification
* NeurIPS 2019-Learning to Self-Train for Semi-Supervised Few-Shot Classification
* NeurIPS 2019-Adaptive Cross-Modal Few-shot Learning
* NeurIPS 2019-Cross Attention Network for Few-shot Classification
* NeurIPS 2019-Incremental Few-Shot Learning with Attention Attractor Networks
* ICML 2019-LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning
{:.message}

## [Large Output Spaces](../../my_docs/large-output-spaces.md)
* NeurIPS 2019-Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Output Spaces
* AISTATS 2019-Stochastic Negative Mining for Learning with Large Output Spaces
{:.message}

## [Poincaré, Hyperbolic, Curvilinear](../../my_docs/Poincare-Hyperbolic-Curvilinear.md)
* NeurIPS 2019-Multi-relational Poincaré Graph Embeddings
* NeurIPS 2019-Numerically Accurate Hyperbolic Embeddings Using Tiling-Based Models
* NeurIPS 2019-Curvilinear Distance Metric Learning
{:.message}



## [Wasserstein](../../my_docs/wasserstein.md)
* NeurIPS 2019-Generalized Sliced Wasserstein Distances 
* NeurIPS 2019-Tree-Sliced Variants of Wasserstein Distances
* NeurIPS 2019-Sliced Gromov-Wasserstein
* NeurIPS 2019-Wasserstein Dependency Measure for Representation Learning
{:.message}


## [Semi-supervised or Unsupervised Learning](../../my_docs/Semi-Un-Supervised-Learning.md)
* CVPR 2019-Label Propagation for Deep Semi-supervised Learning
* NeurIPS 2017-Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
* ICLR 2019-Unsupervised Learning via Meta-Learning
{:.message}




## [NeurIPS 2019-Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers](https://arxiv.org/pdf/1905.10630.pdf)
**NOTE**: 
In deep neural nets, lower level embedding layers account for a large portion of the total number of parameters.**Tikhonov regularization, graph-based regularization, and hard parameter sharing are approaches that introduce explicit biases into training in a hope to reduce statistical complexity.** Alternatively, we propose stochastically shared embeddings (SSE), a data-driven approach to regularizing embedding layers, which stochastically transitions between embeddings during stochastic gradient descent (SGD). Because SSE integrates seamlessly with existing SGD algorithms, it can be used with only minor modifications when training large scale neural networks. We develop two versions of SSE: SSE-Graph using knowledge graphs of embeddings; SSE-SE using no prior information. We provide theoretical guarantees for our method and show its empirical effectiveness on 6 distinct tasks, from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages. **We find that when used along with widely-used regularization methods such as weight decay and dropout, our proposed SSE can further reduce overfitting, which often leads to more favorable generalization results.** <br />
We conducted **experiments for a total of 6 tasks from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages.** 
{:.message}

