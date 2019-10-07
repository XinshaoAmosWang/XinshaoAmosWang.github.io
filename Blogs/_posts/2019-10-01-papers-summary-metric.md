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
**NOTE**: 
{:.message}


## [Large Output Spaces](../../my_docs/large-output-spaces.md)
**NOTE**: 
{:.message}


## [Poincaré, Hyperbolic, Curvilinear](../../my_docs/Poincare-Hyperbolic-Curvilinear.md)
**NOTE**: 
{:.message}

## [Wasserstein](../../my_docs/wasserstein.md)
**NOTE**: 
{:.message}


## [Semi-supervised or Unsupervised Learning](../../my_docs/Semi-Un-Supervised-Learning.md)
**NOTE**: 
{:.message}



## [NeurIPS 2019-Metric Learning for Adversarial Robustness](https://arxiv.org/pdf/1909.00900.pdf)
**NOTE**: 
Deep networks are well-known to be fragile to adversarial attacks. Using several standard image datasets and established attack mechanisms, we conduct an empirical analysis of deep representations under attack, and find that the **attack causes the internal representation to shift closer to the "false" class. Motivated by this observation, we propose to regularize the representation space under attack with metric learning in order to produce more robust classifiers.** By carefully sampling examples for metric learning, our learned representation not only **increases robustness, but also can detect previously unseen adversarial samples.** Quantitative experiments show improvement of robustness accuracy by up to 4% and detection efficiency by up to 6% according to Area Under Curve (AUC) score over baselines.
{:.message}


## [NeurIPS 2019-Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers](https://arxiv.org/pdf/1905.10630.pdf)
**NOTE**: 
In deep neural nets, lower level embedding layers account for a large portion of the total number of parameters.**Tikhonov regularization, graph-based regularization, and hard parameter sharing are approaches that introduce explicit biases into training in a hope to reduce statistical complexity.** Alternatively, we propose stochastically shared embeddings (SSE), a data-driven approach to regularizing embedding layers, which stochastically transitions between embeddings during stochastic gradient descent (SGD). Because SSE integrates seamlessly with existing SGD algorithms, it can be used with only minor modifications when training large scale neural networks. We develop two versions of SSE: SSE-Graph using knowledge graphs of embeddings; SSE-SE using no prior information. We provide theoretical guarantees for our method and show its empirical effectiveness on 6 distinct tasks, from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages. **We find that when used along with widely-used regularization methods such as weight decay and dropout, our proposed SSE can further reduce overfitting, which often leads to more favorable generalization results.** <br />
We conducted **experiments for a total of 6 tasks from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages.** 
{:.message}

