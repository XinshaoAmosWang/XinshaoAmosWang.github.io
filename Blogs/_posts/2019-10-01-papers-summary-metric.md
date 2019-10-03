---
layout: post
title: Paper Summary on Distance Metric, Representation Learning
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

Paper Notes on Distance Metric, Representation Learning
{:.message}

## [NeurIPS 2019-Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Output Spaces]()
**NOTE**: 
Not available yet. 
{:.message}


## [AISTATS 2019-Stochastic Negative Mining for Learning with Large Output Spaces](http://proceedings.mlr.press/v89/reddi19a/reddi19a.pdf)
**NOTE**: 
In this paper we specifically consider retrieval tasks
where the objective is to output the k most relevant
classes for an input out of a very large number of
possible classes. Training and test examples consist of
pairs (x, y) where x represents the input and y is one
class that is relevant for it. This setting is common
in retrieval tasks: for example, x might represent a
search query, and y a document that a user clicked on
in response to the search query. The goal is to learn
a set-valued classifier that for any input x outputs a
set of k classes that it believes are most relevant for x,
and the model is evaluated based on whether the class
y is captured in these k classes.
{:.message}


## [NeurIPS 2019-Multi-relational Poincaré Graph Embeddings](https://arxiv.org/pdf/1905.09791.pdf)
**NOTE**: 
Hyperbolic embeddings have recently gained attention in machine learning due to their ability to **represent hierarchical data more accurately and succinctly than their Euclidean analogues**. However, **multi-relational knowledge graphs often exhibit multiple simultaneous hierarchies, which current hyperbolic models do not capture.** To address this, we propose a model that embeds multi-relational graph data in the Poincaré ball model of hyperbolic space. Our Multi-Relational Poincaré model (MuRP) learns relation-specific parameters to transform entity embeddings by Möbius matrix-vector multiplication and Möbius addition. Experiments on the hierarchical WN18RR knowledge graph show that our multi-relational Poincaré embeddings outperform their Euclidean counterpart and existing embedding methods on the link prediction task, particularly at lower dimensionality. 
{:.message}


## [NeurIPS 2019-Numerically Accurate Hyperbolic Embeddings Using Tiling-Based Models]()
**NOTE**: 
Not available yet. <br />
Related work: <br />
&nbsp; &nbsp;  [HyperE: Hyperbolic Embeddings for Entities](https://hazyresearch.github.io/hyperE/).<br /> 
{:.message}

## [NeurIPS 2019-Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers](https://arxiv.org/pdf/1905.10630.pdf)
**NOTE**: 
In deep neural nets, lower level embedding layers account for a large portion of the total number of parameters.**Tikhonov regularization, graph-based regularization, and hard parameter sharing are approaches that introduce explicit biases into training in a hope to reduce statistical complexity.** Alternatively, we propose stochastically shared embeddings (SSE), a data-driven approach to regularizing embedding layers, which stochastically transitions between embeddings during stochastic gradient descent (SGD). Because SSE integrates seamlessly with existing SGD algorithms, it can be used with only minor modifications when training large scale neural networks. We develop two versions of SSE: SSE-Graph using knowledge graphs of embeddings; SSE-SE using no prior information. We provide theoretical guarantees for our method and show its empirical effectiveness on 6 distinct tasks, from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages. **We find that when used along with widely-used regularization methods such as weight decay and dropout, our proposed SSE can further reduce overfitting, which often leads to more favorable generalization results.** <br />
We conducted **experiments for a total of 6 tasks from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages.** 
{:.message}


## [NeurIPS 2019-Curvilinear Distance Metric Learning]()
**NOTE**: 
Not available yet. 
{:.message}

## [NeurIPS 2019-Metric Learning for Adversarial Robustness](https://arxiv.org/pdf/1909.00900.pdf)
**NOTE**: 
Deep networks are well-known to be fragile to adversarial attacks. Using several standard image datasets and established attack mechanisms, we conduct an empirical analysis of deep representations under attack, and find that the **attack causes the internal representation to shift closer to the "false" class. Motivated by this observation, we propose to regularize the representation space under attack with metric learning in order to produce more robust classifiers.** By carefully sampling examples for metric learning, our learned representation not only **increases robustness, but also can detect previously unseen adversarial samples.** Quantitative experiments show improvement of robustness accuracy by up to 4% and detection efficiency by up to 6% according to Area Under Curve (AUC) score over baselines.
{:.message}


## [NeurIPS 2019-Generalized Sliced Wasserstein Distances](https://arxiv.org/abs/1902.00434)
**NOTE**: 
Wasserstein Distances
{:.message}

