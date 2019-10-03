---
layout: post
title: Paper Summary on Distance Metric, Representation Learning
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

Paper Notes on Distance Metric, Representation Learning
{:.message}


## [NeurIPS 2019-Unsupervised Meta Learning for Few-Show Image Classification]()
**NOTE**: 

{:.message}

## [NeurIPS 2019-Adaptive Cross-Modal Few-shot Learning]()
**NOTE**: 

{:.message}

## [NeurIPS 2019-Cross Attention Network for Few-shot Classification]()
**NOTE**: 

{:.message}


## [NeurIPS 2019-Incremental Few-Shot Learning with Attention Attractor Networks](https://arxiv.org/pdf/1810.07218.pdf)
**NOTE**: 
Machine learning classifiers are often trained to recognize a set of pre-defined classes. However, in many real applications, it is often desirable to have the flexibility of learning additional concepts, without re-training on the full training set. This paper addresses this problem, incremental few-shot learning, where a regular classification network has already been trained to recognize a set of base classes; and several extra novel classes are being considered, each with only a few labeled examples. After learning the novel classes, the model is then evaluated on the overall performance of both base and novel classes. To this end, we propose a meta-learning model, the Attention Attractor Network, which regularizes the learning of novel classes. In each episode, we train a set of new weights to recognize novel classes until they converge, and we show that the technique of recurrent back-propagation can back-propagate through the optimization process and facilitate the learning of the attractor network regularizer. We demonstrate that the learned attractor network can recognize novel classes while remembering old classes without the need to review the original training set, outperforming baselines that do not rely on an iterative optimization process.
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


## [NeurIPS 2019-Tree-Sliced Variants of Wasserstein Distances](https://arxiv.org/abs/1902.00342)
**NOTE**: 
Wasserstein Distances
{:.message}


## [NeurIPS 2019-Sliced Gromov-Wasserstein](https://arxiv.org/pdf/1905.10124.pdf)
**NOTE**: 
Wasserstein Distances
{:.message}



## [NeurIPS 2019-Wasserstein Dependency Measure for Representation Learning](https://arxiv.org/pdf/1903.11780.pdf)
**NOTE**: 
Mutual information maximization has emerged
as a powerful learning objective for unsupervised
representation learning obtaining state-of-the-art
performance in applications such as object recognition, speech recognition, and reinforcement
learning. However, such approaches are fundamentally limited since a tight lower bound on mutual information requires sample size exponential
in the mutual information. This limits the applicability of these approaches for prediction tasks
with high mutual information, such as in video
understanding or reinforcement learning. In these
settings, such techniques are prone to overfit, both
in theory and in practice, and capture only a few
of the relevant factors of variation. This leads to
incomplete representations that are not optimal
for downstream tasks. In this work, we empirically demonstrate that mutual information-based
representation learning approaches do fail to learn
complete representations on a number of designed
and real-world tasks. To mitigate these problems
we introduce the Wasserstein dependency measure, which learns more complete representations
by using the Wasserstein distance instead of the
KL divergence in the mutual information estimator. We show that a practical approximation to
this theoretically motivated solution, constructed
using Lipschitz constraint techniques from the
GAN literature, achieves substantially improved
results on tasks where incomplete representations
are a major challenge.
{:.message}

