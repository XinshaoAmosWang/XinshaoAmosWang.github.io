---
layout: post
title: Paper Summary on Distance Metric, Representation Learning
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

Paper Notes on Distance Metric, Representation Learning
{:.message}

## [CVPR 2019-Label Propagation for Deep Semi-supervised Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.pdf)
**NOTE**: 
Semi-supervised learning is becoming increasingly important because it can combine data carefully labeled by humans with abundant unlabeled data to train deep neural networks. Classic methods on semi-supervised learning that have focused on transductive learning have not been fully exploited in the inductive framework followed by modern deep learning. The same holds for the manifold assumption—that similar examples should get the same prediction. <br />
In this work, **we employ a transductive label propagation method that is based on the manifold assumption to make predictions on the entire dataset and use these predictions to generate pseudo-labels for the unlabeled data and train a deep neural network.** At the core of the transductive method lies a nearest neighbor graph of the dataset that we create based on the embeddings of the same network. 
Therefore our learning process iterates between these two steps. We improve performance on several datasets especially in the few labels regime and show that our work is complementary to current state of the art.
{:.message}


## [NeurIPS 2019-Learning to Self-Train for Semi-Supervised Few-Shot Classification](https://arxiv.org/pdf/1906.00562.pdf)
**NOTE**: 
Few-shot classification (FSC) is challenging due to the scarcity of labeled training data (e.g. only one labeled data point per class). Meta-learning has shown to achieve promising results by learning to initialize a classification model for FSC. In this paper we propose a novel semi-supervised meta-learning method called learning to self-train (LST) that leverages unlabeled data and specifically meta-learns how to cherry-pick and label such unsupervised data to further improve performance. To this end, we train the LST model through a large number of semi-supervised few-shot tasks. On each task, we train a few-shot model to predict pseudo labels for unlabeled data, and then iterate the self-training steps on labeled and pseudo-labeled data with each step followed by fine-tuning. We additionally learn a soft weighting network (SWN) to optimize the self-training weights of pseudo labels so that better ones can contribute more to gradient descent optimization. We evaluate our LST method on two ImageNet benchmarks for semi-supervised few-shot classification and achieve large improvements over the state-of-the-art method. <br />
[Code url](https://github.com/xinzheli1217/learning-to-self-train)
{:.message}


## [ICLR 2018-Meta-Learning for Semi-Supervised Few-Shot Classification](https://openreview.net/pdf?id=HJcSzz-CZ)
**NOTE**: 
In few-shot classification, we are interested in learning algorithms that train a classifier from only a handful of labeled examples. Recent progress in few-shot classification has featured meta-learning, in which a parameterized model for a learning algorithm is defined and trained on episodes representing different classification problems, each with a small labeled training set and its corresponding test set. In this work, we advance this few-shot classification paradigm towards a scenario where unlabeled examples are also available within each episode. We consider two situations: one where all unlabeled examples are assumed to belong to the same set of classes as the labeled examples of the episode, as well as the more challenging situation where examples from other distractor classes are also provided. To address this paradigm, we propose novel extensions of Prototypical Networks (Snell et al., 2017) that are augmented with the ability to use unlabeled examples when producing prototypes. These models are trained in an end-to-end way on episodes, to learn to leverage the unlabeled examples successfully. We evaluate these methods on versions of the Omniglot and miniImageNet benchmarks, adapted to this new framework augmented with unlabeled examples. We also propose a new split of ImageNet, consisting of a large set of classes, with a hierarchical structure. Our experiments confirm that our Prototypical Networks can learn to improve their predictions due to unlabeled examples, much like a semi-supervised algorithm would. <br />
TL;DR: **We propose novel extensions of Prototypical Networks that are augmented with the ability to use unlabeled examples when producing prototypes.**
{:.message}

## [ICLR 2019-Unsupervised Learning via Meta-Learning](https://openreview.net/pdf?id=r1My6sR9tX)
**NOTE**: 
A central goal of unsupervised learning is to acquire representations from unlabeled data or experience that can be used for more effective learning of downstream tasks from modest amounts of labeled data. <br />
Many prior unsupervised learning works aim to do so by developing proxy objectives based on **reconstruction, disentanglement, prediction, and other metrics.** <br />
Instead, we develop an unsupervised meta-learning method that explicitly optimizes for the ability to learn a variety of tasks from small amounts of data. To do so, we construct tasks from unlabeled data in an automatic way and run meta-learning over the constructed tasks. Surprisingly, we find that, **when integrated with meta-learning, relatively simple task construction mechanisms, such as clustering embeddings**, lead to good performance on a variety of downstream, human-specified tasks. Our experiments across four image datasets indicate that our unsupervised meta-learning approach acquires a learning algorithm without any labeled data that is applicable to a wide range of downstream classification tasks, improving upon the embedding learned by four prior unsupervised learning methods.
{:.message}


## [NeurIPS 2019-Unsupervised Meta Learning for Few-Show Image Classification](https://arxiv.org/pdf/1811.11819.pdf)
**NOTE**: 
Few-shot or one-shot learning of classifiers for images or videos is an important next frontier in computer vision. The extreme paucity of training data means that the learning must start with a significant inductive bias towards the type of task to be learned. One way to acquire this is by meta-learning on tasks similar to the target task. However, if the meta-learning phase requires labeled data for a large number of tasks closely related to the target task, it not only increases the difficulty and cost, but also conceptually limits the approach to variations of well-understood domains.<br />
In this paper, we propose UMTRA, an algorithm that performs meta-learning on an unlabeled dataset in an unsupervised fashion, without putting any constraint on the classifier network architecture. The only requirements towards the dataset are: sufficient size, diversity and number of classes, and relevance of the domain to the one in the target task. Exploiting this information, UMTRA generates synthetic training tasks for the meta-learning phase.<br />
We evaluate UMTRA on few-shot and one-shot learning on both image and video domains. To the best of our knowledge, we are the first to evaluate meta-learning approaches on UCF-101. On the Omniglot and Mini-Imagenet few-shot learning benchmarks, UMTRA outperforms every tested approach based on unsupervised learning of representations, while alternating for the best performance with the recent CACTUs algorithm. Compared to supervised model-agnostic meta-learning approaches, UMTRA trades off some classification accuracy for a vast decrease in the number of labeled data needed. For instance, on the five-way one-shot classification on the Omniglot, we retain 85% of the accuracy of MAML, a recently proposed supervised meta-learning algorithm, while reducing the number of required labels from 24005 to 5.
{:.message}

## [NeurIPS 2019-Adaptive Cross-Modal Few-shot Learning]()
**NOTE**: 
Metric-based meta-learning techniques have successfully been applied to few-shot classification problems. In this paper, we propose to leverage cross-modal information to enhance metric-based few-shot learning methods. Visual and semantic feature spaces have different structures by definition. **For certain concepts, visual features might be richer and more discriminative than text ones. While for others, the inverse might be true. Moreover, when the support from visual information is limited in image classification, semantic representations (learned from unsupervised text corpora) can provide strong prior knowledge and context to help learning.** Based on these two intuitions, we propose a mechanism that can adaptively combine information from both modalities according to new image categories to be learned. Through a series of experiments, we show that by this adaptive combination of the two modalities, our model outperforms current uni-modality few-shot learning methods and modality-alignment methods by a large margin on all benchmarks and few-shot scenarios tested. Experiments also show that **our model can effectively adjust its focus on the two modalities. The improvement in performance is particularly large when the number of shots is very small.**
{:.message}

## [NeurIPS 2019-Cross Attention Network for Few-shot Classification]()

**NOTE**: 
Not available yet. 
{:.message}


## [NeurIPS 2019-Incremental Few-Shot Learning with Attention Attractor Networks](https://arxiv.org/pdf/1810.07218.pdf)
**NOTE**: 
Machine learning classifiers are often trained to recognize a set of pre-defined classes. However, in many real applications, it is often desirable to have the flexibility of learning additional concepts, without re-training on the full training set. This paper addresses this problem, **incremental few-shot learning, where a regular classification network has already been trained to recognize a set of base classes; and several extra novel classes are being considered, each with only a few labeled examples. After learning the novel classes, the model is then evaluated on the overall performance of both base and novel classes.** To this end, we propose a meta-learning model, the Attention Attractor Network, **which regularizes the learning of novel classes.** In each episode, we train a set of new weights to recognize novel classes until they converge, and we show that **the technique of recurrent back-propagation can back-propagate through the optimization process and facilitate the learning of the attractor network regularizer.** We demonstrate that **the learned attractor network can recognize novel classes while remembering old classes without the need to review the original training set**, outperforming baselines that do not rely on an iterative optimization process.
{:.message}

## [ICML 2019-LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning](http://proceedings.mlr.press/v97/li19c/li19c.pdf)
**NOTE**: 
In this work, we propose **a novel meta-learning
approach for few-shot classification**, which learns
**transferable prior knowledge across tasks** and
directly **produces network parameters for similar unseen tasks with training samples**. Our approach,
called LGM-Net, includes two key modules,
namely, TargetNet and MetaNet. The **TargetNet module is a neural network for solving a specific task** and the **MetaNet module aims at learning to generate functional weights for TargetNet by observing training samples.** We also present an intertask normalization strategy for the training process to leverage common information
shared across different tasks. The experimental
results on Omniglot and miniImageNet datasets
demonstrate that LGM-Net can effectively adapt
to similar unseen tasks and achieve competitive
performance, and the results on synthetic datasets
show that transferable prior knowledge is learned
by the MetaNet module via mapping training
data to functional weights. **LGM-Net enables fast learning and adaptation** since no further
tuning steps are required compared to other metalearning approaches.
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

