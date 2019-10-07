---
layout: page
title: Few-shot Learning
description: >
  
comment: true
---


:+1: means being highly related to my personal research interest. 
{:.message}

## :+1: [ICLR 2018-Meta-Learning for Semi-Supervised Few-Shot Classification](https://openreview.net/pdf?id=HJcSzz-CZ)
**NOTE**: 
In few-shot classification, we are interested in learning algorithms that train a classifier from only a handful of labeled examples. **Recent progress in few-shot classification has featured meta-learning,** in which a parameterized model for a learning algorithm is defined and trained on **episodes representing different classification problems, each with a small labeled training set and its corresponding test set.** In this work, **we advance this few-shot classification paradigm towards a scenario where unlabeled examples are also available within each episode.** We consider **two situations: one where all unlabeled examples are assumed to belong to the same set of classes as the labeled examples of the episode, as well as the more challenging situation where examples from other distractor classes are also provided.** To address this paradigm, we propose novel extensions of Prototypical Networks (Snell et al., 2017) that are augmented with the ability to use unlabeled examples when producing prototypes. These models are trained in an end-to-end way on episodes, to learn to leverage the unlabeled examples successfully. We **evaluate these methods on versions of the Omniglot and miniImageNet benchmarks, adapted to this new framework augmented with unlabeled examples.** We also propose **a new split of ImageNet**, consisting of a large set of classes, with a hierarchical structure. Our experiments confirm that our Prototypical Networks can learn to improve their predictions due to unlabeled examples, much like a semi-supervised algorithm would. <br />
TL;DR: **We propose novel extensions of Prototypical Networks that are augmented with the ability to use unlabeled examples when producing prototypes.** <br />
Related: [Overview of learning to learn](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/) <br />
[Code for "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"](https://github.com/cbfinn/maml)
{:.message}


## :+1: [NeurIPS 2019-Unsupervised Meta Learning for Few-Show Image Classification](https://arxiv.org/pdf/1811.11819.pdf)
**NOTE**: 
Few-shot or one-shot learning of classifiers for images or videos is an important next frontier in computer vision. The extreme paucity of training data means that the learning must start with a significant inductive bias towards the type of task to be learned. One way to acquire this is by meta-learning on tasks similar to the target task. However, **if the meta-learning phase requires labeled data for a large number of tasks closely related to the target task, it not only increases the difficulty and cost, but also conceptually limits the approach to variations of well-understood domains.** <br />
In this paper, we propose UMTRA, an algorithm that performs meta-learning on an unlabeled dataset in an unsupervised fashion, without putting any constraint on the classifier network architecture. **The only requirements towards the dataset are: sufficient size, diversity and number of classes, and relevance of the domain to the one in the target task. Exploiting this information, UMTRA generates synthetic training tasks for the meta-learning phase.** <br />
We evaluate UMTRA on few-shot and one-shot learning on **both image and video domains.** To the best of our knowledge, we are the first to evaluate meta-learning approaches on UCF-101. On the Omniglot and Mini-Imagenet few-shot learning benchmarks, UMTRA outperforms every tested approach based on unsupervised learning of representations, while alternating for the best performance with the recent CACTUs algorithm. Compared to supervised model-agnostic meta-learning approaches, UMTRA trades off some classification accuracy for a vast decrease in the number of labeled data needed. For instance, on the five-way one-shot classification on the Omniglot, we retain 85% of the accuracy of MAML, a recently proposed supervised meta-learning algorithm, while reducing the number of required labels from 24005 to 5.
{:.message}


## :+1: [NeurIPS 2019-Learning to Self-Train for Semi-Supervised Few-Shot Classification](https://arxiv.org/pdf/1906.00562.pdf)
**NOTE**: 
Few-shot classification (FSC) is challenging due to the scarcity of labeled training data (e.g. only one labeled data point per class). Meta-learning has shown to achieve promising results by learning to initialize a classification model for FSC. In this paper we propose a novel semi-supervised meta-learning method called learning to self-train (LST) that **leverages unlabeled data and specifically meta-learns how to cherry-pick and label such unsupervised data to further improve performance.** To this end, we train the LST model through a large number of semi-supervised few-shot tasks. **On each task, we train a few-shot model to predict pseudo labels for unlabeled data, and then iterate the self-training steps on labeled and pseudo-labeled data with each step followed by fine-tuning.** We additionally learn **a soft weighting network (SWN) to optimize the self-training weights of pseudo labels so that better ones can contribute more to gradient descent optimization.** We evaluate our LST method on two ImageNet benchmarks for semi-supervised few-shot classification and achieve large improvements over the state-of-the-art method. <br />
Code url: [https://github.com/xinzheli1217/learning-to-self-train](https://github.com/xinzheli1217/learning-to-self-train)
{:.message}


## [NeurIPS 2019-Adaptive Cross-Modal Few-shot Learning](https://arxiv.org/pdf/1902.07104.pdf)
**NOTE**: 
Metric-based meta-learning techniques have successfully been applied to few-shot classification problems. In this paper, we propose to leverage cross-modal information to enhance metric-based few-shot learning methods. Visual and semantic feature spaces have different structures by definition. **For certain concepts, visual features might be richer and more discriminative than text ones. While for others, the inverse might be true. Moreover, when the support from visual information is limited in image classification, semantic representations (learned from unsupervised text corpora) can provide strong prior knowledge and context to help learning.** Based on these two intuitions, we propose a mechanism that can adaptively combine information from both modalities according to new image categories to be learned. Through a series of experiments, we show that by this adaptive combination of the two modalities, our model outperforms current uni-modality few-shot learning methods and modality-alignment methods by a large margin on all benchmarks and few-shot scenarios tested. Experiments also show that **our model can effectively adjust its focus on the two modalities. The improvement in performance is particularly large when the number of shots is very small.** <br />
Code url: [https://github.com/ElementAI/am3](https://github.com/ElementAI/am3)
{:.message}

## [NeurIPS 2019-Cross Attention Network for Few-shot Classification]()

**NOTE**: 
Not available yet. 
{:.message}




## [NeurIPS 2019-Incremental Few-Shot Learning with Attention Attractor Networks](https://arxiv.org/pdf/1810.07218.pdf)
**NOTE**: 
Machine learning classifiers are often trained to recognize a set of pre-defined classes. However, in many real applications, it is often desirable to have the flexibility of learning additional concepts, without re-training on the full training set. This paper addresses this problem, **incremental few-shot learning, where a regular classification network has already been trained to recognize a set of base classes; and several extra novel classes are being considered, each with only a few labeled examples. After learning the novel classes, the model is then evaluated on the overall performance of both base and novel classes.** To this end, we propose a meta-learning model, the Attention Attractor Network, **which regularizes the learning of novel classes.** In each episode, we train a set of new weights to recognize novel classes until they converge, and we show that **the technique of recurrent back-propagation can back-propagate through the optimization process and facilitate the learning of the attractor network regularizer.** We demonstrate that **the learned attractor network can recognize novel classes while remembering old classes without the need to review the original training set**, outperforming baselines that do not rely on an iterative optimization process.
{:.message}

## :+1: [ICML 2019-LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning](http://proceedings.mlr.press/v97/li19c/li19c.pdf)
**NOTE**: 
In this work, we propose **a novel meta-learning
approach for few-shot classification**, which learns
**transferable prior knowledge across tasks** and
**directly produces network parameters for similar unseen tasks with training samples**. Our approach,
called LGM-Net, includes two key modules,
namely, TargetNet and MetaNet. The **TargetNet module is a neural network for solving a specific task** and the **MetaNet module aims at learning to generate functional weights for TargetNet by observing training samples.** We also present an intertask normalization strategy for the training process to **leverage common information shared across different tasks.** The experimental results on Omniglot and miniImageNet datasets
demonstrate that LGM-Net can effectively adapt to similar unseen tasks and achieve competitive performance, and the results on synthetic datasets
show that transferable prior knowledge is learned
by the MetaNet module via **mapping training data to functional weights.** **LGM-Net enables fast learning and adaptation** since no further tuning steps are required compared to other metalearning approaches. <br />
Code url: [https://github.com/likesiwell/LGM-Net/](https://github.com/likesiwell/LGM-Net/)
{:.message}



