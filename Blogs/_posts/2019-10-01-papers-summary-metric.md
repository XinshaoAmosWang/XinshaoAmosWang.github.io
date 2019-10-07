---
layout: post
title: Paper Summary on Distance Metric, Representation Learning
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

Paper Notes on Distance Metric, Representation Learning <br />
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


## :+1: [CVPR 2019-Label Propagation for Deep Semi-supervised Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.pdf)
**NOTE**: 
Semi-supervised learning is becoming increasingly important because it can combine data carefully labeled by humans with abundant unlabeled data to train deep neural networks. Classic methods on semi-supervised learning that have focused on transductive learning have not been fully exploited in the inductive framework followed by modern deep learning. The same holds for the manifold assumption—that similar examples should get the same prediction. <br />
In this work, we employ a transductive label propagation method that is based on the **manifold assumption to make predictions** on the entire dataset and use these predictions to generate pseudo-labels for the unlabeled data and train a deep neural network. At the core of the transductive method lies **a nearest neighbor graph of the dataset that we create based on the embeddings of the same network.** 
Therefore our learning process **iterates between these two steps.** We improve performance on several datasets especially in the few labels regime and show that our work is complementary to current state of the art.
{:.message}

## :+1: [ICLR 2019-Unsupervised Learning via Meta-Learning](https://openreview.net/pdf?id=r1My6sR9tX)
**NOTE**: 
A central goal of unsupervised learning is to acquire representations from unlabeled data or experience that can be used for more effective learning of downstream tasks from modest amounts of labeled data. <br />
Many prior unsupervised learning works aim to do so by developing **proxy objectives based on reconstruction, disentanglement, prediction, and other metrics.** <br />
Instead, we develop an unsupervised meta-learning method that explicitly optimizes for the ability to learn a variety of tasks from small amounts of data. To do so, **we construct tasks from unlabeled data in an automatic way and run meta-learning over the constructed tasks.** Surprisingly, we find that, **when integrated with meta-learning, relatively simple task construction mechanisms, such as clustering embeddings**, lead to good performance on a variety of downstream, human-specified tasks. Our experiments across four image datasets indicate that our unsupervised meta-learning approach acquires a learning algorithm without any labeled data that is applicable to a wide range of downstream classification tasks, improving upon the embedding learned by four prior unsupervised learning methods.
{:.message}












## [NeurIPS 2019-Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers](https://arxiv.org/pdf/1905.10630.pdf)
**NOTE**: 
In deep neural nets, lower level embedding layers account for a large portion of the total number of parameters.**Tikhonov regularization, graph-based regularization, and hard parameter sharing are approaches that introduce explicit biases into training in a hope to reduce statistical complexity.** Alternatively, we propose stochastically shared embeddings (SSE), a data-driven approach to regularizing embedding layers, which stochastically transitions between embeddings during stochastic gradient descent (SGD). Because SSE integrates seamlessly with existing SGD algorithms, it can be used with only minor modifications when training large scale neural networks. We develop two versions of SSE: SSE-Graph using knowledge graphs of embeddings; SSE-SE using no prior information. We provide theoretical guarantees for our method and show its empirical effectiveness on 6 distinct tasks, from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages. **We find that when used along with widely-used regularization methods such as weight decay and dropout, our proposed SSE can further reduce overfitting, which often leads to more favorable generalization results.** <br />
We conducted **experiments for a total of 6 tasks from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages.** 
{:.message}



## [NeurIPS 2019-Metric Learning for Adversarial Robustness](https://arxiv.org/pdf/1909.00900.pdf)
**NOTE**: 
Deep networks are well-known to be fragile to adversarial attacks. Using several standard image datasets and established attack mechanisms, we conduct an empirical analysis of deep representations under attack, and find that the **attack causes the internal representation to shift closer to the "false" class. Motivated by this observation, we propose to regularize the representation space under attack with metric learning in order to produce more robust classifiers.** By carefully sampling examples for metric learning, our learned representation not only **increases robustness, but also can detect previously unseen adversarial samples.** Quantitative experiments show improvement of robustness accuracy by up to 4% and detection efficiency by up to 6% according to Area Under Curve (AUC) score over baselines.
{:.message}

