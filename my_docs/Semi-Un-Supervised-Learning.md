---
layout: page
title: Semi-supervised or Unsupervised Learning
description: >

image: /assets/img/blog/steve-harvey.jpg
comment: true
---


:+1: means being highly related to my personal research interest. 
{:.message}






## :+1: [CVPR 2019-Label Propagation for Deep Semi-supervised Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.pdf)
**NOTE**: 
Semi-supervised learning is becoming increasingly important because it can combine data carefully labeled by humans with abundant unlabeled data to train deep neural networks. Classic methods on semi-supervised learning that have focused on transductive learning have not been fully exploited in the inductive framework followed by modern deep learning. The same holds for the manifold assumption—that similar examples should get the same prediction. <br />
In this work, we employ a transductive label propagation method that is based on the **manifold assumption to make predictions** on the entire dataset and use these predictions to generate pseudo-labels for the unlabeled data and train a deep neural network. At the core of the transductive method lies **a nearest neighbor graph of the dataset that we create based on the embeddings of the same network.** 
Therefore our learning process **iterates between these two steps.** We improve performance on several datasets especially in the few labels regime and show that our work is complementary to current state of the art.
{:.message}

## :+1: [NeurIPS 2017-Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://papers.nips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf)
**NOTE**: 
The recently proposed **Temporal Ensembling** has achieved state-of-the-art results in
several **semi-supervised** learning benchmarks. It maintains **an exponential moving average of label predictions on each training example,** and **penalizes predictions that are inconsistent with this target.** However, because the targets change only once
per epoch, Temporal Ensembling becomes unwieldy when learning large datasets.
To overcome this problem, we propose **Mean Teacher, a method that averages model weights instead of label predictions.** As an additional benefit, Mean Teacher
improves test accuracy and **enables training with fewer labels than Temporal Ensembling.** Without changing the network architecture, Mean Teacher achieves an
error rate of 4.35% on SVHN with 250 labels, outperforming Temporal Ensembling
trained with 1000 labels. We also show that a good network architecture is crucial
to performance. Combining Mean Teacher and Residual Networks, we improve
the state of the art on CIFAR-10 with 4000 labels from 10.55% to 6.28%, and on
ImageNet 2012 with 10% of the labels from 35.24% to 9.11%.
{:.message}


## :+1: [ICLR 2019-Unsupervised Learning via Meta-Learning](https://openreview.net/pdf?id=r1My6sR9tX)
**NOTE**: 
A central goal of unsupervised learning is to acquire representations from unlabeled data or experience that can be used for more effective learning of downstream tasks from modest amounts of labeled data. <br />
Many prior unsupervised learning works aim to do so by developing **proxy objectives based on reconstruction, disentanglement, prediction, and other metrics.** <br />
Instead, we develop an unsupervised meta-learning method that explicitly optimizes for the ability to learn a variety of tasks from small amounts of data. To do so, **we construct tasks from unlabeled data in an automatic way and run meta-learning over the constructed tasks.** Surprisingly, we find that, **when integrated with meta-learning, relatively simple task construction mechanisms, such as clustering embeddings**, lead to good performance on a variety of downstream, human-specified tasks. Our experiments across four image datasets indicate that our unsupervised meta-learning approach acquires a learning algorithm without any labeled data that is applicable to a wide range of downstream classification tasks, improving upon the embedding learned by four prior unsupervised learning methods.
{:.message}








