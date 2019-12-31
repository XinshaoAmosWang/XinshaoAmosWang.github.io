---
layout: post
title: Foundations of Deep Learning, Machine Learning
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---

:+1: means being highly related to my personal research interest. 
{:.message}


## ICCV 2019: Robustness
* [Scalable Verified Training for Provably Robust Image Classification](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gowal_Scalable_Verified_Training_for_Provably_Robust_Image_Classification_ICCV_2019_paper.pdf)
* [Improving Adversarial Robustness via Guided Complement Entropy](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Improving_Adversarial_Robustness_via_Guided_Complement_Entropy_ICCV_2019_paper.pdf)
 * [Bilateral Adversarial Training: Towards Fast Training of More Robust Models
Against Adversarial Attacks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Bilateral_Adversarial_Training_Towards_Fast_Training_of_More_Robust_Models_ICCV_2019_paper.pdf)
* [Human uncertainty makes classification more robust](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peterson_Human_Uncertainty_Makes_Classification_More_Robust_ICCV_2019_paper.pdf)
* [Subspace Structure-aware Spectral Clustering for Robust Subspace Clustering](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yamaguchi_Subspace_Structure-Aware_Spectral_Clustering_for_Robust_Subspace_Clustering_ICCV_2019_paper.pdf)
{:.message}

## :+1: [NeurIPS 2019-Learning Data Manipulation for Augmentation and Weighting](https://papers.nips.cc/paper/9706-learning-data-manipulation-for-augmentation-and-weighting.pdf)
* Our
approach builds upon a recent connection of supervised learning and reinforcement
learning (RL), and adapts an off-the-shelf reward learning algorithm from RL for
joint data manipulation learning and model training. Different parameterization
of the “data reward” function instantiates different manipulation schemes.
* We
showcase data augmentation that learns a text transformation network, and data
weighting that dynamically adapts the data sample importance. Experiments show
the resulting algorithms significantly improve the image and text classification
performance in low data regime and class-imbalance problems.
{:.message}



## :+1: [ICLR 2019-Critical Learning Periods in Deep Networks](https://openreview.net/forum?id=BkeStsCcKQ)
* Counterintuitively, information rises rapidly in the early phases of
training, and then decreases, preventing redistribution of information resources in a phenomenon we refer to as a loss of “Information Plasticity”
* Our analysis suggests that the
first few epochs are critical for the creation of strong connections that are optimal relative
to the input data distribution. Once such strong connections are created, they do not appear
to change during additional training.
* The initial learning transient, under-scrutinized compared to asymptotic behavior, plays a key role in determining
the outcome of the training process. 
* The early transient is critical in determining the
final solution of the optimization associated with training an artificial neural network. In particular,
the effects of sensory deficits during a critical period cannot be overcome, no matter how much
additional training is performed.
* Our experiments show that, rather than helpful, pre-training can be detrimental, even if the
tasks are similar (e.g., same labels, slightly blurred images).
{:.message}

## :+1: [NeurIPS 2019-Time Matters in Regularizing Deep Networks: Weight Decay and Data Augmentation Affect Early Learning Dynamics, Matter Little Near Convergence](https://papers.nips.cc/paper/9252-time-matters-in-regularizing-deep-networks-weight-decay-and-data-augmentation-affect-early-learning-dynamics-matter-little-near-convergence.pdf)
* Regularization is typically understood as improving generalization by altering
the landscape of local extrema to which the model eventually converges. Deep
neural networks (DNNs), however, challenge this view: We show that removing
regularization after an initial transient period has little effect on generalization,
even if the final loss landscape is the same as if there had been no regularization.
* In some cases, generalization even improves after interrupting regularization. 
* Conversely, if regularization is applied only after the initial transient, it has no effect
on the final solution, whose generalization gap is as bad as if regularization never
happened.
* What matters for training deep networks is not just
whether or how, but when to regularize.
* The phenomena we observe are manifest
in different datasets (CIFAR-10, CIFAR-100, SVHN, ImageNet), different architectures (ResNet-18, All-CNN), different regularization methods (weight decay, data
augmentation, mixup), different learning rate schedules (exponential, piece-wise
constant). They collectively suggest that there is a “critical period” for regularizing
deep networks that is decisive of the final performance. More analysis should,
therefore, focus on the transient rather than asymptotic behavior of learning.
*  **Imposing regularization all along, however, causes over-smoothing**, whereas the ground-truth disparity field is typically discontinuous. So, **regularization is introduced initially and then removed to capture fine details.**
{:.message}


## [NeurIPS 2019-Inherent Weight Normalization in Stochastic Neural Networks](https://papers.nips.cc/paper/8591-inherent-weight-normalization-in-stochastic-neural-networks)
![Full-width image](/imgs/Inherent_weight_normalisation.png){:.lead data-width="200" data-height="100"}
{:.message}


## [NeurIPS 2019-Weight Agnostic Neural Networks](https://papers.nips.cc/paper/8777-weight-agnostic-neural-networks.pdf)
Not all neural network architectures are created equal, some perform much better
than others for certain tasks. But how important are the weight parameters of a
neural network compared to its architecture? In this work, we question to what
extent neural network architectures alone, without learning any weight parameters,
can encode solutions for a given task. We propose a search method for neural
network architectures that can already perform a task without any explicit weight
training. To evaluate these networks, we populate the connections with a single
shared weight parameter sampled from a uniform random distribution, and measure
the expected performance. We demonstrate that our method can find minimal neural
network architectures that can perform several reinforcement learning tasks without
weight training. On a supervised learning domain, we find network architectures
that achieve much higher than chance accuracy on MNIST using random weights.
Interactive version of this paper at [https://weightagnostic.github.io/](https://weightagnostic.github.io/)
{:.message}