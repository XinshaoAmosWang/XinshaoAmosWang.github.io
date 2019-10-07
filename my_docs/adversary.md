---
layout: page
title: GAN, Adversary Examples, Adversary Machine Learning
description: >
  
comment: true
---


:+1: means being highly related to my personal research interest. 
{:.message}


## [Adversarial Examples Reading List](../../my_docs/Adversarial-Examples-Reading-List.md)
**NOTE**: 
To be done *55 papers with key words on Adversarial from NeurIPS 2019*.
{:.message}


## [ICML 2019-Improving Adversarial Robustness via Promoting Ensemble Diversity](http://proceedings.mlr.press/v97/pang19a/pang19a.pdf)
**NOTE**: 
Though deep neural networks have achieved significant progress on various tasks, often enhanced by model ensemble, existing high-performance models can be vulnerable to adversarial attacks. **Many efforts have been devoted to enhancing the robustness of individual networks and then constructing a straightforward ensemble, e.g., by directly averaging the outputs, which ignores the interaction among networks.** This paper presents a new method that **explores the interaction among individual networks to improve robustness for ensemble models.** Technically, we define a new notion of **ensemble diversity in the adversarial setting** as the diversity among non-maximal predictions of individual members, and **present an adaptive diversity promoting (ADP) regularizer to encourage the diversity,** which leads to globally better robustness for the ensemble by making adversarial examples difficult to transfer among individual members. Our method is computationally efficient and compatible with the defense methods acting on individual networks. Empirical results on various datasets verify that our method can improve adversarial robustness while maintaining state-of-the-art accuracy on normal examples.
{:.message}


## :+1: [NeurIPS 2019-Metric Learning for Adversarial Robustness](https://arxiv.org/pdf/1909.00900.pdf)
**NOTE**: 
Deep networks are well-known to be fragile to adversarial attacks. Using several standard image datasets and established attack mechanisms, we conduct an empirical analysis of deep representations under attack, and find that the attack causes the internal representation to shift closer to the "false" class. Motivated by this observation, **we propose to regularize the representation space under attack with metric learning in order to produce more robust classifiers.** **By carefully sampling examples for metric learning,** our learned representation not only increases robustness, but also can detect previously unseen adversarial samples. Quantitative experiments show improvement of robustness accuracy by up to 4% and detection efficiency by up to 6% according to Area Under Curve (AUC) score over baselines.
{:.message}



## :+1:  [NeurIPS 2019-Reducing Noise in GAN Training with Variance Reduced Extragradient](https://arxiv.org/abs/1904.08598)
**NOTE**: We study the effect of the **stochastic gradient noise** on the training of generative adversarial networks (GANs) and show that it can prevent the convergence of standard game optimization methods, while the batch version converges. We address this issue with a novel **stochastic variance-reduced extragradient (SVRE)** optimization algorithm that improves upon the best convergence rates proposed in the literature. We observe empirically that SVRE performs similarly to a batch method on MNIST while being computationally cheaper, and that SVRE yields more stable GAN training on standard datasets.
{:.message}




## [NeurIPS 2019-Certified Adversarial Robustness with Additive Gaussian Noise](https://arxiv.org/pdf/1809.03113.pdf)
**NOTE**: The existence of adversarial data examples has drawn significant attention in the deep-learning community; such data are seemingly minimally perturbed relative to the original data, but lead to very different outputs from a deep-learning algorithm. Although a significant body of work on developing defense models has been developed, most such models are heuristic and are often vulnerable to adaptive attacks. Defensive methods that provide theoretical robustness guarantees have been studied intensively, yet most fail to obtain non-trivial robustness when a large-scale model and data are present. To address these limitations, we introduce a framework that is scalable and provides certified bounds on the norm of the input manipulation for constructing adversarial examples. We establish a connection between robustness against adversarial perturbation and additive random noise, and propose a training strategy that can significantly improve the certified bounds. Our evaluation on MNIST, CIFAR-10 and ImageNet suggests that our method is scalable to complicated models and large data sets, while providing competitive robustness to state-of-the-art provable defense methods.
{:.message}


