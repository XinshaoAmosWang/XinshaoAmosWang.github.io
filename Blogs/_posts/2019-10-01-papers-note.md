---
layout: post
title: Paper Notes
description: >
  
image: /assets/img/blog/steve-harvey.jpg
---

Paper Notes on ...
{:.message}

## [NeurIPS19-Generalized Sliced Wasserstein Distances](https://arxiv.org/abs/1902.00434)
**NOTE**: Wasserstein Distances
{:.message}



## [ICML19-A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks](http://proceedings.mlr.press/v97/simsekli19a/simsekli19a.pdf)
**NOTE**: Stochastic Gradient Noise
{:.message}


## [NIPS19-First Exit Time Analysis of Stochastic Gradient Descent Under Heavy-Tailed Gradient Noise](https://arxiv.org/pdf/1906.09069.pdf)
**NOTE**: Stochastic Gradient Noise
{:.message}


## [NIPS19-Noise-tolerant fair classification](https://arxiv.org/abs/1901.10837)
**NOTE**: Existing work on the problem operates under the assumption that the sensitive feature available in one's training sample is perfectly reliable. This assumption may be violated in many real-world cases: for example, respondents to a survey may choose to conceal or obfuscate their group identity out of fear of potential discrimination. This poses the question of whether one can still learn fair classifiers given noisy sensitive features.
{:.message}



## [NIPS19-Reducing Noise in GAN Training with Variance Reduced Extragradient](https://arxiv.org/abs/1904.08598)
**NOTE**: We study the effect of the stochastic gradient noise on the training of generative adversarial networks (GANs) and show that it can prevent the convergence of standard game optimization methods, while the batch version converges. We address this issue with a novel stochastic variance-reduced extragradient (SVRE) optimization algorithm that improves upon the best convergence rates proposed in the literature. We observe empirically that SVRE performs similarly to a batch method on MNIST while being computationally cheaper, and that SVRE yields more stable GAN training on standard datasets.
{:.message}




## [NIPS19-Combinatorial Inference against Label Noise]()
**NOTE**: Paper is not available yet.
{:.message}




## [NIPS19-Extending Stein's unbiased risk estimator to train deep denoisers with correlated pairs of noisy images](https://arxiv.org/pdf/1902.02452.pdf)
**NOTE**: Recently, Stein's unbiased risk estimator (SURE) has been applied to unsupervised training of deep neural network Gaussian denoisers that outperformed classical non-deep learning based denoisers and yielded comparable performance to those trained with ground truth. While SURE requires only one noise realization per image for training, it does not take advantage of having multiple noise realizations per image when they are available (e.g., two uncorrelated noise realizations per image for Noise2Noise). Here, we propose an extended SURE (eSURE) to train deep denoisers with correlated pairs of noise realizations per image and applied it to the case with two uncorrelated realizations per image to achieve better performance than SURE based method and comparable results to Noise2Noise. Then, we further investigated the case with imperfect ground truth (i.e., mild noise in ground truth) that may be obtained considering painstaking, time-consuming, and even expensive processes of collecting ground truth images with multiple noisy images. For the case of generating noisy training data by adding synthetic noise to imperfect ground truth to yield correlated pairs of images, our proposed eSURE based training method outperformed conventional SURE based method as well as Noise2Noise.
{:.message}


## [NIPS19-Variational Denoising Network: Toward Blind Noise Modeling and Removal](https://arxiv.org/pdf/1908.11314.pdf)
**NOTE**: Blind image denoising is an important yet very challenging problem in computer vision due to the complicated acquisition process of real images. In this work we propose a new variational inference method, which integrates both noise estimation and image denoising into a unique Bayesian framework, for blind image denoising. Specifically, an approximate posterior, parameterized by deep neural networks, is presented by taking the intrinsic clean image and noise variances as latent variables conditioned on the input noisy image. This posterior provides explicit parametric forms for all its involved hyper-parameters, and thus can be easily implemented for blind image denoising with automatic noise estimation for the test noisy image. On one hand, as other data-driven deep learning methods, our method, namely variational denoising network (VDN), can perform denoising efficiently due to its explicit form of posterior expression. On the other hand, VDN inherits the advantages of traditional model-driven approaches, especially the good generalization capability of generative models. VDN has good interpretability and can be flexibly utilized to estimate and remove complicated non-i.i.d. noise collected in real scenarios. Comprehensive experiments are performed to substantiate the superiority of our method in blind image denoising.
{:.message}



## [NIPS19-Neural networks grown and self-organized by noise](https://arxiv.org/abs/1906.01039)
**NOTE**: Living neural networks emerge through a process of growth and self-organization that begins with a single cell and results in a brain, an organized and functional computational device. Artificial neural networks, however, rely on human-designed, hand-programmed architectures for their remarkable performance. Can we develop artificial computational devices that can grow and self-organize without human intervention? In this paper, we propose a biologically inspired developmental algorithm that can 'grow' a functional, layered neural network from a single initial cell. The algorithm organizes inter-layer connections to construct a convolutional pooling layer, a key constituent of convolutional neural networks (CNN's). Our approach is inspired by the mechanisms employed by the early visual system to wire the retina to the lateral geniculate nucleus (LGN), days before animals open their eyes. The key ingredients for robust self-organization are an emergent spontaneous spatiotemporal activity wave in the first layer and a local learning rule in the second layer that 'learns' the underlying activity pattern in the first layer. The algorithm is adaptable to a wide-range of input-layer geometries, robust to malfunctioning units in the first layer, and so can be used to successfully grow and self-organize pooling architectures of different pool-sizes and shapes. The algorithm provides a primitive procedure for constructing layered neural networks through growth and self-organization. Broadly, our work shows that biologically inspired developmental algorithms can be applied to autonomously grow functional 'brains' in-silico.
{:.message}



## [NIPS19-L_DMI: A Novel Information-theoretic Loss Function for Training Deep Nets Robust to Label Noise](https://arxiv.org/pdf/1909.03388.pdf)
**NOTE**: Accurately annotating large scale dataset is notoriously expensive both in time and in money. Although acquiring low-quality-annotated dataset can be much cheaper, it often badly damages the performance of trained models when using such dataset without particular treatment. Various of methods have been proposed for learning with noisy labels. However, they only handle limited kinds of noise patterns, require auxiliary information (e.g,, the noise transition matrix), or lack theoretical justification. In this paper, we propose a novel information-theoretic loss function, LDMI, for training deep neural networks robust to label noise. The core of LDMI is a generalized version of mutual information, termed Determinant based Mutual Information (DMI), which is not only information-monotone but also relatively invariant. \emph{To the best of our knowledge, LDMI is the first loss function that is provably not sensitive to noise patterns and noise amounts, and it can be applied to any existing classification neural networks straightforwardly without any auxiliary information}. In addition to theoretical justification, we also empirically show that using LDMI outperforms all other counterparts in the classification task on Fashion-MNIST, CIFAR-10, Dogs vs. Cats datasets with a variety of synthesized noise patterns and noise amounts as well as a real-world dataset Clothing1M. Codes are available at  https://github.com/Newbeeer/L_DMI
{:.message}



## [NIPS19-Are Anchor Points Really Indispensable in Label-Noise Learning?](https://arxiv.org/pdf/1906.00189.pdf)
**NOTE**: In label-noise learning, \textit{noise transition matrix}, denoting the probabilities that clean labels flip into noisy labels, plays a central role in building \textit{statistically consistent classifiers}. Existing theories have shown that the transition matrix can be learned by exploiting \textit{anchor points} (i.e., data points that belong to a specific class almost surely). However, when there are no anchor points, the transition matrix will be poorly learned, and those current consistent classifiers will significantly degenerate. In this paper, without employing anchor points, we propose a \textit{transition-revision} (T-Revision) method to effectively learn transition matrices, leading to better classifiers. Specifically, to learn a transition matrix, we first initialize it by exploiting data points that are similar to anchor points, having high \textit{noisy class posterior probabilities}. Then, we modify the initialized matrix by adding a \textit{slack variable}, which can be learned and validated together with the classifier by using noisy data. Empirical results on benchmark-simulated and real-world label-noise datasets demonstrate that without using exact anchor points, the proposed method is superior to the state-of-the-art label-noise learning methods.
{:.message}




## [NIPS19-Certified Adversarial Robustness with Additive Gaussian Noise](https://arxiv.org/pdf/1809.03113.pdf)
**NOTE**: The existence of adversarial data examples has drawn significant attention in the deep-learning community; such data are seemingly minimally perturbed relative to the original data, but lead to very different outputs from a deep-learning algorithm. Although a significant body of work on developing defense models has been developed, most such models are heuristic and are often vulnerable to adaptive attacks. Defensive methods that provide theoretical robustness guarantees have been studied intensively, yet most fail to obtain non-trivial robustness when a large-scale model and data are present. To address these limitations, we introduce a framework that is scalable and provides certified bounds on the norm of the input manipulation for constructing adversarial examples. We establish a connection between robustness against adversarial perturbation and additive random noise, and propose a training strategy that can significantly improve the certified bounds. Our evaluation on MNIST, CIFAR-10 and ImageNet suggests that our method is scalable to complicated models and large data sets, while providing competitive robustness to state-of-the-art provable defense methods.
{:.message}


