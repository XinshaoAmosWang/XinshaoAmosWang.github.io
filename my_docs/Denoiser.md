---
layout: page
title: Denoiser, Noise Removal
description: >
  
comment: true
---


:+1: means being highly related to my personal research interest. 
{:.message}



## [NeurIPS 2019-Extending Stein's unbiased risk estimator to train deep denoisers with correlated pairs of noisy images](https://arxiv.org/pdf/1902.02452.pdf)
**NOTE**: Recently, Stein's unbiased risk estimator (SURE) has been applied to **unsupervised training of deep neural network Gaussian denoisers that outperformed classical non-deep learning based denoisers and yielded comparable performance to those trained with ground truth.** While SURE requires only one noise realization per image for training, it does not take advantage of having multiple noise realizations per image when they are available (e.g., two uncorrelated noise realizations per image for Noise2Noise). Here, we propose an extended SURE (eSURE) to train **deep denoisers with correlated pairs of noise realizations per image** and applied it to the case with two uncorrelated realizations per image to achieve better performance than SURE based method and comparable results to Noise2Noise. Then, we further investigated **the case with imperfect ground truth (i.e., mild noise in ground truth) that may be obtained considering painstaking, time-consuming, and even expensive processes of collecting ground truth images with multiple noisy images.** For the case of generating noisy training data by adding synthetic noise to imperfect ground truth to yield correlated pairs of images, our proposed eSURE based training method outperformed conventional SURE based method as well as Noise2Noise.
{:.message}


## [NeurIPS 2019-Variational Denoising Network: Toward Blind Noise Modeling and Removal](https://arxiv.org/pdf/1908.11314.pdf)
**NOTE**: Blind image denoising is an important yet very challenging problem in computer vision due to the complicated acquisition process of real images. In this work we propose a new variational inference method, which integrates both noise estimation and image denoising into a unique Bayesian framework, for blind image denoising. Specifically, an approximate posterior, parameterized by deep neural networks, is presented by taking the intrinsic clean image and noise variances as latent variables conditioned on the input noisy image. This posterior provides explicit parametric forms for all its involved hyper-parameters, and thus can be easily implemented for blind image denoising with automatic noise estimation for the test noisy image. On one hand, as other data-driven deep learning methods, our method, namely variational denoising network (VDN), can perform denoising efficiently due to its explicit form of posterior expression. On the other hand, VDN inherits the advantages of traditional model-driven approaches, especially the good generalization capability of generative models. VDN has good interpretability and can be flexibly utilized to estimate and remove complicated non-i.i.d. noise collected in real scenarios. Comprehensive experiments are performed to substantiate the superiority of our method in blind image denoising. <br />
[Pytorch Code](https://github.com/zsyOAOA/VDNet)
{:.message}

