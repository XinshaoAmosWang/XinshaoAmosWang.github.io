---
layout: post
title: Paper Summary on Noise, Anomalies, Adversaries, Robust Learning, Generalization
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

:+1: means being highly related to my personal research interest. 
{:.message}


## ICCV 2019 on label noise, ...
* [Deep Self-Learning From Noisy Labels](http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Deep_Self-Learning_From_Noisy_Labels_ICCV_2019_paper.pdf): The proposed SMP trains in an iterative manner which
contains two phases: the first phase is to train a network
with **the original noisy label and corrected label** generated
in the second phase.
* [Co-Mining: Deep Face Recognition With Noisy Labels](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Co-Mining_Deep_Face_Recognition_With_Noisy_Labels_ICCV_2019_paper.pdf): We propose a novel **co-mining** framework, which employs two peer networks to **detect the noisy faces,
exchanges the high-confidence clean faces and reweights the clean faces** in a mini-batch fashion.
* [NLNL: Negative Learning for Noisy Labels](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kim_NLNL_Negative_Learning_for_Noisy_Labels_ICCV_2019_paper.pdf): Input image belongs to this label--Positive Learning; Negative Learning (NL)--CNNs are trained using a complementary label as in “input image does not belong to this complementary label.”
* [Symmetric Cross Entropy for Robust Learning With Noisy Labels](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf): Already compared in our method. 
* [O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf): It only requires adjusting the hyper-parameters of the deep network to make its status transfer from overfitting to underfitting (O2U) cyclically. The losses of each sample are recorded during iterations. The higher the normalized average loss of a sample, the higher the probability of being noisy labels.
{:.message}


## :+1: [NeurIPS 2019-Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://papers.nips.cc/paper/8467-meta-weight-net-learning-an-explicit-mapping-for-sample-weighting)
* Targted problems: (1) Corrupted Labels (2) Class imbalance 
* Methodology: Guided by **a small amount of unbiased meta-data**,  to learn an explicit weighting layer which takes training losses as input and outputs examples' weights. 
* Code: [https://github.com/xjtushujun/meta-weight-net](https://github.com/xjtushujun/meta-weight-net)
* **Introduciton: Why are the targeted problems important?** In practice, however, such biased training data are commonly encountered. For instance, practically
collected training samples always contain corrupted labels [10, 11, 12, 13, 14, 15, 16, 17]. A typical
example is a dataset roughly collected from a crowdsourcing system [18] or search engines [19, 20],
which would possibly yield a large amount of noisy labels. Another popular type of biased training
data is those with class imbalance. Real-world datasets are usually depicted as skewed distributions,
with a long-tailed configuration. A few classes account for most of the data, while most classes are
under-represented. Effective learning with these biased training data, which is regarded to be biased
from evaluation/test ones, is thus an important while challenging issue in machine learning [1, 21].
* There exist **two entirely contradictive
ideas for constructing such a loss-weight mapping:** 
    * **Emphasise on harder ones:** Enforce the learning to more emphasize samples with larger loss values
since they are more like to be uncertain hard samples located on the classification boundary. Typical
methods of this category include AdaBoost [22, 23], hard negative mining [24] and focal loss [25]. **This sample weighting manner is known to be necessary for class imbalance problems, since it can
prioritize the minority class with relatively higher training losses.**
    * **Emphasise on easier ones:** The rationality
lies on that these samples are more likely to be high-confident ones with clean labels. Typical methods
include self-paced learning(SPL) [26], iterative reweighting [27, 17] and multiple variants [28, 29, 30].
This weighting strategy has been especially used in noisy label cases, since it inclines to suppress the
effects of samples with extremely large loss values, possibly with corrupted incorrect labels.
    * **Deficiencies:** 
        * How about the case that the training set is both imbalanced and
noisy. 
        * They inevitably involve hyper-parameters, to be manually preset or tuned by cross-validation. 
* Experiments of this work: 
    * Class Imbalance Experiments 
        * ResNet-32 on  long-tailed CIFAR-10 and CIFAR-100. 
    * Corrupted Label Experiments on CIFAR-10 and CIFAR-100 
        * WRN-28-10 with varying noise rates under uniform noise.
        * ResNet-32 with varying noise rates under flip noise - non-uniform noise. 
    * Real-world data-Clothing 1M with ResNet-50
        *  We use the 7k clean data as the meta dataset.

* Problems of this work: 
    * For **the case where the training set is both imbalanced and noisy**, the authors mentioned in the introduction section that conventional methods cannot address this case. 
    However, there is no experiment to demontrate that this method works. 
    * Conventional methods inevitably involve hyper-parameters to tune by cross-validation. 
    However, for the proposed method, **unbiased meta-data is required, which is a more expensive hyper-factor** in practice. 
    Tuning hyper-parameters is cheaper than collecting unbiased meta-data for training the weighting function. 
{:.message}


## :+1: [ICML 2019-Better generalization with less data using robust gradient descent](http://proceedings.mlr.press/v97/holland19a/holland19a.pdf)

## [GAN, Adversary Examples, Adversary Machine Learning](../../my_docs/adversary.md)


## [Label Noise](../../my_docs/Label-Noise.md)
* NeurIPS 2019-L_DMI: A Novel Information-theoretic Loss Function for Training Deep Nets Robust to Label Noise
* NeurIPS 2019-Are Anchor Points Really Indispensable in Label-Noise Learning?
* NeurIPS 2019-Combinatorial Inference against Label Noise
{:.message}




## :+1:  [NeurIPS 2019-Noise-tolerant fair classification](https://arxiv.org/abs/1901.10837)
**NOTE**: Existing work on the problem operates **under the assumption that the sensitive feature available in one's training sample is perfectly reliable.** This assumption may be violated in many real-world cases: for example, respondents to a survey may choose to conceal or obfuscate their group identity out of fear of potential discrimination. This poses the question of whether one can still learn fair classifiers given noisy sensitive features.
{:.message}



## [NeurIPS 2019-Neural networks grown and self-organized by noise](https://arxiv.org/abs/1906.01039)
**NOTE**: **Living neural networks** emerge through a process of growth and self-organization that begins with a single cell and results in a brain, an organized and functional computational device. Artificial neural networks, however, rely on human-designed, hand-programmed architectures for their remarkable performance. **Can we develop artificial computational devices that can grow and self-organize without human intervention?** In this paper, we propose a biologically inspired developmental algorithm that can **'grow' a functional, layered neural network from a single initial cell.** The algorithm organizes inter-layer connections to construct a convolutional pooling layer, a key constituent of convolutional neural networks (CNN's). Our approach is inspired by the mechanisms employed by the early visual system to wire the retina to the lateral geniculate nucleus (LGN), days before animals open their eyes. The key ingredients for robust self-organization are an emergent spontaneous spatiotemporal activity wave in the first layer and a local learning rule in the second layer that 'learns' the underlying activity pattern in the first layer. The algorithm is adaptable to a wide-range of input-layer geometries, robust to malfunctioning units in the first layer, and so can be used to **successfully grow and self-organize pooling architectures of different pool-sizes and shapes.** The algorithm provides a primitive procedure for constructing layered neural networks through growth and self-organization. Broadly, our work shows that biologically inspired developmental algorithms can be applied to autonomously grow functional 'brains' in-silico.
{:.message}


## [Stochastic-Gradient-Noise](../../my_docs/Stochastic-Gradient-Noise.md)
* ICML 2019-A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks
* NeurIPS 2019-First Exit Time Analysis of Stochastic Gradient Descent Under Heavy-Tailed Gradient Noise
{:.message}

## [Denoiser, Noise Removal](../../my_docs/Denoiser.md)
* NeurIPS 2019-Extending Stein’s unbiased risk estimator to train deep denoisers with correlated pairs of noisy images
* NeurIPS 2019-Variational Denoising Network: Toward Blind Noise Modeling and Removal
{:.message}