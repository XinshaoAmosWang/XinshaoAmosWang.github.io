---
layout: post
title: Paper Summary on Label Manipulation, Output Regularisation (Optimisation tricks)
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

:+1: means being highly related to my personal research interest. 
0. [Label Smoothing](#label-smoothing)
0. [Confidence Penalty](#)
0. [Label Correction](#)
0. [Example Weighting](#)

[Related Notes](../2020-02-14-Core-machine-learning-topics/#knowledge-distillation)
{:.message}


## Label Smoothing
* [Does label smoothing mitigate label noise?- Michal Lukasik, Srinadh Bhojanapalli, Aditya Krishna Menon and Sanjiv Kumar](https://arxiv.org/pdf/2003.02819.pdf)
    * **The definition of LS:** Label smoothing is commonly used in training deep learning models, wherein one-hot training labels are mixed with uniform label vectors.

    * While **label smoothing apparently amplifies this problem — being equivalent to injecting symmetric noise to the labels** — we show how it relates to a general family of loss-correction techniques from the label noise literature. Building on this connection, we show that label smoothing is competitive with loss-correction under label noise. 
        * **Do you agree with this?** 

    * Further, we show that when distilling models from noisy data, label smoothing of the teacher is beneficial; this is in contrast to recent findings for noise-free problems, and sheds further light on settings where label smoothing is beneficial.
    
    * Interestingly, there are two competing intuitions. On the one hand, smoothing might mitigate the problem, as it **prevents overconfidence on any one example**. On the other hand, smoothing might accentuate the problem, as it is **equivalent to injecting uniform noise into all labels** [ DisturbLabel Xie et al., 2016].

    * **At first glance, this connection indicates that smoothing has an opposite effect to one such loss-correction technique.** However, we **empirically show that smoothing is competitive with such techniques in denoising**, and that it improves performance of distillation.
        * we present a novel connection of label smoothing to loss correction techniques from the label noise literature;
        * We empirically demonstrate that label smoothing significantly improves performance under label noise, which we explain by relating smoothing to l2 regularisation. 
        * we show that when distilling from noisy labels, smoothing the teacher improves the student. While Müller et al. [2019] established that label smoothing can harm distillation, we show an opposite picture in noisy settings.

* [Does label smoothing mitigate label noise?- Label smoothing meets loss correction](https://arxiv.org/pdf/2003.02819.pdf)
    * 

* [DisturbLabel: Regularizing CNN on the Loss Layer-CVPR 2016-Lingxi Xie, Jingdong Wang, Zhen Wei, Meng Wang, Qi Tian](https://arxiv.org/pdf/1605.00055.pdf)
    * Randomly replaces a part of labels as incorrect values in each iteration.

    * In each training iteration, DisturbLabel randomly selects a small subset of samples (from those in the current mini-batch) and randomly sets their ground-truth labels to be incorrect, which results in a noisy loss function and, consequently, noisy gradient back-propagation.

    * DisturbLabel works on each mini-batch independently.

* [Rethinking the inception architecture for computer vision-CVPR 2016 Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z](https://arxiv.org/pdf/1512.00567.pdf)
    * Model Regularization via Label Smoothing
    * LS is firstly proposed in this paper. 

* [Distilling the knowledge in a neural network-NeurIPS 2015 Workshop-Hinton, G., Vinyals, O., Dean, J](https://arxiv.org/pdf/1503.02531.pdf)
    
    * **Soft targets definition**: An obvious way to transfer the generalization ability of the cumbersome model to a small model is
    to **use the class probabilities produced by the cumbersome model as “soft targets” for training the small model.**

    * **More information and less variance**: When the soft targets have high entropy, they provide **much more information** per training case than hard targets and **much less variance in the gradient between training cases**, so the small model can often be trained on much    less data than the original cumbersome model and using a much higher learning rate.

    * **Why?**: 
        * **Feasibility**: **Caruana and his collaborators [1]** have shown that it is possible to
        compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique.
        * For tasks like MNIST in which the cumbersome model almost always produces the correct answer with very high confidence, much of the information about the learned function resides in the ratios of very small probabilities in the soft targets. For example, one version of a 2 may be given a probability of 10−6 of being a 3 and 10−9 of being a 7 whereas for another version it may be the other way around. 
        * **Relative Probabilities?** This is valuable information that defines a rich similarity structure over the data (i. e. it says which 2’s look like 3’s and which look like 7’s) but it has very little influence on the cross-entropy cost function during the transfer stage because **the probabilities are so close to zero.** 

        * **Matching Logits?**: **Caruana and his collaborators** circumvent this problem by **using the logits (the inputs to the final softmax) rather than the probabilities produced by the softmax as the targets** for learning the small model and they minimize the squared difference between the logits produced by the cumbersome model and the logits produced by the small model.
    
    * **Distillation Definition**: 
        * Our more general solution, called “distillation”, is to **raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets**. We then use the same high temperature when training the small model to match these soft targets. We show later that matching the logits of the cumbersome model is actually a special case of distillation.
        * we call “distillation” to transfer the knowledge from the cumbersome model to a small model that is more suitable for deployment.
    
    * **Why Temperature?** => **Matching Logits is a special case of distillaiton?**
        * Using a higher value for T produces a softer probability distribution over classes.
        

    * **Knowledge Definition**: 
        * Relative probabilities: For cumbersome models that learn to discriminate between a large number of classes, the normal training objective is to maximize the average log probability of the correct answer, but a side-effect of the learning is that **the trained model assigns probabilities to all of the incorrect answers and even when these probabilities are very small**, **some of them are much larger than others**. **The relative probabilities of incorrect answers** tell us a lot about how the cumbersome model tends to generalize.
        
    * **Training Data**: The transfer set that is used to train the small model could consist entirely of unlabeled data [1] or we could use the original training set.  
        * We have found that using the original training set works well, especially if we add a small term to the objective function that encourages the small model to predict the true targets as well as matching the soft targets provided by the cumbersome model.

    * **Case Analysis**
        * In the simplest form of distillation: knowledge is transferred to the distilled model by training it on a transfer set and using a soft target distribution for each case in the transfer set that is produced by using the cumbersome model with a high temperature in its softmax. The same high temperature is used when training the distilled model, but after it has been trained it uses a temperature of 1.
        * **Two objectives: matching correct labels and soft targets generated by a cumbersome model**: When the correct labels are known for all or some of the transfer set, this method can be significantly improved by also training the distilled model to produce the correct labels. One way to do this is to use the correct labels to modify the soft targets, but we found that a better way is to simply use a weighted average of two different objective functions.
        * Matching logits is a special case of distillation? **Matching softer probabilities produced with high temperature versus matching logits!**
{:.message}

## Entropy Minimization (Minimum Entropy Principle)
* [Semi-supervised Learning by Entropy Minimization-NeurIPS 2015-Yves Grandvalet, Yoshua Bengio](https://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf)
    * We consider the **semi-supervised learning problem**, where a decision rule is to be learned from labeled and unlabeled data. 
    * A series of experiments illustrates that the proposed solution **benefits from unlabeled data**.
    * The method challenges **mixture models** when the data are sampled from the **distribution class spanned by the generative model**. The performances are definitely in favor of minimum entropy regularization when generative models are misspecified, and the weighting of unlabeled data provides robustness to the violation of the “cluster assumption”. 

    * Finally, we also illustrate that the method can also be far superior to manifold learning in high dimension spaces.
* 
{:.message}
