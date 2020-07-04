---
layout: post
title: In deep metric learning, The improvements over time have been marginal?
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---


Recently, in paper [A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf), it is reported that the improvements over time have been marginal at best. Is it true? I present my personal viewpoints as follows:
* First of all, acedemic research progress is naturally slow, continuous and tortuous. Beyond, it is full of flaws on its progress. For example, 
    * In person re-identification, several years ago, some researchers vertically split one image into several parts for alignment, which is against the design of CNNs and non-meaningful. Because deep CNNs are designed to be invariant against translation, so that hand-crafted alignment is unnecessary. 

    * The Adam optimiser is found to be very sensitive to the setting of delta recently.

    * [Does Mean Absolute Error Treat Examples Equally?](https://arxiv.org/pdf/1903.12141.pdf)

    * [How to understand a loss function in a right way? Does a loss function have to be differentiable?](https://arxiv.org/pdf/1905.11233.pdf)

    * Is [DisturbLabel](https://openaccess.thecvf.com/content_cvpr_2016/papers/Xie_DisturbLabel_Regularizing_CNN_CVPR_2016_paper.pdf) a meaningful regulariser? If so, it makes me think that we should **deliberately generate label noise** at the training dataset! Is not it ridiculous? You will cultivate your own opinion after reading [ProSelfLC: Progressive Self Label Correction for Training Robust Deep Neural Networks](https://arxiv.org/pdf/2005.03788.pdf).

    * [Confidence penalty or rewarding, which one to go ahead?](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#storyline)

    * [To trust a learner (i.e. a deep model) or human annotations (i.e. textbooks for supervision), which one to choose?](https://arxiv.org/pdf/2005.03788.pdf)
    
    * ...
    
* There are some vital breakthroughs over the time although they seem to be trivial now. 
    * [Multibatch Method](https://www.cse.huji.ac.il/~shashua/papers/multibatch-nips16.pdf), after which people rarely use rigid input formats. 
        * Before this milestone, we heard a lot about siamese networks, triplet networks, etc. 
        * After Multibatch Method, we construct doublets, triplets, or high-order tuples directly in the embedding space.

    * The importance of sample mining/weighting becomes clearer for our community. Of course, there exist many variants of sample mining/weighting for different scenarios.

    * Our community become much more open-minded: all methods which learn discriminative representations can be categorised into deep metric learning, e.g., softmax + categorical cross entropy.

    * Deep metric learning tends to follow a similar training setting with few-shot training, i.e., in one training batch, C classes and K examples per class are randomly sampled. 
    Naturally, we can make C and K random to increase the training stochasticity.  

    * ...





