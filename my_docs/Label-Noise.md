---
layout: page
title: Label Noise
description: >
  
comment: true
---


:+1: means being highly related to my personal research interest. 
{:.message}



## :+1: [NeurIPS 2019-L_DMI: A Novel Information-theoretic Loss Function for Training Deep Nets Robust to Label Noise](https://arxiv.org/pdf/1909.03388.pdf)
**NOTE**: Accurately annotating large scale dataset is notoriously expensive both in time and in money. Although acquiring low-quality-annotated dataset can be much cheaper, it often badly damages the performance of trained models when using such dataset without particular treatment. Various of methods have been proposed for learning with noisy labels. However, **they only handle limited kinds of noise patterns, require auxiliary information (e.g,, the noise transition matrix), or lack theoretical justification.** In this paper, we propose a novel **information-theoretic loss function,** LDMI, for training deep neural networks robust to label noise. The core of LDMI is a generalized version of mutual information, termed Determinant based Mutual Information (DMI), which is not only information-monotone but also relatively invariant. \emph{To the best of our knowledge, LDMI is the first loss function that is **provably not sensitive to noise patterns and noise amounts**, and it can be applied to any existing classification neural networks straightforwardly **without any auxiliary information**}. In addition to theoretical justification, we also empirically show that using LDMI outperforms all other counterparts in the classification task on **Fashion-MNIST, CIFAR-10, Dogs vs. Cats datasets with a variety of synthesized noise patterns and noise amounts as well as a real-world dataset Clothing1M**. <br /> 
Codes are available at [https://github.com/Newbeeer/L_DMI](https://github.com/Newbeeer/L_DMI)
{:.message}



## [NeurIPS 2019-Are Anchor Points Really Indispensable in Label-Noise Learning?](https://arxiv.org/pdf/1906.00189.pdf)
**NOTE**: In label-noise learning, \textit{noise transition matrix}, denoting the probabilities that clean labels flip into noisy labels, plays a central role in building \textit{statistically consistent classifiers}. Existing theories have shown that the transition matrix can be learned by exploiting \textit{anchor points} (i.e., data points that belong to a specific class almost surely). However, when there are no anchor points, the transition matrix will be poorly learned, and those current consistent classifiers will significantly degenerate. In this paper, without employing anchor points, we propose a \textit{transition-revision} (T-Revision) method to effectively learn transition matrices, leading to better classifiers. Specifically, to learn a transition matrix, we first initialize it by exploiting data points that are similar to anchor points, having high \textit{noisy class posterior probabilities}. Then, we modify the initialized matrix by adding a \textit{slack variable}, which can be learned and validated together with the classifier by using noisy data. Empirical results on benchmark-simulated and real-world label-noise datasets demonstrate that without using exact anchor points, the proposed method is superior to the state-of-the-art label-noise learning methods.
{:.message}






## :+1: [NeurIPS 2019-Combinatorial Inference against Label Noise]()
**NOTE**: Paper is not available yet.
{:.message}
