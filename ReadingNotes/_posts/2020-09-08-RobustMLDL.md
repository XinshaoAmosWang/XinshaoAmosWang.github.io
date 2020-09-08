---
layout: post
title: Robust DL/ML 
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---

In general, robust deep learning covers: missing labels (semisupervised learning); noisy labels (noise detection and correction); regularisation techniques; sample imbalance (long-tailed class distribution); adversarial learning; and so on. 

0. [ICML-20 papers](#icml-20-papers)
0. [The design of loss functions (i.e., optimisation objectives or output regularistion)](#the-design-of-loss-functions-ie-optimisation-objectives-or-output-regularistion)
{:.message}

### ICML-20 papers
* [Error-Bounded Correction of Noisy Labels, Songzhu Zheng, Pengxiang Wu, Aman Goswami, Mayank Goswami, Dimitris Metaxas, Chao Chen](https://proceedings.icml.cc/paper/2020/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)
    * To be robust against label noise, many successful methods rely on the noisy classifiers (i.e., models trained on the noisy training data) to determine whether a label is trustworthy. However, it remains unknown why this heuristic works well in practice.
    * In this paper, **we provide the first theoretical explanation** for these methods.
    * We prove that the prediction of a noisy classifier can indeed be a good indicator of whether the label of a training data is clean.
    * Based on the theoretical result, we propose a novel algorithm that corrects the labels based on the noisy classifier prediction. The corrected labels are **consistent with the true Bayesian optimal classifier with high probability.**
    * We prove that when the noisy classifier has **low confidence on the label of a datum, such label is likely corrupted.** In fact, we can quantify the threshold of confidence, below which the label is likely to be corrupted, and above which is it likely to be not. We also empirically show that the bound in our theorem is tight.
    * We provide a theorem quantifying how a noisy classifier’s prediction correlates to the purity of a datum’s label. This provides theoretical explanation for data-recalibrating methods for noisy labels.
    * Inspired by the theorem, we propose **a new label correction algorithm with guaranteed success rate.**
    * A Bayes optimal classifier is the minimizer of the risk over all possible hypotheses.
    * We also have a burn-in stage in which we train the networkusing the original noisy labels for $$m$$ epochs. During theburn-in stage, we use the original cross-entropy loss;
    * After the burn-in stage, we want to avoid overfitting of theneural network. To achieve this goal, we introduce aretroactive loss term. The  idea  is  to  enforce  the consistency between $$f$$ and the prediction of the model at a previous epoch. 
    * In all experiments, **we use early stopping on validation set to tune hyperparameters and report theperformance on test set.**

    * **[Simple and Effective ProSelfLC: Progressive Self Label Correction](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/)**

* [Learning with Bounded Instance- and Label-dependent Label Noise, Jiacheng Cheng, Tongliang Liu, Kotagiri Ramamohanarao, Dacheng Tao](https://proceedings.icml.cc/paper/2020/file/f2d887e01a80e813d9080038decbbabb-Paper.pdf)
    * **Binary classification => Not highly useful.** 
    * . Specifically, we introduce the concept of **distilled examples**, i.e. examples whose labels are identical with the labels assigned for them by the Bayes optimal classifier, and prove that **under certain conditions classifiers learnt on distilled examples will converge to the Bayes optimal classifier**. 
    *  Inspired by the idea of learning with distilled examples, we then propose a learning algorithm with **theoretical guarantees for its robustness to BILN**.


* [Normalized Loss Functions for Deep Learning with Noisy Labels Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano, Sarah Erfani, James Bailey](https://proceedings.icml.cc/book/2020/hash/77f959f119f4fb2321e9ce801e2f5163)
    * **This work is motivated by [DM and IMAE](https://xinshaoamoswang.github.io/blogs/2020-06-14-Robust-Deep-LearningviaDerivativeManipulationIMAE/)**

    * We provide new theoretical insights into robust loss func-tions demonstrating that a simple normalization can makeany loss function robust to noisy labels.
    
    * We identify that existing robust loss functions suffer from an underfitting problem.  To address this, we propose ageneric framework Active Passive Loss(APL) to build new loss functions with **theoretically guaranteed robustness and sufficient learning properties.**
    
    * **Robustness and Convergence?**
    

* [Searching to Exploit Memorization Effect in Learning with Noisy Labels QUANMING YAO, Hansi Yang, Bo Han, Gang Niu, James Kwok](https://proceedings.icml.cc/static/paper_files/icml/2020/3285-Paper.pdf)
    * Sample selection approaches: select $$R(t)$$ small-loss samples based on network’s predictions
    * Formulation as an AutoML Problem  (complex algorithm personally);
    * Bi-level optimisation    
    * **No sample selection is needed: [DM and IMAE](https://xinshaoamoswang.github.io/blogs/2020-06-14-Robust-Deep-LearningviaDerivativeManipulationIMAE/)**

* [Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates, Yang Liu, Hongyi Guo](https://proceedings.icml.cc/static/paper_files/icml/2020/4950-Paper.pdf)
    * Overall, this method is complex due to **peer samples**. 
    * **The motivation/highlight is not novel**: without Knowing Noise Rates.  Our main goal is to provide an al-ternative that does not require the specification of the noiserates, nor an additional estimation step for the noise.
    * Peer loss is invariant to label noise when optimizing with it. This effect helps us get rid of theestimation of noise rates.

    * i) is robust to asymmetriclabel noise with **formal theoretical guarantees**  and  ii)  requires  no  prior  knowledge  or  estimationof the noise rates (**no need for specifying noise rates**).

    * We also provide preliminary results on **how peer loss generalizes to multi-class clas-sification problems.**

    * Relevant work 1: [neurips-19: $$L_{DMI}$$: A Novel Information-theoretic Loss Functionfor Training Deep Nets Robust to Label Noise](https://papers.nips.cc/paper/8853-l_dmi-a-novel-information-theoretic-loss-function-for-training-deep-nets-robust-to-label-noise.pdf) To the best ofour knowledge, $$L_{DMI}$$ is the first loss function that is provably robust to instance-independent label noise, regardless of noise pattern, and it can be applied to any existing classification neural networks straightforwardly without any auxiliary information. In addition to theoretical justification, we also empirically show that using $$L_{DMI}$$ outperforms all other counterparts in the classification task on both image dataset and natural language dataset include Fashion-MNIST, CIFAR-10, Dogs vs. Cats, MR with a variety of synthesized noise patterns and noise amounts,as well as a real-world dataset Clothing1M.
    The core of $$L_{DMI}$$ is a generalized version of mutual information, termed Determinant based Mutual Information (DMI), which is not only information-monotone but also relatively invariant.

    * Relevant work 2: [Water from Two Rocks: Maximizing the Mutual Information](https://arxiv.org/pdf/1802.08887.pdf)

    * **No loss function is needed: [DM and IMAE](https://xinshaoamoswang.github.io/blogs/2020-06-14-Robust-Deep-LearningviaDerivativeManipulationIMAE/)**


* [Federated Learning with Only Positive Labels Felix Xinnan Yu, Ankit Singh Rawat, Aditya Menon, Sanjiv Kumar](https://proceedings.icml.cc/book/2020/hash/2e2079d63348233d91cad1fa9b1361e9) 
{:.message}



### The design of loss functions (i.e., optimisation objectives or output regularistion) 
* :+1: [Improved Training Speed, Accuracy, and Data Utilization Through Loss Function Optimization](https://arxiv.org/pdf/1905.11528.pdf) 
    * Speed, Accuracy, Data Efficiency, etc; 
    * BAIKAL loss;
    * Genetic Loss Function Optimization (GLO) builds loss functions hierarchically from a set of operators and leaf nodes;
    * A general framework for loss function metalearning, covering both novel loss function discovery and optimization, is developed and evaluated experimentally.
    * **No loss function is needed: [DM and IMAE](https://xinshaoamoswang.github.io/blogs/2020-06-14-Robust-Deep-LearningviaDerivativeManipulationIMAE/)**

* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)

* [On loss functions for deep neural networks in classification => with theory of robustness and convergence](https://arxiv.org/pdf/1702.05659.pdf)
    * We try to investigate how particular choices of loss functions affect deep models and their learning dynamics, as well as resulting classifiers robustness to various effects;
    * We present new insights into theoretical properties of a couple of these losses;
    * We provide experimental evaluation of resulting models’ properties, including the effect on speed of learning, final performance, input data and label noise robustness as well as convergence.
    * So why is using these two loss functions ($$L_1$$, $$L_2$$ losses) unpopular? Is there anything fundamentally wrong with this formulation from the mathematical perspective? While the following observation is not definitive, it shows an insight into what might be the issue causing slow convergence of such methods.
    *  **Lack of convexity** comes from the same argument since **second derivative wrt. to any weight in the final layer of the model changes sign (as it is equivalent to first derivative being non-monotonic)**. 
    
    **Proposition 2**. $$L_1$$, $$L_2$$ losses applied to probabilities estimates coming
    from sigmoid (or softmax) have **non-monotonic partial derivatives wrt. to the output of the final layer (and the loss is not convex nor concave wrt. to last layer weights)**. Furthermore, **they vanish in both infinities, which slows down learning of heavily misclassified examples**.

    * **No loss function is needed: [DM and IMAE](https://xinshaoamoswang.github.io/blogs/2020-06-14-Robust-Deep-LearningviaDerivativeManipulationIMAE/)**
{:.message}




