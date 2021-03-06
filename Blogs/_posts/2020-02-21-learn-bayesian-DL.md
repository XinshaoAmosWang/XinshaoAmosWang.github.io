---
layout: post
title: Learning Bayesian Deep Learning, Uncertainty & Variational Techniques
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---



0. [Blogs](#blogs)
0. [Papers on Theories](#papers-on-theories)
0. [Papers on Applications](#papers-on-applications)
{:.message}


### What am I working on now? Discussions are Welcome!
* [Interpreting $$ p(y\|x) $$ and modelling example weighting](../2020-02-18-code-releasing)

* Going to stop treating $$ p(y\|x) $$ as a classfication confidence metric, since it is determinstic. $$ p(y\|x) $$  is not for deciding whether certain or uncertain.


* $$ p(y\|x) $$ is good as a metric of whether x matches y, though not a good metric indicating whether x is blur or not.  

* Utilities of Uncertainties
{:.message}


### Blogs
* [Everything that Works Works Because it's Bayesian: Why Deep Nets Generalize?](https://www.inference.vc/everything-that-works-works-because-its-bayesian-2/)
* [Yann LeCun's Comments](https://www.facebook.com/yann.lecun/posts/10154058859142143)
* [YARIN GAL's PhD Thesis](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html?fbclid=IwAR1lNokscvPVsGFICXDQBhVa2bweIq-mkft6EfUkj9CR8tAIYJ7mNy3Qag8)
{:.message}


### Papers on Theories
* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning-ICML 2016-YARIN GAL](https://arxiv.org/pdf/1506.02142.pdf)
* [YARIN GAL's PhD Thesis](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)
* [A Bayesian Perspective on Generalization and Stochastic Gradient Descent-ICLR 2018 Google Brain-Samuel L. Smith and Quoc V. Le](https://openreview.net/forum?id=BJij4yg0Z)
* [Bayesian Deep Learning and a Probabilistic Perspective of Generalization--arXiv 2020 New York University-Andrew Gordon Wilson Pavel Izmailov](https://arxiv.org/pdf/2002.08791.pdf)
* [Sharp Minima Can Generalize For Deep Nets-ICML 2017](https://arxiv.org/pdf/1703.04933.pdf)
* [Theory of Deep Learning III: Generalization Properties of SGD](https://cbmm.mit.edu/sites/default/files/publications/CBMM-Memo-067.pdf)
* [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima-ICLR 2017](https://openreview.net/forum?id=H1oyRlYgg)
* [The Marginal Value of Adaptive Gradient Methods in Machine Learning-NIPS 2017](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning)
* [Stochastic Gradient Descent as Approximate Bayesian Inference-JMLR 2017](http://www.jmlr.org/papers/volume18/17-214/17-214.pdf)
* [A Variational Analysis of Stochastic Gradient Algorithms-ICML 2016](http://proceedings.mlr.press/v48/mandt16.pdf)

* [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/pdf/1503.02406.pdf)
* [On the Difference Between the Information Bottleneck and the Deep Information Bottleneck](https://arxiv.org/pdf/1912.13480.pdf)
* [Mutual Information Neural Estimation](http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf)
{:.message}

### Papers on Applications

* [Robust Person Re-Identification by Modelling Feature Uncertainty](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Robust_Person_Re-Identification_by_Modelling_Feature_Uncertainty_ICCV_2019_paper.pdf)
* [Probabilistic Face Embeddings](https://arxiv.org/pdf/1904.09658.pdf)
* [Rethinking Person Re-Identification with Confidence](https://arxiv.org/pdf/1906.04692v1.pdf)
* [Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/pdf/1802.04865.pdf)
* [Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples](https://openreview.net/forum?id=ryiAv2xAZ)
{:.message}