---
layout: post
title: Notes on Core ML Topics
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---



0. [Kullback-Leibler Divergence](#kullback-leibler-divergence)
0. [What is the main difference between GAN and autoencoder](#what-is-the-main-difference-between-gan-and-autoencoder)
0. [What's the difference between a Variational Autoencoder (VAE) and an Autoencoder?](#whats-the-difference-between-a-variational-autoencoder-vae-and-an-autoencoder)

0. [Knowledge Distillation](#knowledge-distillation)
0. [Confidence penalty & Label Smoothing && Ouput Regularisation](#confidence-penalty--label-smoothing--ouput-regularisation)

0. [Uncertainty](#uncertainty)
0. [Long-tailed Recognition](#long-tailed-recognition)
0. [Meta-learning](#meta-learning)
0. [Ensemble methods](#ensemble-methods)
{:.message}


### Knowledge Distillation
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)
    * **Knowledge definition**: A more
abstract view of the knowledge, that frees it from any particular instantiation, is that it is a learned mapping from input vectors to output vectors.

    *  An obvious way to **transfer the generalization ability of the cumbersome model to a small model** is
to use the class probabilities produced by the cumbersome model as **“soft targets” for training the
small model.** When the soft targets
have high entropy, they provide much more information per training case than hard targets and much
less variance in the gradient between training cases, so the small model can often be trained on much
less data than the original cumbersome model and using a much higher learning rate.

    * [A very simple way to improve the performance of almost any machine learning
algorithm is to train many different models on the same data and then to average
their predictions](#ensemble-methods) => cumbersome and may be too computationally expensive

    * **Compress/distill the knowledge in an ensemble into a single model** which is much easier to deploy (distilling the knowledge in an ensemble of models into a single model);
{:.message}

### Confidence penalty & Label Smoothing && Ouput Regularisation
* [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://openreview.net/pdf?id=HkCjNI5ex)
    * **Output regularisation**: Regularizing the output distribution of large, deep neural networks has largely been unexplored.  Output regularization has the property that it is invariant
to the parameterization of the underlying neural network.
    * **Knowledge definition**: To motivate output regularizers, we can view the knowledge of a model as the conditional distribution it produces over outputs given an input (Hinton et al., 2015) as opposed to the learned values
of its parameters.
    * **Distillation definition:** explicitly training a small network to assign the same probabilities to incorrect
classes as a large network or ensemble of networks that generalizes well.
    * **Two output regularizers**: 
        * A maximum entropy based confidence penalty;
        * Label smoothing (uniform and unigram). 
        * We connect a maximum entropy based confidence penalty to label smoothing through the direction of the KL divergence.
    * ANNEALING AND THRESHOLDING THE CONFIDENCE PENALTY
        *  Suggesting a confidence penalty
that is weak at the beginning of training and strong near convergence. 
        * Only penalize output distributions when they are below a certain entropy threshold
* Label/Objective smoothing: 
    * Smoothing the labels with a uniform distribution-[Rethinking the Inception Architecture](https://arxiv.org/pdf/1512.00567.pdf)
    
    * Smooth the labels with a teacher model [Distilling, Hinton et al., 2015](https://arxiv.org/pdf/1503.02531.pdf) 
    
    * Smooth the labels with the model’s own distribution-[TRAINING DEEP NEURAL NETWORKS
ON NOISY LABELS WITH BOOTSTRAPPING (Reed et al., 2014)](https://arxiv.org/pdf/1412.6596.pdf)
    
    * Adding label noise simply-[Disturblabel: Regularizing cnn on the loss layer--CVPR 2016](https://www.zpascal.net/cvpr2016/Xie_DisturbLabel_Regularizing_CNN_CVPR_2016_paper.pdf)
    
    * **Distillation and self-distillation both
regularize a network by incorporating information about the ratios between incorrect classes.**

* Label-Smoothing Regularization proposed in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)-A mechanism for encouraging the model to be less confident.
    * Over-fitting 
    * Reduces the ability of the model to **adapt: bounded gradient**

* Virtual adversarial training (VAT) [Distributional
smoothing by virtual adversarial examples](https://arxiv.org/pdf/1507.00677.pdf)
    * Another promising smoothing regularizer. However, it has multiple hyperparameters, significantly more computation in grid-searching
{:.message}

### Uncertainty
{:.message}

### Long-tailed Recognition
{:.message}

### Meta-learning
{:.message}

### Ensemble methods
* [Ensemble methods in machine learning](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=7F08BF5ADDA7E8BB9D5F40EA4241AD81?doi=10.1.1.228.2236&rep=rep1&type=pdf)
    * Ensemble methods are learning algorithms that construct **a set of classifiers** and then classify new data points by taking **a weighted vote of their predictions.** 
    * The original ensemble method is **Bayesian averaging** but more recent algorithms include error correcting output coding, Bagging and boosting. **This paper reviews these methods and explains why ensembles can often perform better than any single classifier.**
        * Bayesian Voting Enumerating the Hypotheses. 
        * Bagging: Bagging presents the learning algorithm with a training set that consists of a sample of $m$ training examples drawn randomly with replacement from the original training set of $m$ items. 
        * ...
{:.message}



### Kullback-Leibler Divergence
 * [How to approximate our data (choose a parameterized distribution => optimise its parameters): KL Divergence helps us to measure just how much information we lose when we choose an approximation compared with our observations.](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
    * The most important metric in information theory is called **Entropy**, typically denoted as $\mathbf{H}$. The definition of Entropy for a probability distribution is: $\mathbf{H}=-\sum_{i=1}^{n} p(\mathbf{x}_i) \log p(\mathbf{x}_i) $.
    * If we use $\log_2$ for our calculation we can interpret entropy as "the minimum number of bits it would take us to encode our information".
* [Intuitive Guide to Understanding KL Divergence](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-understanding-kl-divergence-2b382ca2b2a8)
    * What is a distributin?
    * What is an event?
    * Problem we’re trying to solve: choose a parameterized distribution => optimise its parameters): KL Divergence helps us to measure just how much information we lose when we choose an approximation compared with our observations.
{:.message}

### What is the main difference between GAN and autoencoder?
* [An autoencoder learns to represent some input information very efficiently, and subsequently how to reconstruct the input from it's compressed form.](https://datascience.stackexchange.com/a/55094)
    ~ :) ~[An autoencoder compresses its input down to a vector - with much fewer dimensions than its input data, and then transforms it back into a tensor with the same shape as its input over several neural net layers. They’re trained to reproduce their input, so it’s kind of like learning a compression algorithm for that specific dataset.](https://qr.ae/TzM5Mv)
* [A GAN uses an adversarial feedback loop to learn how to generate some information that "seems real" (i.e. looks the same/sounds the same/is otherwise indistinguishable from some real data)](https://datascience.stackexchange.com/a/55094) ~ :) ~ [Instead of being given a bit of data as input, it’s given a small vector of random numbers. The generator network tries to transform this little vector into a realistic sample from the training data. The discriminator network then takes this generated sample(and some real samples from the dataset) and learns to guess whether the samples are real or fake.](https://qr.ae/TzM5Mv)

* [Another difference: while they both fall under the umbrella of unsupervised learning, they are different approaches to the problem. A GAN is a generative model - it’s supposed to learn to generate realistic *new* samples of a dataset. Variational autoencoders are generative models, but normal “vanilla” autoencoders just reconstruct their inputs and can’t generate realistic new samples.](https://qr.ae/TzM5Mv)

* [Autoencoders learn a given distribution comparing its input to its output, this is good for learning hidden representations of data, but is pretty bad for generating new data. Mainly because we learn an averaged representation of the data thus the output becomes pretty blurry.
    Generative Adversarial Networks take an entirely different approach. They use another network (so-called Discriminator) to measure the distance between the generated and the real data.
    The main advantage of GANs over Autoencoders in generating data is that they can be conditioned by different inputs. For example, you can learn the mapping between two domains: satellite images to google maps [1] . Or you can teach the generator to reproduce several classes of data: generating the MNIST dataset[2] .
    ](https://qr.ae/TzMSyS)

    

    
* [ Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
* [Coding: GANs vs. Autoencoders: Comparison of Deep Generative Models](https://towardsdatascience.com/gans-vs-autoencoders-comparison-of-deep-generative-models-985cf15936ea)
{:.message}



### What's the difference between a Variational Autoencoder (VAE) and an Autoencoder?
* [Intuitively Understanding Variational Autoencoders – Towards Data Science by Irhum Shafkat.](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

* <span class='quora-content-embed' data-name='Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder/answer/Vishal-Sharma-154'>Read <a class='quora-content-link' data-width='560' data-height='260' href='https://www.quora.com/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder/answer/Vishal-Sharma-154' data-type='answer' data-id='66853410' data-key='a5099035f08fbac1ed45a4bb7a1c5d2c' load-full-answer='False' data-embed='trhonms'><a href='https://www.quora.com/Vishal-Sharma-154'>Vishal Sharma</a>&#039;s <a href='/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder?top_ans=66853410'>answer</a> to <a href='/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder' ref='canonical'><span class="rendered_qtext">What&#039;s the difference between a Variational Autoencoder (VAE) and an Autoencoder?</span></a></a> on <a href='https://www.quora.com'>Quora</a><script type="text/javascript" src="https://www.quora.com/widgets/content"></script></span>

* [ Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
* [Using Artificial Intelligence to Augment Human Intelligence](https://distill.pub/2017/aia/)
    
* [VAEs and GANs
Mihaela Rosca](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/VAE_GANS_by_Rosca.pdf)
* [Going Beyond GAN? New DeepMind VAE Model Generates High Fidelity Human Faces](https://syncedreview.com/2019/06/06/going-beyond-gan-new-deepmind-vae-model-generates-high-fidelity-human-faces/)
{:.message}
