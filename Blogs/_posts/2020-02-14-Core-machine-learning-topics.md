---
layout: post
title: Notes on Core ML Techniques
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
0. [Long-tailed Recognition-Sample Imbalance](#long-tailed-recognition)
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
* [The variational auto-encoder](https://ermongroup.github.io/cs228-notes/extras/vae/)

* [Information Constraints on Auto-Encoding Variational Bayes-NeurIPS 2018](https://papers.nips.cc/paper/7850-information-constraints-on-auto-encoding-variational-bayes.pdf)

* [Decision-Making with Auto-Encoding Variational Bayes](https://arxiv.org/pdf/2002.07217.pdf)

* [Auto-Encoding Variational Bayes-ICLR 2014](https://arxiv.org/pdf/1312.6114.pdf)

* [Deep Variational Information Bottleneck-ICLR2017](https://openreview.net/forum?id=HyxQzBceg)

* [Deep Variational Canonical Correlation Analysis](https://arxiv.org/pdf/1610.03454.pdf)

* [The information bottleneck (IB) principle--The information
bottleneck method](https://arxiv.org/pdf/physics/0004057.pdf)

* [CS 228 - Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/)

* [Rethinking Person Re-Identification with Confidence](https://arxiv.org/pdf/1906.04692.pdf)
{:.message}



### Long-tailed Recognition
* [DECOUPLING REPRESENTATION AND CLASSIFIER
FOR LONG-TAILED RECOGNITION-ICLR2020](https://openreview.net/references/pdf?id=nHObduxXz)
    * Representation Learning: We first train models to learn representations with different sampling strategies, including the standard instance-based sampling, class-balanced sampling and a mixture of them. 
    * Classification: We study three different basic approaches to obtain a classifier with balanced decision boundaries, on top of the learned representations.
{:.message}




### Meta-learning
* [Confusion on the definition of Meta-learning](https://www.reddit.com/r/MachineLearning/comments/f6e25t/r_confusion_on_the_definition_of_metalearning/)

* [Few-shot Learning is an instantiation of Meta-learning](../../my_docs/few-shot)
* [MetaCleaner: Learning to Hallucinate Clean Representations for Noisy-Labeled Visual Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_MetaCleaner_Learning_to_Hallucinate_Clean_Representations_for_Noisy-Labeled_Visual_Recognition_CVPR_2019_paper.pdf)
    * Noisy Weighting: estimate the confidence scores of all the images in the noisy subset;  **MetaCleaner compares these representations in the feature space => discover relations between images => generate the confidence score of each image in the subset.** 
    * Clean Hallucinating: to hallucinate a `clean‘ representation of a class from the noisy subset, by summarizing the noisy images with their confidence scores;
    * **MetaCleaner as a new layer before classifier: batch size $K \times N => K$, $K$ categories, $N$ images per class in the batch**.  
    * Different from prototypical network, our MetaCleaner mainly develops a robust classifier to reduce confusion of noisy labels. Hence, it adaptively uses the weighted prototype as a ‘clean’ representation to generalize softmax classifier, instead of using the mean prototype to construct a metric classifier for low-shot learning.
    * **Why is this called meta-learning?** 
* [Learning to Learn From Noisy Labeled Data-CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_to_Learn_From_Noisy_Labeled_Data_CVPR_2019_paper.pdf)
    * [My Understanding: https://github.com/LiJunnan1992/MLNT/issues/1](https://github.com/LiJunnan1992/MLNT/issues/1)
        * Iteratively Improve the Teacher/Oracle == Soft Target
        * **Meta-obejctive-training/testing:**   The meta-training sees synthetic noisy training examples. After training on them, the meta-testing evaluates its consistency with oracle and aims to maximise the consistency, i.e., making it unaffected after seeing synthetic noise.
    * [Reddit Analysis](https://www.reddit.com/r/MachineLearning/comments/bws5iv/r_cvpr_2019_noisetolerant_training_work_learning/): Extremely complex in practice. However, the ideas are interesting and novel. 
* [Learning to Reweight Examples for Robust Deep Learning-ICML 2018](https://arxiv.org/pdf/1803.09050.pdf)- Simultaneously minimize the loss on **a clean unbiased validation set**.
    * **Meta-objective**: a novel meta-learning algorithm that learns to assign weights to training examples based on their gradient directions. 
    * **Solution**:  Suppose that **a pair of training and validation examples are very similar**, and they also provide
**similar gradient directions**, then this training example is
helpful and should be up-weighted, and conversely, if they
provide **opposite gradient directions**, this training example
is harmful and should be downweighed.
* [Meta-Weight-Net: Learning an Explicit Mapping
For Sample Weighting-NeurIPS 2019](https://papers.nips.cc/paper/8467-meta-weight-net-learning-an-explicit-mapping-for-sample-weighting.pdf)
    * The major difference with [Learning to Reweight Examples](https://arxiv.org/pdf/1803.09050.pdf) is that the weights are implicitly learned there, without an explicit weighting function. 
        * **I am skeptical and not convinced here!**
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
