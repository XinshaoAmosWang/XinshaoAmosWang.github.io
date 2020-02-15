---
layout: post
title: Notes on Core ML Topics
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---




* What is the main difference between GAN and autoencoder?
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



* What's the difference between a Variational Autoencoder (VAE) and an Autoencoder?
    * [Intuitively Understanding Variational Autoencoders – Towards Data Science by Irhum Shafkat.](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

    * <span class='quora-content-embed' data-name='Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder/answer/Vishal-Sharma-154'>Read <a class='quora-content-link' data-width='560' data-height='260' href='https://www.quora.com/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder/answer/Vishal-Sharma-154' data-type='answer' data-id='66853410' data-key='a5099035f08fbac1ed45a4bb7a1c5d2c' load-full-answer='False' data-embed='trhonms'><a href='https://www.quora.com/Vishal-Sharma-154'>Vishal Sharma</a>&#039;s <a href='/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder?top_ans=66853410'>answer</a> to <a href='/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder' ref='canonical'><span class="rendered_qtext">What&#039;s the difference between a Variational Autoencoder (VAE) and an Autoencoder?</span></a></a> on <a href='https://www.quora.com'>Quora</a><script type="text/javascript" src="https://www.quora.com/widgets/content"></script></span>

    * [ Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
    * [Using Artificial Intelligence to Augment Human Intelligence](https://distill.pub/2017/aia/)
    
    * [VAEs and GANs
Mihaela Rosca](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/VAE_GANS_by_Rosca.pdf)
    * [Going Beyond GAN? New DeepMind VAE Model Generates High Fidelity Human Faces](https://syncedreview.com/2019/06/06/going-beyond-gan-new-deepmind-vae-model-generates-high-fidelity-human-faces/)
{:.message}
