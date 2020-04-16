---
layout: post
title: Paper Summary on Distance Metric, Representation Learning
description: >
  
image: /assets/img/blog/steve-harvey.jpg
comment: true
---

:+1: means being highly related to my personal research interest. 
0. [ICCV 2019, CVPR 2020 Deep Metric Learning](#iccv-2019-cvpr-2020-deep-metric-learning)
0. [CVPR 2019 Deep Metric Learning](#cvpr-2019-deep-metric-learning)
0. [Few-shot Learning](#few-shot-learning)
0. [Large Output Spaces](#large-output-spaces)
0. [Poincaré, Hyperbolic, Curvilinear](#poincaré-hyperbolic-curvilinear)
0. [Wasserstein](#wasserstein)
0. [Semi-supervised or Unsupervised Learning](#semi-supervised-or-unsupervised-learning)
0. [NeurIPS 2019-Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers](#neurips-2019-stochastic-shared-embeddings-data-driven-regularization-of-embedding-layers)
{:.message}


## ICCV 2019, CVPR 2020 Deep Metric Learning
* [Mic: Mining interclass characteristics for improved metric learning-Karsten Roth∗ , Biagio Brattoli⋆ , Bjorn Ommer](http://openaccess.thecvf.com/content_ICCV_2019/papers/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.pdf)
    * The common approach to metric learning is to **enforce a representation that is invariant under all factors but the ones of interest.** (Very Common Practice)
    * In contrast, we propose to **explicitly learn the latent characteristics that are shared by and go across object classes**. We can then directly explain away structured visual variability, rather than assuming it to be unknown random noise. (Being Contrastive is Interesting! => Regularisation Technique?)
    * We propose a novel surrogate task to learn visual characteristics shared across classes with a separate encoder. This encoder is trained jointly with the encoder for class information by reducing their mutual information.
    * ResNet-50 + PyTorch
    * Complex methods for me:  **The number of clusters is set before training to a fixed, problem-specific value**: 30 for CUB200-2011 [37], 200 for CARS196 [19], 50 for Stanford Online Products [28], 150 for In-Shop Clothes [43] and 50 for PKU VehicleID [21]. We update the cluster labels every other epoch.
    * For all experiments, we use the original images without bounding boxes.


* [Cross-Batch Memory for Embedding Learning-Xun Wang∗ , Haozhi Zhang∗ , Weilin Huang†, Matthew R. Scott](https://arxiv.org/pdf/1912.06798.pdf)
    * We propose a cross-batch memory (XBM) mechanism that memorizes the embeddings of past iterations, allowing the model to collect sufficient hard negative pairs across multiple minibatches - even over the whole dataset.
    * GoogLeNet V1, V2 and ResNet-50
    
* [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling, Qian, Qi and Shang, Lei and Sun, Baigui and Hu, Juhua and Li, Hao and Jin, Rong](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)
    * **Multiple Centers or Adaptive Number of Centers => Softmax Loss** 
    * Analogous to  ProxyNCA or ProxyTriplet 
    * **Considering that images in CUB-2011 and Cars196 are similar to those in ImageNet, we freeze BN on these two data sets and keep BN training on the rest one.** Embeddings of examples and centers have the unit length in the experiments. 
    * Backbone: GoogLeNet V2 (Inception with BN)
    * During training, only random horizontal mirroring and random crop are used as the data augmentation. A single center crop is taken for test.
    * CUB-2011: We note that different works report the results with different dimension of embeddings while the size of embeddings has a significant impact on the performance. **For fair comparison, we report the results for the dimension of 64, which is adopted by many existing methods and the results with 512 feature embeddings, which reports the state-of-the-art results on most of data sets.**
    
* [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/pdf/2002.10857.pdf)
    *  Motivation: aiming to maximize the within-class similarity $$s_p$$ and minimize the between-class similarity $$s_n$$. We find a majority of loss functions, including the triplet loss and the softmax plus cross-entropy loss, embed $$s_n$$ and $$s_p$$ into similarity pairs and seek to reduce $$(s_n − s_p)$$. Such an optimization manner is inflexible, because the penalty strength on every single similarity score is restricted to be equal. 
    * Our intuition is that if a similarity score deviates far from the optimum, it should be emphasized.
    * we simply re-weight each similarity to highlight the less-optimized similarity scores. It results in a Circle loss, which is named due to its circular decision boundary. 
    *  Circle loss offers a more flexible optimization approach towards a more definite convergence target,
compared with the loss functions optimizing $$(s_n − s_p)$$.
    * (1)  a unified loss function; (2) flexible optimization; (3) definite convergence status. 
    * Evaluation:
        * Tasks: 
            * Face recognition 
            * Person re-identification (Market-1501,MSMT17)
            * Fine-grained image retrieval (CUB-100-2011, CARS-196, SOP-11318)

        * Net architecture-1: ResNet50 (globla) + MGN (local features) for person reid (. Our implementation concatenates all the part features into a single feature vector for simplici);

        * Net architecture-2: GoogLeNet (BN-Inception) for CUB, CARS, SOP, 512-D embeddings;
    
    * The performance is not better than Ranked List Loss on SOP. 
    
* [Sampling Wisely: Deep Image Embedding by Top-k Precision Optimization](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lu_Sampling_Wisely_Deep_Image_Embedding_by_Top-K_Precision_Optimization_ICCV_2019_paper.pdf)
    * In contrast, in this paper, we propose a novel deep image embedding algorithm with end-to-end optimization to top-k precision, the evaluation metric that is **closely related to user experience.**
    * Specially, our loss function is constructed with **Wisely Sampled “misplaced” images along the top-k nearest neighbor decision boundary,** so that the gradient descent update directly
promotes the concerned metric, top-k precision.
    * **Our theoretical analysis** on the upper bounding and consistency properties of the proposed loss supports that minimizing our proposed loss is equivalent to maximizing top-k precision
    * Evaluation:
        * Datasets: CUB-200-2011, CARS-196, SOP 
        * PyTorch + Adam
        * Net architecture: Densenet 201, GoogLeNet V2 (Inception with BN)
        * Finetuning
        * Embedding size: 64, 512? 
        * Input size: warp (256x256) => crop (227x227)
        * Testing: only center crop
    * The performance is not better than Ranked List Loss
{:.message}

## CVPR 2019 Deep Metric Learning 
* [Divide and Conquer the Embedding Space for Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf) 
:+1:  
    * ResNet-50
    * Each learner will learn a separate distance metric using only a subspace of the original embedding space and **a part of the data**. 

    * Natural hard negatives mining: Finally, **the splitting and sampling connect to hard negative mining**, which is verified by them. (I appreciate this ablation study in Table 6 )
    * Divide means: 
    (1) Splitting the training data into K Clusters; 
    (2) Splitting the embedding into K Slices.  


* [Deep Metric Learning to Rank=FastAP](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf) :+1:
    * ResNet-18 & ResNet-50
    * Our main contribution is a novel solution to optimizing Average Precision under the Euclidean metric, based on the probabilistic interpretation of AP as the area under precision-recall curve, as well as distance quantization.
    * We also propose a category-based minibatch sampling strategy and a large-batch training heuristic.
    * On three **few-shot image retrieval datasets**, FastAP consistently outperforms competing methods, which often involve complex optimization heuristics or costly model ensembles.


* [Multi-Similarity Loss With General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) :+1:
    * Objective of the proposed multi-similarity loss, which aims to collect informative pairs, and weight these pairs through their own and relative similarities.
    * GoogLeNet V2 (Inception BN)


* [Ranked List Loss for Deep Metric Learning](https://arxiv.org/pdf/1903.03238.pdf) :+1:
    * GoogLeNet V2 (Inception BN)

* [Stochastic Class-Based Hard Example Mining for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) :+1:
    * Inception V1
    * Scale linearly to the number of classes. 
    * The methods proposed by Movshovitz-Attias et al. [14] and Wen et al. [34] are related to ours in a sense that class representatives are jointly trained with the feature extractor. 
However, their goal is to formulate new losses using the class representatives whereas we use them for hard negative mining.
    * Given an anchor instance, our algorithm first selects a few hard negative classes based on the class-to-sample distances and then performs a refined search in an instance-level only from the selected classes.

* A Theoretically Sound Upper Bound on the Triplet Loss for Improving the Efficiency of Deep Distance Metric Learning :+1:

* **Unsupervised** Embedding Learning via Invariant and Spreading Instance Feature :+1:


* [Signal-To-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) :+1:

    * We propose a robust SNR distance metric based on Signal-to-Noise Ratio (SNR) for measuring the similarity of image pairs for deep metric learning. Compared with Euclidean distance metric, our SNR distance metric can further jointly reduce the intra-class distances and enlarge the inter-class distances for learned features.
    * SNR in signal processing is used to measure the level of a desired signal to the level of noise, and a larger SNR value means a higher signal quality.
    For similarity measurement in deep metric learning, a pair of learned features x and y can be given as y = x + n, where n can be treated as a noise. Then, the SNR is the ratio of the feature variance and the noise variance.
    * To show the generality of our SNR-based metric, we also extend our approach to hashing retrieval learning.


* [Spectral Metric for Dataset Complexity Assessment](http://openaccess.thecvf.com/content_CVPR_2019/papers/Branchaud-Charron_Spectral_Metric_for_Dataset_Complexity_Assessment_CVPR_2019_paper.pdf) :+1:

    * Related work: [Measuring the Intrinsic Dimension of Objective Landscapes ICLR 2018](https://openreview.net/forum?id=ryup8-WCW), 
    [How Complex is your classification problem? A survey on measuring classification complexity Survey on complexity measures](https://arxiv.org/abs/1808.03591)


* Deep Asymmetric Metric Learning via Rich Relationship Mining :+1:
    * DAMLRRM relaxes the constraint on positive pairs to extend the generalization capability. We build positive pairs training pool by constructing a minimum connected tree for each category instead of considering all positive pairs within a mini-batch. As a result, there will exist a direct or indirect path between any positive pair, which ensures the relevance being bridged to each other. The inspiration comes from ranking on manifold [58] that spreads the relevance to their nearby neighbors one by one.
    * Idea is novel. The results on SOP are not good, only 69.7 with GoogLeNet



* [Hybrid-Attention Based Decoupled Metric Learning for Zero-Shot Image Retrieval](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid-Attention_Based_Decoupled_Metric_Learning_for_Zero-Shot_Image_Retrieval_CVPR_2019_paper.pdf) :-1:
    * Very complex: object attention, spatial attention, random walk graph, etc.

* [Deep Metric Learning Beyond Binary Supervision](https://arxiv.org/pdf/1904.09626.pdf) :-1:
    * Binary supervision indicating whether a pair of images are of the same class or not.
    * Using continuous labels
    * Learn the degree of similarity rather than just the order.
    * A triplet mining strategy adapted to metric learning with continuous labels.
    * Image retrieval tasks with continuous labels in terms of human poses, room layouts and image captions.

* Hardness-aware deep metric learning 
:-1: : data augmentation

* Ensemble Deep Manifold Similarity Learning using Hard Proxies :-1: random walk algorithm, ensemble models.

* Re-Ranking via Metric Fusion for Object Retrieval and Person Re-Identification :-1:

* Deep Embedding Learning With Discriminative Sampling Policy :-1:
* Point Cloud Oversegmentation With Graph-Structured Deep Metric Learning :-1: 
* Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval :-1:
* A Compact Embedding for Facial Expression Similarity :-1:
* [RepMet: Representative-Based Metric Learning for Classification and Few-Shot Object Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karlinsky_RepMet_Representative-Based_Metric_Learning_for_Classification_and_Few-Shot_Object_Detection_CVPR_2019_paper.pdf) :-1:
* Eliminating Exposure Bias and Metric Mismatch in Multiple Object Tracking :-1:
{:.message}

## [Few-shot Learning](../../my_docs/few-shot.md) 
* ICLR 2018-Meta-Learning for Semi-Supervised Few-Shot Classification
* NeurIPS 2019-Unsupervised Meta Learning for Few-Show Image Classification
* NeurIPS 2019-Learning to Self-Train for Semi-Supervised Few-Shot Classification
* NeurIPS 2019-Adaptive Cross-Modal Few-shot Learning
* NeurIPS 2019-Cross Attention Network for Few-shot Classification
* NeurIPS 2019-Incremental Few-Shot Learning with Attention Attractor Networks
* ICML 2019-LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning
{:.message}

## [Large Output Spaces](../../my_docs/large-output-spaces.md)
* NeurIPS 2019-Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Output Spaces
* AISTATS 2019-Stochastic Negative Mining for Learning with Large Output Spaces
{:.message}

## [Poincaré, Hyperbolic, Curvilinear](../../my_docs/Poincare-Hyperbolic-Curvilinear.md)
* NeurIPS 2019-Multi-relational Poincaré Graph Embeddings
* NeurIPS 2019-Numerically Accurate Hyperbolic Embeddings Using Tiling-Based Models
* NeurIPS 2019-Curvilinear Distance Metric Learning
{:.message}



## [Wasserstein](../../my_docs/wasserstein.md)
* NeurIPS 2019-Generalized Sliced Wasserstein Distances 
* NeurIPS 2019-Tree-Sliced Variants of Wasserstein Distances
* NeurIPS 2019-Sliced Gromov-Wasserstein
* NeurIPS 2019-Wasserstein Dependency Measure for Representation Learning
{:.message}


## [Semi-supervised or Unsupervised Learning](../../my_docs/Semi-Un-Supervised-Learning.md)
* CVPR 2019-Label Propagation for Deep Semi-supervised Learning
* NeurIPS 2017-Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
* ICLR 2019-Unsupervised Learning via Meta-Learning
{:.message}




## [NeurIPS 2019-Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers](https://arxiv.org/pdf/1905.10630.pdf)
**NOTE**: 
In deep neural nets, lower level embedding layers account for a large portion of the total number of parameters.**Tikhonov regularization, graph-based regularization, and hard parameter sharing are approaches that introduce explicit biases into training in a hope to reduce statistical complexity.** Alternatively, we propose stochastically shared embeddings (SSE), a data-driven approach to regularizing embedding layers, which stochastically transitions between embeddings during stochastic gradient descent (SGD). Because SSE integrates seamlessly with existing SGD algorithms, it can be used with only minor modifications when training large scale neural networks. We develop two versions of SSE: SSE-Graph using knowledge graphs of embeddings; SSE-SE using no prior information. We provide theoretical guarantees for our method and show its empirical effectiveness on 6 distinct tasks, from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages. **We find that when used along with widely-used regularization methods such as weight decay and dropout, our proposed SSE can further reduce overfitting, which often leads to more favorable generalization results.** <br />
We conducted **experiments for a total of 6 tasks from simple neural networks with one hidden layer in recommender systems, to the transformer and BERT in natural languages.** 
{:.message}

