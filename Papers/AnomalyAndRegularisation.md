---
title: Papers on Anomalies, Regularisation
comment: true
---

Recent papers on Anomalies, Regularisation.


### [ICLR 2019: Learning deep representations by mutual information estimation and maximization](https://openreview.net/forum?id=Bklr3j0cKX)

* **Abstract:** 
    * **Problem&Solution:** This work investigates unsupervised learning of representations by maximizing mutual information between an input and the output of a deep neural network encoder. Importantly, we show that structure matters: incorporating knowledge about locality in the input into the objective can significantly improve a representation's suitability for downstream tasks. We further control characteristics of the representation by matching to a prior distribution adversarially.
    * **Importance:** Our method, which we call Deep InfoMax (DIM), outperforms a number of popular unsupervised learning methods and compares favorably with fully-supervised learning on several classification tasks in with some standard architectures. DIM opens new avenues for unsupervised learning of representations and is an important step towards flexible formulations of representation learning objectives for specific end-goals.

By structuring the network and objectives to encode input locality or priors on the representation, DIM learns features that are useful for downstream tasks without relying on reconstruction or a generative model. 

* [Strong points of the paper are](https://openreview.net/forum?id=Bklr3j0cKX&noteId=BkxA0Kt3nQ): 
    * This gives a principled design of the objective function based on the mutual information between the input data point and output representation. 
    * The performance is gained by incorporating local structures and matching of representation distribution to a certain target (called a prior).



### [ICLR 2019: Deep Anomaly Detection with Outlier Exposure](https://openreview.net/forum?id=HyxCxhRcY7)

* Abstract: 
    * **Targeted problem:** The use of larger and more complex inputs in deep learning magnifies the difficulty of distinguishing between anomalous and in-distribution examples. At the same time, diverse image and text data are available in enormous quantities. We propose leveraging these data to improve deep anomaly detection by training anomaly detectors against an auxiliary dataset of outliers, an approach we call Outlier Exposure (OE). This enables anomaly detectors to generalize and detect unseen anomalies. 
    * Results: 
    In extensive experiments on natural language processing and small- and large-scale vision tasks, we find that Outlier Exposure significantly improves detection performance. We also observe that cutting-edge generative models trained on CIFAR-10 may assign higher likelihoods to SVHN images than to CIFAR-10 images; we use OE to mitigate this issue. We also analyze the flexibility and robustness of Outlier Exposure, and identify characteristics of the auxiliary dataset that improve performance.
    * TL, DR: OE teaches anomaly detectors to learn heuristics for detecting unseen anomalies; experiments are in classification, density estimation, and calibration in NLP and vision settings; we do not tune on test distribution samples, unlike previous work. 
    * Code: [https://github.com/hendrycks/outlier-exposure](https://github.com/hendrycks/outlier-exposure)


* Comments: [https://openreview.net/forum?id=HyxCxhRcY7&noteId=Skl4qxWN67](https://openreview.net/forum?id=HyxCxhRcY7&noteId=Skl4qxWN67)
    * For softmax classifier, the OE loss forces the posterior distribution to become uniform distribution on outlier dataset. 

* Comments: [https://openreview.net/forum?id=HyxCxhRcY7&noteId=Bye5XYYT37](https://openreview.net/forum?id=HyxCxhRcY7&noteId=Bye5XYYT37)
    * For classification, the fine-tuning objective encourages out-of-distribution samples to have a uniform distribution over all class labels. 
    * For density estimation, the objective encourages out-of-distribution samples to be ranked as less probability than in-distribution samples. 
    * The biggest weakness in this paper is the assumption that we have access to out-of-distribution data, and that we will encounter data from that same distribution in the future. For the typical anomaly detection setting, we expect that anomalies could look like almost anything. For example, in network intrusion detection (a common application of anomaly detection), future attacks are likely to have different characteristics than past attacks, but will still look unusual in some way. The challenge is to define "normal" behavior in a way that captures the full range of normal while excluding "unusual" examples. This topic has been studied for decades.
    * My initial read of this paper was incorrect -- the authors do indeed separate the outlier distribution used to train the detector from the outlier distribution used for evaluation. 


### [ICLR 2020 Under review: Deep Semi-Supervised Anomaly Detection](https://openreview.net/forum?id=HkgH0TEYwH)

 * Abstract: 
    * Problem: Typically anomaly detection is treated as an unsupervised learning problem. In practice however, one may have---in addition to a large set of unlabeled samples---access to a small pool of labeled samples, e.g. a subset verified by some domain expert as being normal or anomalous. 
    * Prior work: Semi-supervised approaches to anomaly detection aim to utilize such labeled samples, but most proposed methods are limited to merely including labeled normal samples. Only a few methods take advantage of labeled anomalies, with existing deep approaches being domain-specific. 
    * This work: we present Deep SAD, an end-to-end deep methodology for general semi-supervised anomaly detection. Using an information-theoretic perspective on anomaly detection, we derive a loss motivated by the idea that the entropy of the latent distribution for normal data should be lower than the entropy of the anomalous distribution. 
    * Evalation: We demonstrate in extensive experiments on MNIST, Fashion-MNIST, and CIFAR-10, along with other anomaly detection benchmark datasets, that our method is on par or outperforms shallow, hybrid, and deep competitors, yielding appreciable performance improvements even when provided with only little labeled data.
     * Code: [https://tinyurl.com/y6rwhn5r](https://tinyurl.com/y6rwhn5r). 

* Comments: [https://openreview.net/forum?id=HkgH0TEYwH&noteId=SylNjaV9iS](https://openreview.net/forum?id=HkgH0TEYwH&noteId=SylNjaV9iS)
    * Other recent work on deep anomaly detection utilizing self-supervised classification have incorporated the use of anomalies during training in a similar way, albeit without theoretical justification [5]. In these methods normal samples are trained to minimize a classification loss. On the other hand anomalous samples are trained so that the softmax output distribution has high entropy, not for misclassification. This results in a network where the softmax output from normal samples are concentrated around at the corners of the probability simplex, and anomalous samples are diffusely spread around the center. Our information-theoretic framework offers a potential explanation for why such an objective is natural and connects it to our method.

* Comments: [https://openreview.net/forum?id=HkgH0TEYwH&noteId=BylBYZcAFH](https://openreview.net/forum?id=HkgH0TEYwH&noteId=BylBYZcAFH)
    * Starting from the assumption that abnormal data are sampled from background unpredicted distribution, rather than “cluster” assumption, it is argued that conventional discriminative formulation is not applicable. Motivated by the recent deep AD methods (e.g., deep SVDD), the paper proposes to approach semi-supervised AD from the information theoretic perspective where 1) mutual information between raw data and learnt representation should be maximized (infomax principle), 2) entropy of labeled positive data should be minimized (“compactness” constraint), and 3) enrtropy of labeled negative data should be maximized to reflect the uncertainty assumption of anomaly. The solution is implemented by the encoder of a pre-trained autoencoder that is further fine tuned to enforce entropy assumption on all types of training data. 






