---
image: /assets/img/blog/steve-harvey.jpg
---





## Xinshao Wang, PDRA, University of Oxford. 
{:.lead}

* Machine Learning (Deep Metric Learning, Robust Representation Learning under Adverse
Conditions, e.g., Noisy Data and Sample Imbalance). 

* Computer Vision (Image/Video Recognition, Person Re-identification). 

* [What am I working on now? Discussions are Welcome!](../blogs/2020-02-21-learn-bayesian-DL/#what-am-i-working-on-now-discussions-are-welcome) 

* [Something new to come soon!]()

## Featured Research Delivering: 

* [Open discussion on deep robustness, please and thanks!](https://www.reddit.com/r/MachineLearning/comments/hjlayq/r_open_discussion_on_deep_robustness_please/)

* [Progressive Self Label Correction (ProSelfLC) for Training Robust Deep Neural Networks](../blogs/2020-06-07-Progressive-self-label-correction)

* [Code Releasing of Recent Work--Derivative Manipulation and IMAE](../blogs/2020-06-14-code-releasing)


* [Ranked List Loss for Deep Metric Learning](https://arxiv.org/pdf/1903.03238.pdf)


* [Instance Cross Entropy for Deep Metric Learning and its application in SimCLR-A Simple Framework for Contrastive Learning of Visual Representations](https://www.reddit.com/r/MachineLearning/comments/f4x1sh/r_instance_cross_entropy_for_deep_metric_learning/?utm_content=post&utm_medium=twitter&utm_source=share&utm_name=submit&utm_term=t3_f4x1sh)

## Hightlight: Robust Learning and Inference under Adverse Conditions, e.g., noisy labels or observations, outliers, adversaries, sample imbalance (long-tailed), etc. 

**Why important?**

DNNs can brute forcelly fit well training examples with random lables (non-meaningful patterns): 
* [Derivative Manipulation and IMAE](../blogs/2020-06-14-code-releasing)
* [Progressive Self Label Correction (ProSelfLC) for Training Robust Deep Neural Networks](../blogs/2020-06-07-Progressive-self-label-correction)
* [Understanding deep learning requires rethinking generalization](https://openreview.net/pdf?id=Sy8gdB9xx)
* [A Closer Look at Memorization in Deep Networks](https://arxiv.org/pdf/1706.05394.pdf)
* Fortunately, **the concept of adversarial examples become universe/unrestricted now, i.e., any examples that fool a model can be viewed as a adversary**. For example:
    * Examples with noisy labels which are fitted well during training;
    * Out-of-distribution data points which are fitted well during training or get high confidence scores during testing;
    * Examples with small pixel perturbation and perceptually ignorable which fool a model.

In the large-scale training datasets, noisy training data points generally exist. Specifically and explicitly, the observations and their corresponding semantic labels may not matched. 








##  Are deep models robust to massive noise intrinsically?

* No: [DNNs can fit well training examples with random lables.](https://arxiv.org/abs/1611.03530)
* Yes: [Deep Learning is Robust to Massive Label Noise](https://arxiv.org/abs/1705.10694)?

* You may have your own answer if you read [Featured Research Delivering](#featured-research-delivering),   [ProSelfLC & Confidence penalty & Label Smoothing & Ouput Regularisation](../blogs/2020-06-07-Progressive-self-label-correction)



## Intuitive concepts to keep in mind

* The definition of abnormal examples: A training example, i.e., an observation-label pair, is abnormal when an obserevation and its corresponding annotated label for learning supervision are semantically unmatched. 

* Fitting of abnormal examples: When a deep model fits an abnormal example, i.e., mapping an oberservation to a semantically unmatched label, this abnormal example can be viewed as an successful adversary, i.e., an unrestricted adversarial example. 

* Learning objective: A deep model is supposed to extract/learn meaningful patterns from training data, while avoid fitting any anomaly. 


## Related papers reading 
* [OutlierDetection_RobustInference using Mahalanobis Distance](https://github.com.cnpmjs.org/XinshaoAmosWang/DerivativeManipulation/blob/master/OutlierDetection_RobustInference.pptx.pdf)

* [Distance is not always what it seems](https://blogs.sas.com/content/iml/2012/02/15/what-is-mahalanobis-distance.html)

* [Detecting outliers in SAS: Part 3: Multivariate location and scatter & MCD: Robust estimation by subsampling](https://blogs.sas.com/content/iml/2012/02/02/detecting-outliers-in-sas-part-3-multivariate-location-and-scatter.html)



## Linkedin Profile

<div class="LI-profile-badge"  data-version="v1" data-size="medium" data-locale="en_US" data-type="horizontal" data-theme="dark" data-vanity="xinshaowang">

<a class="LI-simple-link" href='https://uk.linkedin.com/in/xinshaowang?trk=profile-badge'>Xinshao Wang, PDRA, University of Oxford. </a>

</div>