---
image: /assets/img/blog/steve-harvey.jpg
---

<!---
<div class="LI-profile-badge"  data-version="v1" data-size="medium" data-locale="en_US" data-type="horizontal" data-theme="dark" data-vanity="xinshaowang">
<a class="LI-simple-link" href='https://uk.linkedin.com/in/xinshaowang?trk=profile-badge'>Xinshao Wang, PhD Student, Queens University Belfast, Anyvision. </a>
</div>
-->

Xinshao Wang, PhD Student, Queens University Belfast, Anyvision. 
{:.lead}

* Machine Learning (Deep Metric Learning, Robust Representation Learning under Adverse
Conditions, e.g., Noisy Data and Sample Imbalance). 

* Computer Vision (Image/Video Recognition, Person Re-identification). 



### Robust Learning and Inference under Adverse Conditions, e.g., noisy labels, noisy observations, outliers, adversaries, etc. 

**Why is it important?**

DNNs can fit well training examples with random lables. 'Understanding deep learning requires rethinking generalization, [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)'

In the large-scale training datasets, noisy training data points generally exist. Specifically and explicitly, the observations and their corresponding semantic labels may not matched. 

Fortunately, the concept of adversarial examples become universe/unrestricted now, i.e., any examples that fool a model can be viewed as a adversary, e.g., examples with noisy labels which are fitted well during training, outliers which are fitted well during training or get high confidence scores during testing, examples with small pixel perturbation and perceptually ignorable which fool the model.


### Research Delivering: 
* [Derivative Manipulation for General Example Weighting](https://github.com/XinshaoAmosWang/DerivativeManipulation)
* [IMAE for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE)
* [Instance Cross Entropy for Deep Metric Learning and its application in SimCLR-A Simple Framework for Contrastive Learning of Visual Representations](https://www.reddit.com/r/MachineLearning/comments/f4x1sh/r_instance_cross_entropy_for_deep_metric_learning/?utm_content=post&utm_medium=twitter&utm_source=share&utm_name=submit&utm_term=t3_f4x1sh)
* [Code Releasing of My Recent Work-Derivative Manipulation](https://xinshaoamoswang.github.io/blogs/2020-02-18-code-releasing/)


###  Are deep models robust to massive noise intrinsically?

* No: [DNNs can fit well training examples with random lables.](https://arxiv.org/abs/1611.03530)
* Yes: [Deep Learning is Robust to Massive Label Noise](https://arxiv.org/abs/1705.10694)?

* You may have your own answer if you read [Research Delivering](#research-delivering),   [Confidence penalty & Label Smoothing && Ouput Regularisation](https://xinshaoamoswang.github.io/blogs/2020-02-14-Core-machine-learning-topics/#confidence-penalty--label-smoothing--ouput-regularisation)



### Intuitive concepts to keep in mind

* The definition of abnormal examples: A training example, i.e., an observation-label pair, is abnormal when an obserevation and its corresponding annotated label for learning supervision are semantically unmatched. 

* Fitting of abnormal examples: When a deep model fits an abnormal example, i.e., mapping an oberservation to a semantically unmatched label, this abnormal example can be viewed as an successful adversary, i.e., an unrestricted adversarial example. 

* Learning objective: A deep model is supposed to extract/learn meaningful patterns from training data, while avoid fitting any anomaly. 


### Related papers reading 
[https://drive.google.com/file/d/1fU3N_u-_puOwEbupK6aOENerP2S45tZX/view?usp=sharing](https://drive.google.com/file/d/1fU3N_u-_puOwEbupK6aOENerP2S45tZX/view?usp=sharing)

