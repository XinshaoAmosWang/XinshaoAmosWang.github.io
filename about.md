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

* Machine Learning: Deep Metric Learning, Robust Representation Learning under Arbitrary Anomalies. 

* Computer Vision: Image/Video Recognition, Person Re-identification (ReID). 



## Robust Learning and Robust Inference in the context of deep learning: noisy labels, noisy observations, outliers, adversaries, etc. 

**Why is it important?**

DNNs can fit well training examples with random lables. 'Understanding deep learning requires rethinking generalization, [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)'

In the large-scale training datasets, noisy training data points generally exist. Specifically and explicitly, the observations and their corresponding semantic labels may not matched. `Emphasis Regularisation by Gradient Rescaling for Training Deep Neural Networks with Noisy Labels, [https://arxiv.org/pdf/1905.11233.pdf](https://arxiv.org/pdf/1905.11233.pdf)'

Fortunately, the concept of adversarial examples become universe/unrestricted now, i.e., any examples that fool a model can be viewed as a adversary, e.g., examples with noisy labels which are fitted well during training, outliers which are fitted well during training or get high confidence scores during testing, examples with small pixel perturbation and perceptually ignorable which fool the model.

Paper reading: [https://drive.google.com/file/d/1fU3N_u-_puOwEbupK6aOENerP2S45tZX/view?usp=sharing](https://drive.google.com/file/d/1fU3N_u-_puOwEbupK6aOENerP2S45tZX/view?usp=sharing)

Github page: [https://github.com/XinshaoAmosWang/Emphasis-Regularisation-by-Gradient-Rescaling/wiki/Robust-Learning-and-Robust-Inference-in-the-context-of-deep-learning:-noisy-examples,-outliers,-adversaries,-etc](https://github.com/XinshaoAmosWang/Emphasis-Regularisation-by-Gradient-Rescaling/wiki/Robust-Learning-and-Robust-Inference-in-the-context-of-deep-learning:-noisy-examples,-outliers,-adversaries,-etc)


**Are deep models robust to massive noise intrinsically?**

* No: [DNNs can fit well training examples with random lables.](https://arxiv.org/abs/1611.03530)
* Yes: [Deep Learning is Robust to Massive Label Noise](https://arxiv.org/abs/1705.10694)?

* You may have your own answer if you read: 

    * [How to preserve MAE's (mean absolute error) noise-tolerance and improve its fitting ability?](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE)

    * [An Extremely Simple and Principled Solution for Avoiding Overfitting and Achieving Better Generalisation](https://github.com/XinshaoAmosWang/Emphasis-Regularisation-by-Gradient-Rescaling)   



**Intuitive concepts to keep in mind**

* The definition of abnormal examples: A training example, i.e., an observation-label pair, is abnormal when an obserevation and its corresponding annotated label for learning supervision are semantically unmatched. 

* Fitting of abnormal examples: When a deep model fits an abnormal example, i.e., mapping an oberservation to a semantically unmatched label, this abnormal example can be viewed as an successful adversary, i.e., an unrestricted adversarial example. 

* Learning objective: A deep model is supposed to extract/learn meaningful patterns from training data, while avoid fitting any anomaly. 