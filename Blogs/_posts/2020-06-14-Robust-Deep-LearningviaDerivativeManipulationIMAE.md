---
layout: post
title: Robust Deep Learning via Derivative Manipulation and IMAE
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---


For source codes, the usage is conditioned on academic use only and kindness to cite our work: Derivative Manipulation and IMAE.<br />
As a young researcher, your interest and kind citation (star) will definitely mean a lot for me and my collaborators.<br />
For any specific discussion or potential future collaboration, please feel free to contact me. 

### When talking about robustness/regularisation, our community tend to connnect it merely to better test performance. I advocate caring training performance as well because: 
* If noisy training examples are fitted well, a model has learned something wrong;
* If clean ones are not fitted well,  a model is not good enough. 
* There is a potential arguement that the test dataset can be infinitely large theorectically, thus being significant. 
  * Personal comment: Though being true theorectically, in realistic deployment, we obtain more testing samples as time goes, accordingly we generally choose to retrain or fine-tune to make the system adaptive. Therefore, this arguement does not make much sense. 

### We really need to rethink robust losses and optimisation in deep learning!
* In [Normalized Loss Functions for Deep Learning with Noisy Labels](https://arxiv.org/abs/2006.13554),   it is stated in the abstract that "**we theoretically show by applying a simple normalization that: any loss can be made robust to noisy labels. However, in practice, simply being robust is not sufficient for a loss function to train accurate DNNs.**"
  * This statement is **Quite** **Contradictory**: A ROBUST LOSS IS NOT SUFFICIENT (i.e., ROBUST AND ACCURATE)?  => **Then what is value to say whether a loss is robust or not?**

* For me, **a trained robust model should be accurate on both training and testing datasets.**

* **I remark that we are the first to thoroughly analyse robust losses, e.g., MAE's underfitting, and how it weights data points.**


0. [IMAE for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters](../../my_docs/IMAE_Code_Illustration)
    * Following work: [Derivative Manipulation for General Example Weighting](https://arxiv.org/pdf/1905.11233.pdf)

0. [Derivative Manipulation for General Example Weighting](../../my_docs/DM_Code_Illustration)
    * Preliminary: [IMAE for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude's Variance Matters](https://arxiv.org/pdf/1903.12141.pdf#arXiv%20Paper.IMAE%20for%20Noise-Robust%20Learning)

0. Github Pages
    * [DerivativeManipulation](https://github.com/XinshaoAmosWang/DerivativeManipulation)
    * [IMAE](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE)
0. Citation
```
@article{wang2019derivative,
  title={Derivative Manipulation for General Example Weighting},
  author={Wang, Xinshao and Kodirov, Elyor and Hua, Yang and Robertson, Neil M},
  journal={arXiv preprint arXiv:1905.11233},
  year={2019}
}
```
```
@article{wang2019imae,
  title={ {IMAE} for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude's Variance Matters},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Robertson, Neil M},
  journal={arXiv preprint arXiv:1903.12141},
  year={2019}
}
```
{:.message}

