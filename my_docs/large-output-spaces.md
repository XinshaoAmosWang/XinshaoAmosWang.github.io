---
layout: page
title: Large Output Spaces
description: >

image: /assets/img/blog/steve-harvey.jpg
comment: true
---


:+1: means being highly related to my personal research interest. 
{:.message}





## [NeurIPS 2019-Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Output Spaces]()
**NOTE**: 
Not available yet. 
{:.message}


## [AISTATS 2019-Stochastic Negative Mining for Learning with Large Output Spaces](http://proceedings.mlr.press/v89/reddi19a/reddi19a.pdf)
**NOTE**: 
In this paper we specifically consider retrieval tasks
where the objective is to output the k most relevant
classes for an input out of a very large number of
possible classes. Training and test examples consist of
pairs (x, y) where x represents the input and y is one
class that is relevant for it. This setting is common
in retrieval tasks: for example, x might represent a
search query, and y a document that a user clicked on
in response to the search query. **The goal is to learn a set-valued classifier that for any input x outputs a set of k classes that it believes are most relevant for x, and the model is evaluated based on whether the class y is captured in these k classes.** <br />
To this end, we first define **a family of surrogate losses** and show that they are **calibrated and convex under certain conditions on the loss parameters and data distribution, thereby establishing a statistical and analytical basis for using these losses.** Furthermore, we identify a particularly intuitive class of loss functions in the aforementioned family and show that they are amenable to practical implementation in the large output space setting (i.e. **computation is possible without evaluating scores of all labels) by developing a technique called Stochastic Negative Mining.** We also provide **generalization error bounds for the losses in the family.** Finally, we conduct experiments which demonstrate that Stochastic Negative Mining yields benefits over commonly used negative sampling approaches.
{:.message}



