---
layout: post
title: I Love Learning and Applying Mathematics, Statistics!
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---



0. [Linear algebra](#linear-algebra)
0. [Probabilistic view of the world](#probabilistic-view-of-the-world)
0. [Optimisation](#optimisation)
{:.message}


### Linear algebra
* [Properties of the Covariance Matrix](http://www.robots.ox.ac.uk/~davidc/pubs/tt2015_dac1.pdf)
* [Covariance and Correlation](https://www.mathsisfun.com/data/correlation.html)
    * Interpreting: **Variable = Axis**
    * Covariance: 
        * remove mean for each axis/variable 
        * projection (accumulation/expectation of dot product of observations from different axis/variable)
        * In other words, dot product of two points (dim >= 2) from two axises (out of multiple axises) => expectation/accumulation
    * **Dot product of two variable/axis**
    * Correlation = $$\frac{Covariance(X_i, X_j)} { \sigma_i \times \sigma_j } \in [-1, 1]$$ 
    * For easier and intuitive understanding, looking at $$\frac{(X_i - E(X_i))} { \sigma_i } \sim \mathcal{N}(0,\,1) = \mathbf{e}_i$$ and $$\frac{(X_j - E(X_j))} { \sigma_j } \sim \mathcal{N}(0,\,1) = \lambda\mathbf{e}_i + \sqrt{(1-\lambda^2)} \mathbf{e}_j$$
        * $$\mathbf{e}_i \sim \mathcal{N}(0,\,1)$$, $$\mathbf{e}_j \sim \mathcal{N}(0,\,1)$$
        * $$\mathbf{e}_i \text{ and } \mathbf{e}_j $$ are two variables in two orthogonal axises. 
        * Correlation = $$\lambda \in [-1, 1]$$  
        * Covariance = $$\lambda \sigma_i \sigma_j$$ 
        * $$
\begin{aligned}
  E\{\frac{(X_i - E(X_i))} { \sigma_i }  \frac{(X_j - E(X_j))} { \sigma_j } \} &= E\{ \mathbf{e}_i \cdot \lambda\mathbf{e}_i + \sqrt{(1-\lambda^2)} \mathbf{e}_j \} \\
  &= \lambda E\{ \mathbf{e}_i \cdot \mathbf{e}_i  \} + \sqrt{(1-\lambda^2)} E\{ \mathbf{e}_i \cdot \mathbf{e}_j \} \\
  &= \lambda .
\end{aligned}$$
        * $$
        \begin{aligned}
  E\{ {(X_i - E(X_i))}  \cdot {(X_j - E(X_j))}  \} =  \lambda  \sigma_i  \sigma_j.
    \end{aligned}
    $$
* [Correlation and independence](https://en.wikipedia.org/wiki/Correlation_and_dependence#Correlation_and_independence)
* [Positive semi-definite]()
* [Eigen vectors and Diagonalisation]()
* [Eigen values and Determinant]()
{:.message}

### Probabilistic view of the world
* [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
    * [Distance is not always what it seems](https://blogs.sas.com/content/iml/2012/02/15/what-is-mahalanobis-distance.html)
    * [Detecting outliers in SAS: Part 3: Multivariate location and scatter & MCD: Robust estimation by subsampling](https://blogs.sas.com/content/iml/2012/02/02/detecting-outliers-in-sas-part-3-multivariate-location-and-scatter.html)
    * [Outlier, Anomaly, and Adversaries Detection using Mahalanobis distance](https://github.com.cnpmjs.org/XinshaoAmosWang/DerivativeManipulation/blob/master/OutlierDetection_RobustInference.pptx.pdf)
    * The Mahalanobis distance is a measure of the **distance between a point $$\mathrm{P}$$ and a distribution $$\mathbf{D}$$.**
    It is a multi-dimensional generalization of the idea of measuring how many standard deviations away P is from the mean of D. This distance is zero if P is at the mean of D, and grows as P moves away from the mean along **each principal component axis.** 
        * If each of these axes is re-scaled to have unit variance, then the Mahalanobis distance corresponds to standard Euclidean distance in the transformed space. 
        * The Mahalanobis distance is thus unitless and scale-invariant, and takes into account the correlations of the data set. 
    * Mahalanobis distance is proportional, for a normal distribution, to the square root of the negative log likelihood (after adding a constant so the minimum is at zero).
    * This intuitive approach can be made quantitative by defining **the normalized distance between the test point and the set to be $${\displaystyle {x-\mu } \over \sigma } $$**. By plugging this into the normal distribution we can derive the probability of the test point belonging to the set.
    * Were the distribution to be decidedly non-spherical, for instance ellipsoidal, then we would expect the probability of the test point belonging to the set to depend not only on the distance from the center of mass, but also on the direction. In those directions where the ellipsoid has a short axis the test point must be closer, while in those where the axis is long the test point can be further away from the center.
    * The Mahalanobis distance is the distance of the test point from the center of mass **divided by the width of the ellipsoid in the direction of the test point.** (distance normalisation)
* [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

* [Properties of the Covariance Matrix](http://www.robots.ox.ac.uk/~davidc/pubs/tt2015_dac1.pdf)

* [Differential entropy](https://en.wikipedia.org/wiki/Differential_entropy)

* [What does Determinant of Covariance Matrix give](https://math.stackexchange.com/questions/889425/what-does-determinant-of-covariance-matrix-give)

* [Why do we use the determinant of the covariance matrix when using the multivariate normal?](https://stats.stackexchange.com/questions/89952/why-do-we-use-the-determinant-of-the-covariance-matrix-when-using-the-multivaria)
{:.message}

### Optimisation
* [Concave function](https://en.wikipedia.org/wiki/Concave_function)
{:.message}