---
layout: post
title: I Love Learning and Applying Mathematics, Statistics!
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---


0. [SGD & Newton's Method & Second-order Derivative Optimisation](#sgd--newtons-method--second-order-derivative-optimisation)
0. [Linear algebra](#linear-algebra)
0. [Probabilistic view of the world](#probabilistic-view-of-the-world)
0. [Optimisation](#optimisation)
{:.message}

### SGD & Newton's Method & Second-order Derivative Optimisation
* [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method)
* [Newton's Method: Second-order Derivative Optimisation](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)


### Linear algebra
* [Properties of the Covariance Matrix](http://www.robots.ox.ac.uk/~davidc/pubs/tt2015_dac1.pdf)
* [Covariance and Correlation](https://www.mathsisfun.com/data/correlation.html)
    * Interpreting: **Variable = Axis**
        * Variable = axis/component/one hyper-line
        * Observations of this variable = points of this axis/hyper-line
        * The observations of a variable = one vector of points in this line.
        * $$E(XY)=$$ the dot product of two variables' observation vector (multiple points for each variable) 
        * If $$X,Y$$ are orthogonal, then for any point in $$X$$ and any point in $$Y$$, their dot product are zero, therefore, we have $$E(XY) = 0$$.  
        * Diagonalisation (Orthogonal, Making them independent) =>  Decorrelation
            * In this context, uncorrelation = independence. 
    * Uncorrelation (Orthogonalisaion) using Eigen vectors 
        * **Projection of a normal distribtion** $$ X \sim \mathcal{N}(0,\,{\sigma_x}^2)$$ to a standard normal distribution $$ \mathcal{N}(0,\,1)$$: 
            * **the projected distribution** is $$\mathcal{N}(0,\,\{\cos(\theta) \times \sigma_x \times 1\}^2)$$
            * $$\theta$$ determines their linear dependency: correlation coefficient=$$\cos(\theta)$$. 
        * For each eigen vector: **we project all the original variables to this eigen vector** (an axis in the transformed orthogonal space) => summarise/accumulate those projected variables in this axis => New variable 

            * Two accumulated projected new variables of two axises (eigen vectors) are uncorrelated (being independent now).

            * Eigen value = sum of projected standard deviation.

    * The number of variables, the number of observations, the number of axises/components
        * The number of variables = the number of axis. 
        * When the number of variables (feature dim) > the number of observations? 

    * In a high-dimensional space with orthogonal axises: 
        * Each axis is one independent event/variable. (Without losing generality, feel free to treat it as an unit normal distribution $$~\sim \mathcal{N}(0,\,1)$$) 
        * **The sum of two independent normal random variables is also normal.** However, if the two normal random variables are not independent, then their sum is not necessarily normal.
        * Then, the whole space becomes the combination (summarisation) of multiple independent normal random variables. 
        
        * In this context, **uncorrelation = independence.**
        Independent variables = indepdent axis/components = orthogonal components.

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
* [Correlation and dependence -1](https://en.wikipedia.org/wiki/Correlation_and_dependence#Correlation_and_independence)
    * Correlation and dependencd are totally different concepts/terms for discribing two random variables. 
    * **In the special case** when $${\displaystyle X}$$ and $${\displaystyle Y}$$ are jointly normal, uncorrelatedness is equivalent to independence.
* [Correlation and dependence-2-Bivariate Normal Distribution](https://www.probabilitycourse.com/chapter5/5_3_2_bivariate_normal_dist.php)
    * The sum of two independent normal random variables is also normal. However, **if the two normal random variables are not independent, then their sum is not necessarily normal.**

    * If $$X$$ and $$Y$$ are **independent**, $$P(Y\|X)=P(Y) => E(XY)=E(X)E(Y) => Cov(X,Y)=E((X-E(X))(Y-E(Y)))=E(XY)-E(X)E(Y) = 0 => $$**Uncorrelated**.
    
    * If $$X$$ and $$Y$$ are uncorrelated, $$Cov(X,Y)=0 => E(XY) = E(X)E(Y) => ?$$
        * [Zero Correlation Implies Independence](http://home.iitk.ac.in/~zeeshan/pdf/The%20Bivariate%20Normal%20Distribution.pdf)
* [Why are we interested in
correlation/dependency?](https://www.actuaries.org.uk/system/files/documents/pdf/correlation.pdf)
* [Conditioning and Independence](https://www.probabilitycourse.com/chapter5/5_2_3_conditioning_independence.php)

* [Positive semi-definite]()
* [Eigen vectors and Diagonalisation]()
* [Eigen values and Determinant]()

* [Mahalanobis distance]()
    * Transformation/projection to the orthogonal space, where one axis is an independent normal distribution
    * One each axis: compute the distance $$\frac{(x-u)^T(x-u)}{\sigma^2}$$
    * Summarise all the distances of all axises. 
{:.message}

### Probabilistic view of the world
* Basic concepts/terms
    * Covariance and correlation 
    * Bivariate Normal Distribution
    * The sum of two **independent** normal distributions
    * Distance = $$num \times standard~deviation$$
        => Mahalanobis distance
    * Diagonalisation (Orthogonal, Making them independent) =>  Decorrelation
        * In this context, uncorrelation = independence. 
    
* Uncorrelation (Orthogonalisaion) using Eigen vectors 
    * **Projection of a normal distribtion** $$ X \sim \mathcal{N}(0,\,{\sigma_x}^2)$$ to a standard normal distribution $$ \mathcal{N}(0,\,1)$$: 
        * **the projected distribution** is $$\mathcal{N}(0,\,\{\cos(\theta) \times \sigma_x \times 1\}^2)$$
        * $$\theta$$ determines their linear dependency: correlation coefficient=$$\cos(\theta)$$. 
        
    * For each eigen vector: **we project all the original variables to this eigen vector** (an axis in the transformed orthogonal space) => summarise/accumulate those projected variables in this axis => New variable 

        * Two accumulated projected new variables of two axises (eigen vectors) are uncorrelated (being independent now).

        * Eigen value = sum of projected standard deviation.

    * Eigen vectors are orthogonal, serving as independent axises where independent variables lie in. 
        * Diagonal covariance matrix: each entry is the square of eigen value.
        * Eigen value = sum of projected standard deviation.
        * **Projection of a normal distribtion**
            * **the projected distribution** is $$\mathcal{N}(0,\,\{\cos(\theta) \times \sigma_x \times 1\}^2)$$

* [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
    * [Distance is not always what it seems](https://blogs.sas.com/content/iml/2012/02/15/what-is-mahalanobis-distance.html)
    * [Detecting outliers in SAS: Part 3: Multivariate location and scatter & MCD: Robust estimation by subsampling](https://blogs.sas.com/content/iml/2012/02/02/detecting-outliers-in-sas-part-3-multivariate-location-and-scatter.html)
    * [Sum of normally distributed random variables-Independent random variables](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables)
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