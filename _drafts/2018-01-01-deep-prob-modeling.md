---
layout: post
title: Bayesian regression with a known function form
mathjax: true
---

*Deep probabilistic programming* is a method of performing "Bayesian" probabilistic modeling on deep learning frameworks. It empowers us to make composite and deep network models represent probabilistic uncertainty in predictions.

Deep probabilistic programming may be characterized by doing Bayesian inference using differentiable programming. Thus, deep probabilistic programming may be called "Bayesian differentiable programming".


##### Why Deep and Differentiable Programming?

Deep learning frameworks (e.g., PyTorch, TensorFlow, MxNet) allow us to define a target model in a *deep* and composite network structure assembling the building blocks of component models (= parametric linear or non-linear functions) that run in data-dependent, procedural and conditional manner. Also, they provide a tool to estimate the model parameters in terms of *differentiable* optimization like stochastic gradient decent (SGD) and back-propagation algorithms.

Since Bayesian modeling is based on a probabilistic model of the generative process relating the observed data with the uncertain latent variables (= generating parameters), it is very desirable to have the representational power of a deep and composite network model to sufficiently describe the potentially complex generative processes with multiple input variables. In addition, the exact computation of the posterior distribution of latent variables based on the prior and the observed data is most likely to be intractable. Thus, this requires approximate Bayesian inference techniques, such as variational inferences (VIs). Thankfully, VIs transform the approximate posterior inference problems into the optimization problems where we optimize the hyperparameters of approximate posterior distribution (often assumed to be a Gaussian distribution)


$$q(\theta | \mu_\theta)$$


since differentiable learning solves the


$$c = \pm\sqrt{a^2 + b^2}$$

supported by deep learning is very critical in keeping the sufficient model representative power of the underlying process. DL models can describe quite flexible and complex generative processes.

RNN time-sequence modeling + Bayesian


##### Why Bayesian Inference?

In Bayesian modeling we posit that the model parameters (a.k.a. latent or hidden variables) generating the observed data are *uncertain* in our knowledge. Thus, our beliefs about the true values of *generating* variables are described by a probability. That is, we use a probability to denote our uncertainty about the hidden variables selected to describe the generating process. Let's take the example of a Bayesian parametric regression such as
$$y = f(\mathbf{x}; \theta) + \epsilon $$ where $$\mathbf{x}$$ and $$y$$ are the given input vector and the observed output variable (scalar), and $$\theta$$ is the *uncertain* latent variables (vector) or the generating parameters described by a probability distribution. There are important modeling assumptions.

- $$f(\mathbf{x}; \theta)$$ is an assumed generating function we specify with random noise $$\epsilon \sim \mathrm{Normal}(0, \sigma^2)$$.
- $$f(\mathbf{x}; \theta)$$ is a deterministic function for any sampled value of $$\theta \sim \mathrm{Normal} (\mathbf{0}, \sigma_{\theta}^2 \mathbf{I})$$.
- The level of $$\sigma^2$$ is assumed to be fixed as a constant value and also related to how accurately we may specify our function $$f(\mathbf{x}; \theta)$$.

Whereas a non-Bayesian (deterministic) approach views $$\theta$$ as a fixed variable to be estimated, a Bayesian (probabilistic) approach regards $$\theta$$ as an uncertain variable whose *probability distribution* is to be estimated to explain the observed data. Maximum likelihood (ML) or maximum a posteriori (MAP) estimations are well-known non-Bayesian approaches determining a fixed $$\theta$$.

Now let's represent the above Bayesian regression in terms of probability distributions.

The complete generative process in the Bayesian perspective is always described in the joint probability distribution of all observed and latent variables.
Since $$\mathbf{x}$$ is given and $$\sigma^2$$ is assumed to be fixed  here, the joint distribution for the complete generative process is $$p(y,\theta | \mathbf{x}, \sigma^2)$$. Factorizing
$$p(y,\theta | \mathbf{x}, \sigma^2) = p(y | \theta,
\mathbf{x}, \sigma^2) p(\theta)$$, we represent the complete generative process in the combination of the likelihood and the prior distributions. It is important to note that the exact forms of the likelihood and the prior distributions are part of our modeling assumptions.

- The *likelihood* $$p(y | \theta, \mathbf{x}, \sigma^2)$$ is our assumed probability model to describe a generating process of the observed variable $$y$$ from a sample of latent variables $$\theta$$. Assuming that the likelihood is normally distributed with $$f(\mathbf{x}; \theta)$$ as $$y_\mu$$ (= the expected value of $$y$$) and $$\sigma^2$$ as the Gaussian noise level,
$$
\begin{aligned}
y \sim p(y | \theta, \mathbf{x}, \sigma^2) = \mathrm{Normal}(y_\mu, \sigma^2 ) = \mathrm{Normal}(f(\mathbf{x}; \theta), \sigma^2 ).
\end{aligned}
$$

Note that a known deterministic physical model $$y_\mu = f(\mathbf{x}; \theta)$$ can be easily incorporated into the likelihood.

- The *prior* $$p(\theta)$$ is our assumed probability model to represent the uncertain information of latent variables $$\theta$$ (= model parameters) before we consider the observed data $$y$$. The prior probability of model parameters is represented with some hyperparameters. For example, a Gaussian prior with independent parameters can be described with the hyperparameters of prior mean $$\mathbf{\mu}_\theta$$ and prior variance assumed to be  $$\mathbf{\sigma_{\theta}}^2$$. These hyperparmeters are assumed to be known and fixed.
$$\theta \sim p(\theta | \mathbf{\mu}_\theta, \mathbf{\Lambda} ) = \mathrm{Normal} (\mathbf{\mu}_\theta,
\mathrm{diag}(\mathbf{\sigma_{\theta}}^2))$$.


Bayesian posterior inference is to update our belief or probability about $$\theta$$ after observing the data $$y$$.  Mathematically the Bayes' rule provides this update rule.

$$p(\theta| y, \mathbf{x}, \sigma^2) = \frac{ p(y, \theta| \mathbf{x}, \sigma^2) } { p(y | \mathbf{x}, \sigma^2) }
= \frac{ p(y | \theta,
\mathbf{x}, \sigma^2) p(\theta | \mathbf{\mu}_\theta,
\mathbf{\Lambda} )} { p(y | \mathbf{x}, \sigma^2) }$$



evidence $$p(y | \mathbf{x}, \sigma^2) = \int p(y | \theta,
\mathbf{x}, \sigma^2) p(\theta | \mathbf{\mu}_\theta,
\mathbf{\Lambda} )  \,\mathrm{d} \theta $$


intractable
$$\theta$$ high-dimensional

$$p(\theta| y, \mathbf{x}, \sigma^2)$$ approximated by another probability distribution $$q(\theta | \mathbf{\mu'}_\theta,
\mathbf{\Lambda'} ) = \mathrm{Normal} (\mathbf{\mu'}_\theta,
\mathrm{diag}( \mathbf{ {\sigma_{\theta}'}^2 }  ))$$



Note that the Bayesian posterior inference is to estimate
analysis


Bayesian probabilistic modeling provides a unified scheme on how to update the probabilistic beliefs (or infer the posterior distributions) about modeling parameters or latent variables using observed data. It also assumes the specification of generative processes (or model functions describing how outputs are produced from inputs) and prior distributions of modeling parameters or latent variables. This specification allows for easy incorporation of prior knowledge into the model form and associated parameter uncertainty.

Although the initial choice of compared models and associated prior distributions may depend on our domain knowledge about the underlying problems, bayesian reasoning provides an objective scheme to compare different models and priors.

Bayesian modeling allows us to build more robust and less overfitted models under uncertainty and give probabilistic estimates about target variables in the model. It sounds all good and simple, but a key difficulty in Bayesian probabilistic modeling arises from calculating the posterior distributions for a given complicated model structure and prior. It is very often intractable to compute the "exact" posterior distribution, but the variational inferences (VIs), one of the important methods in deep probabilistic programming, present a commonly-applicable approach to compute the "approximate" bayesian posterior.

Since VIs transform Bayesian posterior inference problems (i.e., learning uncertain modeling parameters or latent variables) into optimization problems, the SGD optimization in underlying deep learning frameworks can solve the posterior inference problems.



Again, the Bayesian approach uses a probability to measure the degree of uncertainty of the variables.


the prior distribution is a Bayesian subjective probability of describing our knowledge about $$\theta$$ in an objective fashion.


is only described by a *subjective* probability. That is,

the probabilistic model is a generative process describing how latent variables  

For example, when observation $$y$$ is assumed to be generated depending on model parameters $$\theta$$, we can describe the overall generative process by *joint distribution* $$p(y, \theta) = p(y | \theta ) p(\theta)$$.  Here, $$p(\theta)$$ is the Bayesian specification about the


, the prior belief about the value of $$\theta$$ before observing $$y$$ may be viewed as Bayesian subjective probability of describing our best knowledge about $$\theta$$ in an objective fashion.

In addition, the likelihood probability $$p(y | \theta)$$ assumes a generating process of observation y for a given $$\theta$$.  In other words, both prior and likelihood requires modeling assumptions in terms of their forms.





Bayesian modeling is generative.

 The only way to describe


(e.g., P(y | $$\mu_y$$) the mean $$\mu$$ of the observation $$y$$)



the probabilities we put as a prior or


Bayesian probabilities about the parameters


are subjective,







(e.g., inferring approximate posterior inference)


conditions (parameterized functions executed on  and perform the model parameter optimization by

model optimization using "differentiable" modeling

hyperparameter


differentiable computing techniques that deep learning provides for   In this manner


differentiable programming power.  


Deep Learning (frameworks) as a style of computational modeling language to design a composite (hybrid) model of simple building blocks and optimize them in differentiable computation (backprop & SGDs).  This is now called “Differentiable Programming”.



A deep probabilistic programming framework (e.g., Pyro, Edward) is an extended deep learning framework (e.g., PyTorch, TensorFlow) enabling probabilistic model specifications and Bayesian inferences.



Deep learning frameworks can be viewed as


provide a good way to specify a model by combining parametric functional blocks


 networks of


as the network of parametric functional blocks




> "Differentiable Programming is little more than a rebranding of the modern collection Deep Learning techniques, the same way Deep Learning was a rebranding of the modern incarnations of neural nets with more than two layers. But the important point is that people are now building a new kind of software by assembling networks of parameterized functional blocks and by training them from examples using some form of gradient-based optimization. An increasingly large number of people are defining the network procedurally in a data-dependant way (with loops and conditionals), allowing them to change dynamically as a function of the input data fed to them. It's really very much like a regular progam, except it's parameterized, automatically differentiated, and trainable/optimizable. Dynamic networks have become increasingly popular (particularly for NLP), thanks to deep learning frameworks that can handle them such as PyTorch and Chainer (note: our old deep learning framework Lush could handle a particular kind of dynamic nets called Graph Transformer Networks, back in 1994. It was needed for text recognition)."





Compared to the original deep learning frameworks that aim to make "deterministic" models for point-estimate predictions, DPP frameworks allow us to 1) specify "probabilistic" models involving the uncertain distributions of model parameters or latent variables, 2) provide approximate Bayesian inferences (e.g., variational inferences) using the powerful stochastic gradient decent algorithms of the original DL frameworks. Thus, DPPs naturally integrate the benefits of bayesian modeling and deep learning.




Deep learning frameworks provide a tool to specify models in "deterministic" neural-networks and other kinds of functional equations. It is straightforward to specify a hierarchical (mixed, hybrid) model that combines multiple "component" models. Each component model may be based on different sets of feature variables. Non-probabilistic deep learning models do not assume the uncertainty in model parameters (e.g., weights in NN).







Let's use Pyro to illustrate how to perform Bayesian regression with a known function form.


Suppose we’re given a dataset $$D$$ of the form
$$D = {(X_i,y_i)}$$ for $$i=1,2,...,N$$
The goal of regression is to fit a function to the data of the form:

$$y_i = f(X_i; \theta ) + \epsilon $$

Note that function $$f$$ can be any known form of deterministic equation or neural network involving the uncertain parameter $$\theta $$.  


Let’s first implement regression in PyTorch and learn point estimates for the parameters $$\theta. Then we’ll see how to incorporate uncertainty into our estimates by using Pyro to implement Bayesian regression.

##### Setup

```python
N = 100  # size of toy data
p = 1    # number of features

def build_linear_dataset(N, noise_std=0.1):
    X = np.linspace(-6, 6, num=N)
    y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)
    X, y = X.reshape((N, 1)), y.reshape((N, 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)
```  

- specify model (prior, likelihood)




- specify inference (form of approximated posterior or variational posterior)



Why probabilistic modeling? To correctly capture uncertainty in models and predictions for unsupervised and semi-supervised learning, and to provide AI systems with declarative prior knowledge.

Why (universal) probabilistic programs? To provide a clear and high-level, but complete, language for specifying complex models.

Why deep probabilistic models? To learn generative knowledge from data and reify knowledge of how to do inference.

Why inference by optimization? To enable scaling to large data and leverage advances in modern optimization and variational inference.




 enables Pyro programs to include stochastic control structure, that is, random choices in a Pyro program can control the presence of other random ... To enable scaling to large data and leverage advances in modern optimization and variational inference.




 This naturally combines the advantages of two schemes.  The deep learning scheme allows us to specify the model structure (function form)

specify a model via neural nets or other parameteric funciton forms


Let $$Y_1,\ldots,Y_N$$ be $$(d+1)$$-dimensional observations (collecting the $$X_n\in\mathbb{R}^d$$ covariate within each $$Y_n\in\mathbb{R}$$ response for shorthand) generated from some model with unknown parameters $$\theta\in\Theta$$.

__Goal__: Find the "true" parameters $$\theta^* \in\Theta$$.

__Intuition__: The idea is to find a set of $$k$$ constraints, or "moments", involving the parameters $$\theta$$. What makes GMMs nice is that you need no information per say about how the model depends on $$\theta$$. Certainly they can be used to construct moments (special case: maximum likelihood estimation (MLE)), but one can use, for example, statistical moments (special case: method of moments (MoM)) as the constraints. Analogously, tensor decompositions are used in the case of spectral methods.

More formally, the $$k$$ __moment conditions__ for a vector-valued function $$g(Y,\cdot):\Theta\to\mathbb{R}^k$$ is

\[
m(\theta^* ) \equiv \mathbb{E}[g(Y,\theta^* )] = 0_{k\times 1},
\]

where $$0_{k\times 1}$$ is the $$k\times 1$$ zero vector.

As we cannot analytically derive the expectation for arbitrary $$g$$, we use the sample moments instead:

\[
\hat m(\theta) \equiv \frac{1}{N}\sum_{n=1}^N g(Y_n,\theta)
\]

By the Law of Large Numbers, $$\hat{m}(\theta)\to m(\theta)$$, so the problem is thus to find the $$\theta$$ which sets $$\hat m(\theta)$$ to zero.

Cases:

* $$\Theta\supset\mathbb{R}^k$$, i.e., there are more parameters than moment
conditions: The model is not [identifiable](http://en.wikipedia.org/wiki/Identifiability). This is the standard scenario in ordinary least squares (OLS) when there are more covariates than observations and so no unique set of parameters $$\theta$$ exist. Solve this by simply constructing more moments!
* $$\Theta=\mathbb{R}^k$$: There exists a unique solution.
* $$\Theta\subset\mathbb{R}^k$$,
i.e., there are fewer parameters than moment conditions: The parameters are overspecified and the best we can do is to minimize $$m(\theta)$$ instead of solve $$m(\theta)=0$$.

Consider the last scenario: we aim to minimize $$\hat m(\theta)$$ in some way, say $$\|\hat m(\theta)\|$$ for some choice of $$\|\cdot\|$$. We define the __weighted norm__ as

$$
\|\hat m(\theta)\|_W^2 \equiv \hat m(\theta)^T W \hat m(\theta),
$$

where $$W$$ is a positive definite matrix.

The __generalized method of moments__ (GMMs) procedure is to find

$$
\hat\theta = {arg\ min}_{\theta\in\Theta}
\left(\frac{1}{N}\sum_{n=1}^N g(Y_n,\theta)\right)^T W
\left(\frac{1}{N}\sum_{n=1}^N g(Y_n,\theta)\right)
$$

Note that while the motivation is for $$\theta\supset\mathbb{R}^k$$, by the unique solution, this is guaranteed to work for $$\Theta=\mathbb{R}^k$$ too. Hence it is a _generalized_ method of moments.

__Theorem__. Under standard assumptions¹, the estimator $$\hat\theta$$ is [consistent](http://en.wikipedia.org/wiki/Consistent_estimator#Bias_versus_consistency) and [asymptotically normal](http://en.wikipedia.org/wiki/Asymptotic_distribution). Furthermore, if

$$
W \propto
\Omega^{-1}\equiv\mathbb{E}[g(Y_n,\theta^*)g(Y_n,\theta^*)^T]^{-1}
$$

then $$\hat \theta$$ is [asymptotically optimal](http://en.wikipedia.org/wiki/Efficiency_(statistics)), i.e., achieves the Cramér-Rao lower bound.

Note that $$\Omega$$ is the covariance matrix of $$g(Y_n,\theta^*)$$ and $$\Omega^{-1}$$ the precision. Thus the GMM weights the parameters of the estimator $$\hat\theta$$ depending on how much "error" remains in $$g(Y,\cdot)$$ per parameter of $$\theta^*$$ (that is, how far away $$g(Y,\cdot)$$ is from 0).

I haven't seen anyone make this remark before, but the GMM estimator can also be viewed as minimizing a log-normal quantity. Recall that the multivariate normal distribution is proportional to

$$
\exp\Big((Y_n-\mu)^T\Sigma^{-1}(Y_n-\mu)\Big)
$$

Setting $$g(Y_n,\theta)\equiv Y_n-\mu$$, $$W\equiv\Sigma$$, and taking the log, this is exactly the expression for the GMM! By the asymptotic normality, this explains why would want to set $$W\equiv\Sigma$$ in order to achieve statistical efficiency.

¹ The standard assumptions can be found in [1]. In practice they will almost always be satisfied, e.g., compact parameter space, $$g$$ is continuously differentiable in a neighborhood of $$\theta^*$$, output of $$g$$ is never infinite, etc.

## References
[1] Alastair Hall. _Generalized Method of Moments (Advanced Texts in Econometrics)_. Oxford University Press, 2005.


@inproceedings{tran2017deep,
  author = {Dustin Tran and Matthew D. Hoffman and Rif A. Saurous and Eugene Brevdo and Kevin Murphy and David M. Blei},
  title = {Deep probabilistic programming},
  booktitle = {International Conference on Learning Representations},
  year = {2017}
}
