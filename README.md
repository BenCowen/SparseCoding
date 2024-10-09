# About
All day, every day we unconsciously segment 
the world into tidy pieces.

From distinguishing and identifying elements 
of the physical space around us to breaking 
down abstract ideas into something more 
familiar-- decomposition on all planes of conception is fundamental to our brains.


In mathematical signal processing we use a wide variety of approaches to replicating this automatic decomposition in computer programs.
There are techniques that simply separate components along axes of highest variation amongst (eg 
Principal Component Analysis), and there are nonlinear techniques that seek to parameterize 
these axes so as to  pull out interpretable components (like [Morphological 
Component Analysis](https://arxiv.org/abs/2208.06056)). There are more explicit approaches like 
mathematically beamforming data streams in order to isolate different source signals, and there are 
more implicit ones like training self-supervised neural networks to encode latent axes of 
interest as they see fit.

Welcome to my personal PyTorch library for exploring some of these concepts (and to host
unit-tested code for reproducing results
from [my PhD thesis](https://www.proquest.com/openview/6821866ac2a973b555473b2376dc61f3/1?pq-origsite=gscholar&cbl=18750&diss=y)).

Some tools herein:
- dictionary learning
- differentiable convex optimization algorithms (AKA "unrolled" learnable encoders, e.g. [LISTA](https://icml.cc/Conferences/2010/papers/449.pdf),  [LSALSA](https://arxiv.org/abs/1802.06875))
- Variational Autoencoder
- some Python wrappers for SQL

Tools I'm working on:
- morphological component analysis tools
- [Beyond Backprop](https://arxiv.org/abs/1806.09077) style layer-parallelized training

[//]: # (## Some preliminary visualizations)

[//]: # (<p align="center">)

[//]: # ( <img src="SCRATCH/celeb-dict-mse/loss_history.png" width="375" height="375"/>)

[//]: # (  <img src="SCRATCH/celeb-dict-mse/dict_hist/top100atoms_e3.png" width="375" height="375"/>)

[//]: # (</p>)

# Formal Sparse Coding Background
It is often useful to represent a signal or image in terms of its basic building blocks.
For example, a smiley-face can be efficiently described as "a circle, two dots, and a curve".
At least, that is more efficient than "pixel 1: value 0.1. Pixel 2: value 1" and so on for 
thousands or millions of pixels.
This is a toy example of "sparse representation"-- i.e., if we have a *dictionary* of shapes and 
curves, 
we can often describe an image as a weighted-sum of those dictionary elements. The fewer the number of dictionary atoms used, 
the more efficient or *sparse* the representation is. With a list of dictionary atoms we can 
then write the corresponding list of weights (or coefficients), this list is a vector called the 
code. Codes are specific to dictionaries, and when they are mostly zeroes, we call them *sparse*.

*Sparse coding* is the problem of generating a dictionary and set of corresponding codes for a 
dataset. The idea that, since the  codes will share a common "language" via the dictionary, it 
can be represented more efficiently than the original dataset. You can also take a look at which 
dictionary atoms are most important, which circle back to signal decomposition as discussed above.

This repository provides some tools and classes for various sparse coding experiments.
As of now, the focus is on learning a linear dictionary (e.g. for vectors, including vectorized image patches) from data.
The training process yields a dictionary-- i.e. a matrix, whose rows are the dictionary elements-- which can be used along with a sparse code to represent a signal.

![CIFAR10](legacy-code/paramSearchResults/CIFAR1010_0210.png)
![asirra16](legacy-code/paramSearchResults/ASIRRA16_0000.png)
![F-mnist10](legacy-code/paramSearchResults/FashionMNIST10_0220.png)

CIFAR-, ASIRRA-, and Fashion-MNIST-based atoms, with patch-sizes 10x10, 16x16, and 10x10, 
respectively.

This procedure is originally described in "Emergence of simple-cell receptive field properties by learning a sparse code for natural images", by Olshausen and Field [Nature, 381:607–609, 1996](https://www.nature.com/articles/381607a0).
It is famously used in "Learning Fast Approximations of Sparse Coding" (Gregor and Lecun), which 
inspired more recent papers, i.e.
* "LSALSA: efficient sparse coding in single and multiple dictionary 
settings" ([Cowen, Saridena, Choromanska](https://arxiv.org/abs/1802.06875))
* "Approximate extraction of late-time returns via morphological component analysis" ([Goehle, 
  Cowen, et al.](https://arxiv.org/abs/2208.06056))
* "Phenomenology Based Decomposition of Sea Clutter with a Secondary Target Classifier" ([Farschian, Cowen, Selesnick](https://ieeexplore.ieee.org/abstract/document/10149773))
* "Joint Sparse Coding and Frame Optimization"  ([Goehle, Cowen](https://ieeexplore.ieee.org/document/10382582))

We train by minimizing<a href="https://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F" title="F" /></a>
with respect to the matrix/dictionary/decoder <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>
:

<a href="https://www.codecogs.com/eqnedit.php?latex=F(\mathbf{A})&space;=&space;\frac1P&space;\sum_{p=1}^P&space;\frac12||&space;\mathbf{y}(p)-\mathbf{A}\mathbf{x^*}(p)||_2^2&space;&plus;&space;\alpha||\mathbf{x^*}(p)||_1," target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(\mathbf{A})&space;=&space;\frac1P&space;\sum_{p=1}^P&space;\frac12||&space;\mathbf{y}(p)-\mathbf{A}\mathbf{x^*}(p)||_2^2&space;&plus;&space;\alpha||\mathbf{x^*}(p)||_1," title="F(\mathbf{A}) = \frac1P \sum_{p=1}^P \frac12|| \mathbf{y}(p)-\mathbf{A}\mathbf{x^*}(p)||_2^2 + \alpha||\mathbf{x^*}(p)||_1," /></a>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\geq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\geq&space;0" title="\alpha \geq 0" /></a>
 is a scalar parameter that balances sparsity with reconstruction error,
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>
 is the dictionary,
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)" title="\mathbf{y}(p)" /></a>
is the p-th training data sample, and
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}(p)" title="\mathbf{x}(p)" /></a>
is its corresponding _optimal sparse code_.

What do we mean by optimal sparse code? And why would we optimize an L1 term that does not include
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>
(hence giving a zero subgradient)? The procedure is as follows.
1. Select a batch of image patches (or whatever training data): <a href="https://www.codecogs.com/eqnedit.php?latex=y(p),...,y(p&plus;B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(p),...,y(p&plus;B)" title="y(p),...,y(p+B)" /></a>
2. Compute optimal codes for each <a href="https://www.codecogs.com/eqnedit.php?latex=y(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(p)" title="y(p)" /></a>.
How? Fix <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>.
With fixed <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F" title="F" /></a>
is convex with respect to <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a>!
So, we compute the argument-minimimum with respect to <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a>,
to obtain an optimal code. We call  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x^*}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x^*}(p)" title="\mathbf{x^*}(p)" /></a>
the optimal code of <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)" title="\mathbf{y}(p)" /></a>,
given the current dictionary. In this repo we compute optimal codes using an algorithm called FISTA.
Note: <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x^*}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x^*}(p)" title="\mathbf{x^*}(p)" /></a>
depends on 
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>,
but it does *NOT* depend on the algorithm used to encode <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)" title="\mathbf{y}(p)" /></a>,
since it is a convex problem with a unique solution) 
3. Next, we un-fix <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>, compute the gradient of <a href="https://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F" title="F" /></a>
with respect to <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a>
and perform backpropagation using the batch. 
4. Re-normalize the columns of <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a>.
5. Go back to Step 1 and pull out a fresh batch, unless <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a> has converged.

In summary, we do not couple the problems of sparse coding (producing codes) and training a decoder (a.k.a. dictionary). Rather, we iterate between them.

After successful optimization, the following should hold:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)&space;\approx&space;\mathbf{A}\mathbf{x}(p)," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)&space;\approx&space;\mathbf{A}\mathbf{x}(p)," title="\mathbf{y}(p) \approx \mathbf{A}\mathbf{x}(p)," /></a>
for <a href="https://www.codecogs.com/eqnedit.php?latex=p=1,...,P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p=1,...,P" title="p=1,...,P" /></a>.

In other words, the sparse vector <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}(p)" title="\mathbf{x}(p)" /></a>
multiplied with the (learned) dictionary <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>
provides an efficient approximation to the signal <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)" title="\mathbf{y}(p)" /></a>.

[//]: # (### TO-DO)

[//]: # (* save dictionary objects)

[//]: # (* put lua version on &#40;maybe...&#41;)

[//]: # (* color version)

[//]: # (* training script for encoders)

[//]: # (* re-formulate "learned FISTA")

[//]: # (* look into SSNAL &#40;see past team emails&#41;)

[//]: # (* C++ Tensorflow framework....!)

[//]: # ()

