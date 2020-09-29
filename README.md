# UPDATE 9 Sept 2020:
I tried pulling and running this to find it's incompatible with the latest pytorch, and with Windows. I will be updating it over the next week-- it won't run right now.
-Ben

# Usage
This demo requires [PyTorch](https://pytorch.org/).

To start an example, execute `python run_demo.py`, using ` --help` to see optional arguments. The default experiment is a dictionary learning demo with MNIST.

[//]: <`matlab_type_demo.py` is a sort of MATLAB-style script (an old version of the code that is less flexible). The results are visualized in the `results` subdirectory. You can play around with both model and optimization parameters in this file.>

## Purpose
The ultimate goal of this repository is to provide a sparse coding library that implements experimental platforms for (1) dictionary learning, (2) traditional/convex code inference (e.g. ISTA, SALSA), and (3) "unrolled" learnable encoders (e.g. LISTA,  [LSALSA](https://arxiv.org/abs/1802.06875)).

Dictionary learning is up-and-running, now. In particular I am building an encoder class that combines (2) and (3). Then I will generalize the class for morphological component analysis (MCA), a sparse-coding approach to source separation.

# Sparse Coding Background
It is often useful to represent a signal or image in terms of its basic building blocks. For example, a smiley-face can be efficiently described as "a circle, two dots, and a curve". At least, that is more efficient than "pixel 1: value 0.1. Pixel 2: value 1" and so on for thousands of pixels. This is a rudimentary example of "sparse representation"-- i.e., if we have a dictionary of shapes and curves, we can often describe an image as a weighted-sum of those dictionary elements. The fewer the number of dictionary atoms used, the more efficient/sparse the representation is. We refer to the list of weights to use as a "sparse code" when most of the weights are zero.

Sparse coding is the problem of jointly recovering the dictionary and the codes, given some data.
This repository provides some tools and classes for various sparse coding experiments.
As of now, the focus is on learning a linear dictionary (e.g. for vectors, including vectorized image patches) from data.
The training process yields a dictionary-- i.e. a matrix, whose rows are the dictionary elements-- which can be used along with a sparse code to represent a signal.

![CIFAR10](paramSearchResults/CIFAR1010_0210.png)
![F-mnist10](paramSearchResults/FashionMNIST10_0220.png)
![asirra16](paramSearchResults/ASIRRA16_0000.png)

CIFAR, Fashion-MNIST, and ASIRRA- based atoms, with patch-sizes 10x10, 10x10, and 16x16, respectively. To see the atoms side-by-side with the corresponding data, ![see this slide](https://github.com/BenCowen/DictionaryLearning/blob/master/data_atoms_comparison.pdf) produced using my Lua-Torch version of this code.

This procedure is originally described in "Emergence of simple-cell receptive field properties by learning a sparse code for natural images", by Olshausen and Field [Nature, 381:607–609, 1996](https://www.nature.com/articles/381607a0).
It is famously used in "Learning Fast Approximations of Sparse Coding" (Gregor and Lecun)
 and recently in "LSALSA: efficient sparse coding in single and multiple dictionary settings" ([Cowen, Saridena, Choromanska](https://arxiv.org/abs/1802.06875)).

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

### TO-DO
* save dictionary objects
* put lua version on (maybe...)
* color version
* training script for encoders
* re-formulate "learned FISTA"
* look into SSNAL (see past team emails)
