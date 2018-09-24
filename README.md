# Dictionary Learning
This repository provides some basic experiments and tools for training a linear dictionary for image patches. The training process yields a dictionary which can be used along with a sparse code to represent a signal.
This procedure is originally described in "Emergence of simple-cell receptive field properties by learning a sparse code for natural images", by Olshausen and Field [Nature, 381:607–609, 1996](https://www.nature.com/articles/381607a0).
It is famously used in "Learning Fast Approximations of Sparse Coding" (Gregor and Lecun)
 and recently in "LSALSA: efficient sparse coding in single and multiple dictionary settings" ([Cowen, Saridena, Choromanska](https://arxiv.org/abs/1802.06875))

We train by minimizing <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{F}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{F}" title="\mathbf{F}" /></a>
w.r.t. <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>
:

<a href="https://www.codecogs.com/eqnedit.php?latex=F(\mathbf{A})&space;=&space;\frac1P&space;\sum_{p=1}^P&space;1/2||&space;\mathbf{y}(p)-\mathbf{A}\mathbf{x}(p)||_2^2&space;&plus;&space;\alpha||\mathbf{x}(p)||_1," target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(\mathbf{A})&space;=&space;\frac1P&space;\sum_{p=1}^P&space;1/2||&space;\mathbf{y}(p)-\mathbf{A}\mathbf{x}(p)||_2^2&space;&plus;&space;\alpha||\mathbf{x}(p)||_1," title="F(\mathbf{A}) = \frac1P \sum_{p=1}^P 1/2|| \mathbf{y}(p)-\mathbf{A}\mathbf{x}(p)||_2^2 + \alpha||\mathbf{x}(p)||_1," /></a>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\geq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\geq&space;0" title="\alpha \geq 0" /></a>
 is the L1 weight,
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>
 is the dictionary,
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)" title="\mathbf{y}(p)" /></a>
is the
<a href="https://www.codecogs.com/eqnedit.php?latex=p^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^{th}" title="p^{th}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=y(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(p)" title="y(p)" /></a>
 signal, and
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}(p)" title="\mathbf{x}(p)" /></a>
is its corresponding sparse code.

After successful minimization, the following should hold:
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)&space;\approx&space;\mathbf{A}\mathbf{x}(p)," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)&space;\approx&space;\mathbf{A}\mathbf{x}(p)," title="\mathbf{y}(p) \approx \mathbf{A}\mathbf{x}(p)," /></a>
for <a href="https://www.codecogs.com/eqnedit.php?latex=p=1,...,P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p=1,...,P" title="p=1,...,P" /></a>.
In other words, the sparse vector <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}(p)" title="\mathbf{x}(p)" /></a>
multiplied with the dictionary <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{A}" title="\mathbf{A}" /></a>
provides an efficient approximation to the signal <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}(p)" title="\mathbf{y}(p)" /></a>

.
