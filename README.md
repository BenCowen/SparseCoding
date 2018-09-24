# Dictionary Learning
This repository provides some basic experiments and tools for training a linear dictionary for image patches. The training process yields a dictionary which can be used along with a sparse code to represent a signal.

We train by minimizing <a href="https://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F" title="F" /></a> w.r.t. <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a>, using stochastic gradient descent:

<a href="https://www.codecogs.com/eqnedit.php?latex=F(A)&space;=&space;\frac1P&space;\sum_{p=1}^P&space;1/2||&space;y(p)-Ax(p)||_2^2&space;&plus;&space;\alpha||x(p)||_1," target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(A)&space;=&space;\frac1P&space;\sum_{p=1}^P&space;1/2||&space;y(p)-Ax(p)||_2^2&space;&plus;&space;\alpha||x(p)||_1," title="F(A) = \frac1P \sum_{p=1}^P 1/2|| y(p)-Ax(p)||_2^2 + \alpha||x(p)||_1," /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\geq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\geq&space;0" title="\alpha \geq 0" /></a> is the L1 weight,e is 
<a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a> is the dictionary,
<a href="https://www.codecogs.com/eqnedit.php?latex=y(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(p)" title="y(p)" /></a>
 is the <a href="https://www.codecogs.com/eqnedit.php?latex=p^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^{th}" title="p^{th}" /></a>
 signal, and <a href="https://www.codecogs.com/eqnedit.php?latex=x(p)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x(p)" title="x(p)" /></a> is its corresponding sparse code


