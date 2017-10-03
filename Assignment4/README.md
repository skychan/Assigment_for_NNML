# Question: Go Further
Of course, you can do much more. For example, explore what number of hidden units works best, and you'll see that that number is indeed much larger than it was on PA3. Or use your PA3 code to properly train the feedforward NN after its RBM initialization. Or add some more hidden layers. Or... creatively combine everything else that you're learning in this course, to see how much this RBM-basedunsupervised pre-training can do.There's only one more question in this assignment, and it's not one that you need to complete in order to get a good mark. This question is worth only 5% of the grade for the assignment, and it's difficult, so it's mostly here for those of you who feel like taking on a challenge. Not only is the question more difficult in itself, but also I'm not going to give you any hints or verification methods or anything else, except that I'll tell you this: you don't need a lot of computer runtime for answering the question.The partition function a.k.a. normalization constant that you see in the formula for the Boltzmann distribution (the probabality of a particular configuration of an RBM), can take a long time to compute. That's because it's a sum of very many numbers: one for each possible configuration. If you have an RBM with only 2 visible units and 1 hidden unit, then those 3 units mean that there are only 8 possible configurations, so then the partition function can easily be computed. But with a bigger RBM, you quickly run into runtime problems.a4_init not only makes test_rbm_w and some data sets, but also small_test_rbm_w, which has only 10 hidden units (it still has 256 visible units). Calculate the logarithm (base e) of the partition function of that small RBM, and report it with at least two digits after the decimal point.

# Explainzation
This question asks to determine the [Partition Function](https://en.wikipedia.org/wiki/Partition_function_(mathematics)) of given connection matrix. The core definition of Partition Function ($Z$) derives from the fact that:

$$
\begin{cases}
P(\mathbf v, \mathbf h) = \frac1Z\exp(-E(\mathbf v,\mathbf h)) \\
-E(\mathbf v, \mathbf h) = \mathbf a^T\mathbf v + \mathbf b^T \mathbf h + \mathbf v^T \mathbf W \mathbf h
\end{cases}
$$

Hence we have:

$$Z = \sum_{\mathbf v, \mathbf h} \exp (-E(\mathbf v, \mathbf h))$$

Since the computitional complexicity sums up to $O(2^{|\mathbf v| + |\mathbf h|})$, it will be nightmare to calculate it directly.

# Hint
With factorization, we can simplify this procedure. 

$$
\begin{align}
Z & = \sum_{\mathbf v, \mathbf h} \exp (-E(\mathbf v, \mathbf h)) \\
& = \sum_{\mathbf h} \sum_{\mathbf v} \exp(\mathbf a^T\mathbf v + \mathbf b^T \mathbf h + \mathbf v^T \mathbf W \mathbf h) \\
& = \sum_{\mathbf h} \exp(\mathbf b^T \mathbf h)\sum_{v_1\in\{0,1\}}\sum_{v_2\in\{0,1\}} \cdots \sum_{v_n\in\{0,1\}} \exp\left(\sum_i \left(a_i v_i + v_i\mathbf W_{i,:}\mathbf h \right)\right) \\
& = \sum_{\mathbf h} \exp(\mathbf b^T \mathbf h)\sum_{v_1\in\{0,1\}}\sum_{v_2\in\{0,1\}} \cdots \sum_{v_n\in\{0,1\}} \prod_i \exp\left(a_i v_i + v_i\mathbf W_{i,:}\mathbf h \right) \\
& = \sum_{\mathbf h} \exp(\mathbf b^T \mathbf h) \prod_i \sum_{v_i\in\{0,1\}}\exp\left(a_i v_i + v_i\mathbf W_{i,:}\mathbf h \right) \\
& = \sum_{\mathbf h} \exp(\mathbf b^T \mathbf h) \prod_i \left( 1 + \exp\left(a_i + \mathbf W_{i,:}\mathbf h \right)\right) \\
& = \sum_{\mathbf h} \exp(\mathbf b^T \mathbf h) \exp\left(\sum_i\log\left( 1 + \exp\left(a_i + \mathbf W_{i,:}\mathbf h \right)\right) \right)
\end{align}
$$

Thus, we reduce the computional complexcity to $O(|\mathbf v|\cdot 2^{|\mathbf h|})$

# Pseudo Code with matlab

Matrix product can help reduce the coding line and enhance its readability meanwhile. Assume we have a matrix $\mathbf A$ with the size of $|v|\times2^{|h|}$ to store basic elements in the calculation.

First, we create a $|h|\times2^{|h|}$ binary matrix $\mathbf B$, every column of which represent a hidden state.

Then, we can calculate as:

```matlab
Z = sum(exp(b'*B + sum(log(exp(W * B + a) + 1))));
```

To be efficient, we need to swap $\mathbf h, \mathbf v$ if $|\mathbf h| > |\mathbf v|$, and transpose the $\mathbf b, \mathbf a, \mathbf W$, one by one.