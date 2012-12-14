OC1-CUDA
========

## Introduction
This project is an attempt to implement an oblique classifier on a GPU using CUDA. The algorithm is inspired from the OC1 algorithm designed by Sreerama K. Murthy. Details of his work can be found [here](http://www.cbcb.umd.edu/~salzberg/announce-oc1.html).

There are two different implementations. The first one parallelize most of the operations within one split of a set but each set is splitted serially. Since I was not satisfied by this implmentation, I implemented the same algorithm with an additional level of parallelism. In the second version, all the splits are done simultenaously. The performances are much better because the first version is linear in the number of node in the decision tree while the second version if linear in the depth of the tree.

## Algorithm
The original algorithm is described in this [thesis](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/thesis.html). I kept the main ideas.

The first step is to find the best parallel split. I do this exhaustively because it is possible to implement it linearly in the number of samples on a GPU while it is exponential in a serial implementation. It exists probabilistic algorithm that are more efficient but more complex to implement in parallel.



## First Version
This version is implemented in the following files
* parallelOC1.py
* impurity.c: kernels related to the computation of impurity of a split
* set_impurity: kernels about compution of the impurity of sets
* classify: kernel about using a tree to classify samples
* parallel_splits.c: kernels about finding the best parallel split
* pertub_coefficients.c: kernels about the pertubations of coefficient phase

