OC1-CUDA
========

## Introduction
This project is an attempt to implement an oblique classifier on a GPU using CUDA. The algorithm is inspired from the OC1 algorithm designed by Sreerama K. Murthy. Details of his work can be found [here](http://www.cbcb.umd.edu/~salzberg/announce-oc1.html).

There are two different implementations. The first one parallelize most of the operations within one split of a set but each set is splitted serially. Since I was not satisfied by this implmentation, I implemented the same algorithm with an additional level of parallelism. In the second version, all the splits are done simultenaously. The performances are much better because the first version is linear in the number of node in the decision tree while the second version if linear in the depth of the tree.

## Algorithm

### Global Structure
The original algorithm is described in this [thesis](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/thesis.html). I kept the main ideas.

The first step is to find the best parallel split. I do this exhaustively because it is possible to implement it linearly in the number of samples on a GPU while it is exponential in a serial implementation. It exists probabilistic algorithm that are more efficient but more complex to implement in parallel.

Then the idea is to modify the coefficients of the hyperplan one after the other. This is the perturbation phase. This cannot be parallelized. However, the parallel split is then compared to random splits. Since these splits are fully independent, they can be perturbed and their impurity can be computed in parallel. This is not the case in the first algorithm which compares in parallel different hyperplans but there is still a serial part. It is fully parallel in the second version.

### Differences between version 1 and 2
The version one tests shifted versions of the currently perturbed hyperplan which leads to better results than just working on the original one. The version 2 doesn't do that for the moment but it is not a technical limitation, it is just that I had not the time to implement it.

## First Version
This version is implemented in the following files
* parallelOC1.py
* impurity.c: kernels related to the computation of impurity of a split
* set_impurity: kernels about compution of the impurity of sets
* classify: kernel about using a tree to classify samples
* parallel_splits.c: kernels about finding the best parallel split
* pertub_coefficients.c: kernels about the pertubations of coefficient phase

## Second version
This version is implemented in the following files
* parallelOC1v2.py
* parallel_build.c: all the kernels

## Test it
Choose a impurity threshold which is the stopping condition: the algorithm will keep split sets until each set has a impurity level under that threshold. Impurity is a real between 0 and 1.
Choose the number of points to be generated in each category.
Choose which version of the algorithm to run: 1 or 2

Then to run the benchmark:
> python parallelOC1Benchmark.py number_of_points impurity_threshold version
