import math
import numpy
import time
import sys

import Cheetah.Template

import pycuda.autoinit
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu

from kNN import Classifier

class DecisionTree():
    def __init__(self):
        self.leftChild = None
        self.rightChild = None
        self.hyperplan = None
        self.leaf = False

def cudaCompile(sourceString, functionName):
    sourceModule = nvcc.SourceModule(sourceString)
    return sourceModule.get_function(functionName)

def loadSource(filename):
    """Return a file as a Cheetah template"""
    sourceFile = open(filename, 'r')
    source = sourceFile.read()
    sourceFile.close()
    return Cheetah.Template.Template(source)

class ParallelOC1(Classifier):
    def __init__(self):
        self.isTrained = False

    def findHyperplan(self, samples, cat, c):
        n = samples.shape[0]
        d = samples.shape[1]
        s = 1
        t = 3

        # Loading and compiling sources
        source = loadSource("parallel_build.c")
        source.c = c
        source.d = d
        source.n = n
        source.s = s
        source.t = n*d
        parallelSplitsKernel = cudaCompile(str(source), "parallel_splits")
        reduceImpurityKernel = cudaCompile(str(source), "reduce_impurity")
        source.t = t
        computeUKernel = cudaCompile(str(source), "compute_U")
        computeImpurityUKernel = cudaCompile(str(source), "compute_impurity_U")
        reduceImpurityUKernel = cudaCompile(str(source), "reduce_impurity_U")
        initIndexUKernel = cudaCompile(str(source), "init_index_U")
        positionHyperplan = cudaCompile(str(source), "compute_position_hyperplans")
        countHyperplan = cudaCompile(str(source), "compute_count_hyperplans")

        # Allocate memory
        impurity = numpy.ones((s, n*d), dtype=numpy.float64)
        index = numpy.zeros((s, n*d), dtype=numpy.uint32)
        for i in range(s):
            index[i] = numpy.array(range(n*d), dtype=numpy.uint32)
        hyperplan = numpy.zeros((s, n*d, d+1), dtype=numpy.float64)
        sieve = numpy.ones(n, dtype=numpy.uint32)
        U = numpy.zeros((s, t, n), dtype=numpy.float64)
        impurityU = numpy.zeros((s, t, n), dtype=numpy.float64)
        indexU = numpy.zeros((s, t, n), dtype=numpy.uint32)
        position = numpy.zeros((s, t, n), dtype=numpy.uint32)
        countL = numpy.zeros((s, t, c),  dtype=numpy.uint32)
        countR = numpy.zeros((s, t, c),  dtype=numpy.uint32)
        Tl = numpy.zeros((s, t),  dtype=numpy.uint32)
        Tr = numpy.zeros((s, t),  dtype=numpy.uint32)

        samples_d = gpu.to_gpu(samples)
        cat_d = gpu.to_gpu(cat)
        impurity_d = gpu.to_gpu(impurity)
        hyperplan_d = gpu.to_gpu(hyperplan)
        sieve_d = gpu.to_gpu(sieve)
        index_d = gpu.to_gpu(index)
        U_d = gpu.to_gpu(U)
        impurityU_d = gpu.to_gpu(impurityU)
        indexU_d = gpu.to_gpu(indexU)
        position_d = gpu.to_gpu(position)
        countL_d = gpu.to_gpu(countL)
        countR_d = gpu.to_gpu(countR)
        Tl_d = gpu.to_gpu(Tl)
        Tr_d = gpu.to_gpu(Tr)

        # Compute the splits on parallel axis
        parallelSplitsKernel(samples_d, sieve_d, cat_d, hyperplan_d, impurity_d,
                             block=(16, 16, 2), grid=(n/16+1, d/16+1, s/2+1))

        # Reduction to find the hyperplan with the smallest impurity
        size = n*d
        while size > 1:
            reduceImpurityKernel(impurity_d, index_d, numpy.uint32(size),
                                 block=(512, 1, 1), grid=(n*d/512+1, s, 1))
            size = math.ceil(size/2.)

        # Array of hyperplan to be optimized
        hyperplans = numpy.zeros((s, t, d+1), dtype=numpy.float64)
        for i in range(s):
            # Adding the best parallel split
            hyperplans[i][0] = hyperplan_d.get()[i][index_d.get()[i][0]]

            # Adding random hyperplans
            for j in range(1, t):
                hyperplans[i][j] = numpy.random.uniform(low=-1, high=1, size=d+1)

        for m in range(d+1):
            print '-'*20
            # Sending the hyperlans to the GPU
            hyperplans_d = gpu.to_gpu(hyperplans)

            # Computing the U vectors
            computeUKernel(samples_d, sieve_d, hyperplans_d, numpy.uint32(m),
                           U_d, block=(16, 1, 32), grid=(t/16+1, s, n/32+1))


            # Compute the impurity of U splits
            computeImpurityUKernel(samples_d, sieve_d, cat_d, hyperplans_d,
                                   impurityU_d ,U_d, block=(32, 16, 1),
                                   grid=(n/32+1, t/16+1, s))

            # Find minimal impurity
            initIndexUKernel(indexU_d, block=(32, 1, 16),
                             grid=(n/32+1, s, t/16+1))
            size = n
            while size > 1:
                reduceImpurityUKernel(impurityU_d, indexU_d, numpy.uint32(size),
                                     block=(32, 1, 16), grid=(n/32+1, s, t/16+1))
                size = math.ceil(size/2.)

            # Compute the impurity of the initial hyperplans
            positionHyperplan(samples_d, sieve_d, hyperplans_d, position_d,
                             block=(1, 16, 32), grid=(s, t/16+1, n/32+1))
            print cat
            print position_d.get()
            print "#"*10
            countHyperplan(cat_d, position_d, countL_d, countR_d, Tl_d, Tr_d,
                           block=(1, 32, 16), grid=(s, t/32+1, 16/c+2))
            print Tl_d.get(), countL_d.get()
            print Tr_d.get(), countR_d.get()
#impurityHyperplan(


        return

#                hyperplan = numpy.zeros(d+2, dtype=numpy.float64)
#                hyperplan[:d+1] = numpy.random.uniform(low=-1, high=1, size=d+1)
#                hyperplan[d+1] = 1
#                hyperplan_d = gpu.to_gpu(hyperplan)
#
#                #TODO Implement that on GPU
#                impurity = impurity3_d.get()
#                minimum = impurity[0]
#                split = 0
#                for index, imp in enumerate(impurity):
#                    if imp < minimum:
#                        split = index
#                        minimum = imp
#                if R == 0:
#                    bestHyperplan = H2[split]
#                    bestImpurity = minimum
#                    bestSieve = position3_d.get()[split]
#                elif imp < bestImpurity:
#                    bestHyperplan = H2[split]
#                    bestImpurity = minimum
#                    bestSieve = position3_d.get()[split]
#            R += 1

        # Compute the two sets of samples
        return (samples[bestSieve>0], cat[bestSieve>0]),\
               (samples[bestSieve<1], cat[bestSieve<1]),\
               bestHyperplan, bestImpurity

    def setImpurity(self, cat, c):
        """Compute the impurity of a set"""
        impuritySource = loadSource("set_impurity.c")
        impuritySource.c = c
        impuritySource.n = cat.shape[0]
        impurityKernel = cudaCompile(str(impuritySource), "set_impurity1")

        count = numpy.zeros(c, dtype=numpy.uint32)
        count_d = gpu.to_gpu(count)
        cat_d = gpu.to_gpu(cat)

        impurityKernel(cat_d, count_d, block=(c, 1, 1), grid=(1, 1, 1))
        count = count_d.get()
        # This could be computed on the GPU but it is a really small computation
        # I am not sure if it is worth it
        return 1 - numpy.sum([x*x for x in count])/float(cat.shape[0]**2), count


    def trainClassifier(self, training_data):
        # Class array
        cat = numpy.array(training_data[:,-1], dtype=numpy.uint32)
        # Samples array
        samples = numpy.array(training_data[:,:-1], dtype=numpy.float64)
        d = samples.shape[1]
        # Number of category TODO compute it from the input
        c = 4
        # Impurity threshold
        p = 0.4

        self.findHyperplan(samples, cat, c)
        return

        # List of sets waiting to be handled
        queue = list()

        # Initial split
        set1, set2, hyperplan, impurity = self.findHyperplan(samples, cat, c)
#        print "Set1: %d, Set2: %d, Impurity: %f, Hyperplan:" %\
#              (len(set1[1]), len(set2[1]), impurity), hyperplan
        self.DT = DecisionTree()
        self.DT.hyperplan = hyperplan[:-1]
        self.length = 3
        impurity1, count1 = self.setImpurity(set1[1], c)
#        print "Impurity 1:", impurity1
        if impurity1 > p:
            queue.insert(0, (set1, self.DT, "L"))
        else:
            node = DecisionTree()
            node.leaf = True
            node.count = count1
            self.DT.leftChild = node
        impurity2, count2 = self.setImpurity(set2[1], c)
#        print "Impurity 2:", impurity2
        if impurity2 > p:
            queue.insert(0, (set2, self.DT, "R"))
        else:
            node = DecisionTree()
            node.leaf = True
            node.count = count2
            self.DT.rightChild = node

        while True:
            try:
                subset = queue.pop()
                samples = subset[0][0]
                cat = subset[0][1]
                parent = subset[1]
                side = subset[2]
            except:
                break

            set1, set2, hyperplan, impurity = self.findHyperplan(samples, cat, c)
#            print hyperplan

#            if len(set1[1]) > 0 and len(set2[1]) > 0:
            node = DecisionTree()
            node.hyperplan = hyperplan[:-1]

            # Adding a link from the parent to the child
            if side == "L":
                parent.leftChild = node
            else:
                parent.rightChild = node

            # If the impurity is above the threshold, then split again
            impurity, count = self.setImpurity(set1[1], c)
#            print "Impurity 1:", impurity
            if impurity > p:
                queue.insert(0, (set1, node, "L"))
            else:
                child = DecisionTree()
                child.leaf = True
                child.count = count
                node.leftChild = child
            impurity, count = self.setImpurity(set2[1], c)
#            print "Impurity 2:", impurity2
            if impurity > p:
                queue.insert(0, (set2, node, "R"))
            else:
                child = DecisionTree()
                child.leaf = True
                child.count = count
                node.rightChild = child
            self.length += 2

        # Build and send the tree to GPU
        # Note that the structure on the GPU doesn't need to have
        # the probability distribution. In any case, a tree is small
        # in memory so that it doesn't really matter
        self.tree = self.buildTree(d, c)
        self.tree_d = gpu.to_gpu(self.tree)
        self.isTrained = True

    def buildTree(self, d, c):
        """Build the tree to be mapped in GPU memory"""
        tree = numpy.zeros((self.length, max(d+1, c)+1), dtype=numpy.float64)
        current = 0
        next = 1

        queue = list([self.DT])
        while len(queue) > 0:
            node = queue.pop()
            if node.leaf:
                tree[current][1:c+1] = node.count
            else:
                tree[current][0] = next
                tree[current][1:d+2] = node.hyperplan
                queue.insert(0, node.leftChild)
                queue.insert(0, node.rightChild)
                next += 2
            current += 1

        return tree

    def displayTree(self):
        queue = list([self.DT])
        while len(queue) > 0:
            node = queue.pop()
            if node.leaf:
                print "Leaf:", node.count
            else:
                print "Hyperplan:", node.hyperplan
                queue.insert(0, node.leftChild)
                queue.insert(0, node.rightChild)

    def classifyInstance(self, samples):
        n = samples.shape[0]
        d = samples.shape[1]
        classifySource = loadSource("classify.c")
        classifySource.d = d
        classifySource.n = n
        classifySource.l = self.tree.shape[0]
        classifySource.w = self.tree.shape[1]
        classifyKernel = cudaCompile(str(classifySource), "classify1")

        categories = numpy.zeros(n, dtype=numpy.uint32)
        categories_d = gpu.to_gpu(categories)
        samples_d = gpu.to_gpu(samples)
        classifyKernel(samples_d, self.tree_d, categories_d, block=(512, 1, 1),
                       grid=(n/512 + 1, 1, 1))
#        for index, cat in enumerate(categories_d.get()):
#            print samples[index], "in category", self.tree[cat][1:]

    def isTrained(self):
        return self.isTrained

