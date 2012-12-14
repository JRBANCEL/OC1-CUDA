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

        # Loading and compiling sources
        impuritySource = loadSource("impurity.c")
        impuritySource.c = c
        impuritySource.d = d
        impuritySource.n = n
        impuritySource.s = d*n
        impurity1_1Kernel = cudaCompile(str(impuritySource), "impurity1")
        impurity2_1Kernel = cudaCompile(str(impuritySource), "impurity2")
        impurity3_1Kernel = cudaCompile(str(impuritySource), "impurity3")

        impuritySource.s = n
        impurity4_2Kernel = cudaCompile(str(impuritySource), "impurity4")
        impurity2_2Kernel = cudaCompile(str(impuritySource), "impurity2")
        impurity3_2Kernel = cudaCompile(str(impuritySource), "impurity3")

        impuritySource.s = 21
        impurity1_3Kernel = cudaCompile(str(impuritySource), "impurity1")
        impurity2_3Kernel = cudaCompile(str(impuritySource), "impurity2")
        impurity3_3Kernel = cudaCompile(str(impuritySource), "impurity3")

        parallelSplitsSource = loadSource("parallel_splits.c")
        parallelSplitsSource.n = n
        parallelSplitsSource.d = d
        parallelSplitsKernel = cudaCompile(str(parallelSplitsSource),
                                           "parallel_splits")

        perturbCoefficientsSource = loadSource("pertub_coefficients.c")
        perturbCoefficientsSource.c = c
        perturbCoefficientsSource.d = d
        perturbCoefficientsSource.n = n
        perturbCoefficientsSource.s = d*n
        perturbCoefficientsKernel1 = cudaCompile(str(perturbCoefficientsSource),
                                                "perturb1")
        perturbCoefficientsKernel2 = cudaCompile(str(perturbCoefficientsSource),
                                                "perturb2")

        # Allocate memory
        splits = numpy.zeros((d*n, d+2), dtype=numpy.float64)
        position = numpy.zeros((d*n, n), dtype=numpy.uint32)
        position2 = numpy.zeros((n, n), dtype=numpy.uint32)
        position3 = numpy.zeros((21, n), dtype=numpy.uint32)
        count = numpy.zeros((n*d, c, 2), dtype=numpy.uint32)
        count2 = numpy.zeros((n, c, 2), dtype=numpy.uint32)
        count3 = numpy.zeros((21, c, 2), dtype=numpy.uint32)
        tl = numpy.zeros(n*d, dtype=numpy.uint32)
        tl2 = numpy.zeros(d, dtype=numpy.uint32)
        tl3 = numpy.zeros(d, dtype=numpy.uint32)
        impurity = numpy.zeros(n*d, dtype=numpy.float64)
        impurity2 = numpy.zeros(n, dtype=numpy.float64)
        impurity3 = numpy.zeros(21, dtype=numpy.float64)

        samples_d = gpu.to_gpu(samples)
        cat_d = gpu.to_gpu(cat)
        splits_d = gpu.to_gpu(splits)
        position_d = gpu.to_gpu(position)
        position2_d = gpu.to_gpu(position2)
        position3_d = gpu.to_gpu(position3)
        count_d = gpu.to_gpu(count)
        count2_d = gpu.to_gpu(count2)
        count3_d = gpu.to_gpu(count3)
        tl_d = gpu.to_gpu(tl)
        tl2_d = gpu.to_gpu(tl2)
        tl3_d = gpu.to_gpu(tl3)
        impurity_d = gpu.to_gpu(impurity)
        impurity2_d = gpu.to_gpu(impurity2)
        impurity3_d = gpu.to_gpu(impurity3)

        # Compute the splits on parallel axis
        parallelSplitsKernel(samples_d, splits_d, block=(512/d, d, 1),
                             grid=(n/(512/d)+1, 1, 1))

        # Compute impurity of the splits
        impurity1_1Kernel(samples_d, splits_d, position_d, block=(32, 32, 1),
                          grid=(n/32+1, n*d/32+1, 1))
        impurity2_1Kernel(cat_d, count_d, tl_d, position_d,
                          block=(c+1, 512/(c+1), 1),
                          grid=(1, n*d/(512/(c+1))+1, 1))
        impurity3_1Kernel(count_d, tl_d, impurity_d, block=(512, 1, 1),
                          grid=(n*d/512+1, 1, 1))

        #TODO Implement that on GPU
        impurity = impurity_d.get()
        minimum = impurity[0]
        hyperplan = 0
        for index, imp in enumerate(impurity):
            if imp <= minimum:
                hyperplan = index
                minimum = imp

        hyperplan = splits_d.get()[hyperplan]
        hyperplan_d = gpu.to_gpu(hyperplan)

        R = 0
        while R < 10:
            # If not first round, pick a random hyperplan
            if R > 0:
                hyperplan = numpy.zeros(d+2, dtype=numpy.float64)
                hyperplan[:d+1] = numpy.random.uniform(low=-1, high=1, size=d+1)
                hyperplan[d+1] = 1
                hyperplan_d = gpu.to_gpu(hyperplan)

            # Perturb coefficients
            U = numpy.zeros(n, dtype=numpy.float64)
            U_d = gpu.to_gpu(U)
            splits2 = numpy.zeros((n, 2), dtype=numpy.float64)
            splits2_d = gpu.to_gpu(splits2)
            for m in range(d+1):
                perturbCoefficientsKernel1(samples_d, hyperplan_d, U_d,
                                           numpy.uint32(m),
                                           block=(512, 1, 1), grid=(n/512+1, 1, 1))
                perturbCoefficientsKernel2(U_d, splits2_d, block=(512, 1, 1),
                                           grid=(n/512+1, 1, 1))

                impurity4_2Kernel(U_d, splits2_d, position2_d, block=(32, 32, 1),
                                  grid=(n/32+1, n/32+1, 1))
                impurity2_2Kernel(cat_d, count2_d, tl2_d, position2_d,
                                  block=(c+1, 512/(c+1), 1), grid=(1, n*d/512+1, 1))
                impurity3_2Kernel(count2_d, tl2_d, impurity2_d,
                                  block=(512, 1, 1), grid=(n*d/512+1, 1, 1))

                #TODO Implement that on GPU
                impurity = impurity2_d.get()
                minimum = impurity[0]
                split = 0
                for index, imp in enumerate(impurity):
                    if imp <= minimum:
                        split = index
                        minimum = imp
                bestSplit = splits2_d.get()[split]

                # Updating the hyperplan
                H1 = hyperplan.copy()
                H1[m] = bestSplit[0]
                H2 = numpy.zeros((21, d+2), dtype=numpy.float64)
                H2[0] = hyperplan
                for alpha in range(1, 21):
                    H2[alpha] = H1
                    H2[alpha][d] *= alpha/5.

                H_d = gpu.to_gpu(H2)

                # Comparing impurity of the current hyperplan and H1
                impurity1_3Kernel(samples_d, H_d, position3_d,
                                  block=(24, 21, 1), grid=(n/24+1, 1, 1))
                impurity2_3Kernel(cat_d, count3_d, tl3_d, position3_d,
                                  block=(c+1, 21, 1), grid=(1, 1, 1))
                impurity3_3Kernel(count3_d, tl3_d, impurity3_d,
                                  block=(21, 1, 1), grid=(1, 1, 1))

                #TODO Implement that on GPU
                impurity = impurity3_d.get()
                minimum = impurity[0]
                split = 0
                for index, imp in enumerate(impurity):
                    if imp < minimum:
                        split = index
                        minimum = imp
                if R == 0:
                    bestHyperplan = H2[split]
                    bestImpurity = minimum
                    bestSieve = position3_d.get()[split]
                elif imp < bestImpurity:
                    bestHyperplan = H2[split]
                    bestImpurity = minimum
                    bestSieve = position3_d.get()[split]
            R += 1

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


    def trainClassifier(self, training_data, p=0.4):
        # Class array
        cat = numpy.array(training_data[:,-1], dtype=numpy.uint32)
        # Samples array
        samples = numpy.array(training_data[:,:-1], dtype=numpy.float64)
        d = samples.shape[1]

        # Compute the number of categories
        nbCat = set()
        for c in range(cat):
            nbCat.add(c)
        c = len(nbCat)

        # List of sets waiting to be handled
        queue = list()

        # Initial split
        set1, set2, hyperplan, impurity = self.findHyperplan(samples, cat, c)
        self.DT = DecisionTree()
        self.DT.hyperplan = hyperplan[:-1]
        self.length = 3
        impurity1, count1 = self.setImpurity(set1[1], c)
        if impurity1 > p:
            queue.insert(0, (set1, self.DT, "L"))
        else:
            node = DecisionTree()
            node.leaf = True
            node.count = count1
            self.DT.leftChild = node
        impurity2, count2 = self.setImpurity(set2[1], c)
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

            node = DecisionTree()
            node.hyperplan = hyperplan[:-1]

            # Adding a link from the parent to the child
            if side == "L":
                parent.leftChild = node
            else:
                parent.rightChild = node

            # If the impurity is above the threshold, then split again
            impurity, count = self.setImpurity(set1[1], c)
            if impurity > p:
                queue.insert(0, (set1, node, "L"))
            else:
                child = DecisionTree()
                child.leaf = True
                child.count = count
                node.leftChild = child
            impurity, count = self.setImpurity(set2[1], c)
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

        # Return the probability distribution (non normalized)
        # of the sample
        result = numpy.zeros((n, self.c), dtype=uint32)
        for index, cat in enumerate(categories_d.get()):
            result[index] = self.tree[cat][1:]
        return result

    def isTrained(self):
        return self.isTrained

