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

    def findHyperplan(self, samples, cat, c, sieve, t=10):
        n = samples.shape[0]
        d = samples.shape[1]
        s = sieve.shape[0]

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
        reduceImpurityHyperplanKernel = cudaCompile(str(source), "reduce_impurity")
        computeUKernel = cudaCompile(str(source), "compute_U")
        computeImpurityUKernel = cudaCompile(str(source), "compute_impurity_U")
        reduceImpurityUKernel = cudaCompile(str(source), "reduce_impurity_U")
        initIndexUKernel = cudaCompile(str(source), "init_index_U")
        positionHyperplan = cudaCompile(str(source), "compute_position_hyperplans")
        countHyperplan = cudaCompile(str(source), "compute_count_hyperplans")
        impurityHyperplanKernel = cudaCompile(str(source), "compute_impurity_hyperplans")
        initIndexKernel = cudaCompile(str(source), "init_index_hyperplans")
        sieveKernel = cudaCompile(str(source), "compute_sieves")
        setImpurity = cudaCompile(str(source), "set_impurity")

        # Allocate memory
        impurity = numpy.ones((s, n*d), dtype=numpy.float64)
        index = numpy.zeros((s, n*d), dtype=numpy.uint32)
        for i in range(s):
            index[i] = numpy.array(range(n*d), dtype=numpy.uint32)
        hyperplan = numpy.zeros((s, n*d, d+1), dtype=numpy.float64)
        U = numpy.zeros((s, t, n), dtype=numpy.float64)
        impurityU = numpy.zeros((s, t, n), dtype=numpy.float64)
        indexU = numpy.zeros((s, t, n), dtype=numpy.uint32)
        position = numpy.zeros((s, t, n), dtype=numpy.uint32)
        countL = numpy.zeros((s, t, c),  dtype=numpy.uint32)
        countR = numpy.zeros((s, t, c),  dtype=numpy.uint32)
        Tl = numpy.zeros((s, t),  dtype=numpy.uint32)
        Tr = numpy.zeros((s, t),  dtype=numpy.uint32)
        impurityHyperplan = numpy.zeros((s, t), dtype=numpy.float64)
        indexHyperplan = numpy.zeros((s, t), dtype=numpy.uint32)
        splits = numpy.zeros((s, t ,n), dtype=numpy.float64)

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
        impurityHyperplan_d = gpu.to_gpu(impurityHyperplan)
        indexHyperplan_d = gpu.to_gpu(indexHyperplan)
        splits_d = gpu.to_gpu(splits)

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
            computeImpurityUKernel(samples_d, sieve_d, cat_d, splits_d,
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
#            print "Position", position_d.get()
#            print "#"*10
            countHyperplan(cat_d, position_d, countL_d, countR_d, Tl_d, Tr_d,
                           block=(1, 32, 16), grid=(s, t/32+1, c/16+2))
#            print Tl_d.get(), countL_d.get()
#            print Tr_d.get(), countR_d.get()
            impurityHyperplanKernel(countL_d, countR_d, Tl_d, Tr_d, impurityHyperplan_d,
                              block=(1, 512, 1), grid=(s, t/512+1, 1))
            impurityBefore = impurityHyperplan_d.get()

            # Modify hyperplans with U and compute their impurity
            modifiedHyperplans = hyperplans.copy()
            splits = splits_d.get()
            for i in range(s):
                for j in range(t):
                    modifiedHyperplans[i][j][m] = splits[i][j][indexU_d.get()[i][j][0]]

            modifiedHyperplans_d = gpu.to_gpu(modifiedHyperplans)
            positionHyperplan(samples_d, sieve_d, modifiedHyperplans_d, position_d,
                             block=(1, 16, 32), grid=(s, t/16+1, n/32+1))
#            print position_d.get()
#            print "#"*10
            countHyperplan(cat_d, position_d, countL_d, countR_d, Tl_d, Tr_d,
                           block=(1, 32, 16), grid=(s, t/32+1, c/16+2))
#            print Tl_d.get(), countL_d.get()
#            print Tr_d.get(), countR_d.get()
            impurityHyperplanKernel(countL_d, countR_d, Tl_d, Tr_d, impurityHyperplan_d,
                              block=(1, 512, 1), grid=(s, t/512+1, 1))
            impurityAfter = impurityHyperplan_d.get()
#            print "Before", impurityBefore
#            print "After", impurityAfter

            # Choose the hyperplan with smallest impurity
            for i in range(s):
                for j in range(t):
                    if impurityAfter[i][j] < impurityBefore[i][j]:
                        hyperplans[i][j] = modifiedHyperplans[i][j]

        # Find minimal impurity for every set
        hyperplans_d = gpu.to_gpu(hyperplans)
        positionHyperplan(samples_d, sieve_d, hyperplans_d, position_d,
                         block=(1, 16, 32), grid=(s, t/16+1, n/32+1))
#        print position_d.get()
#        print "#"*10
        countHyperplan(cat_d, position_d, countL_d, countR_d, Tl_d, Tr_d,
                       block=(1, 32, 16), grid=(s, t/32+1, c/16+2))
#        print Tl_d.get(), countL_d.get()
#        print Tr_d.get(), countR_d.get()
        impurityHyperplanKernel(countL_d, countR_d, Tl_d, Tr_d, impurityHyperplan_d,
                          block=(1, 512, 1), grid=(s, t/512+1, 1))
        initIndexKernel(indexHyperplan_d, block=(1, 512, 1),
                        grid=(s, t/512+1, 1))
#        print "Indexes", indexHyperplan_d
#        print "Impurity", impurityHyperplan_d.get()
        size = t
        while size > 1:
            reduceImpurityHyperplanKernel(impurityHyperplan_d, indexHyperplan_d,
                                          numpy.uint32(size), block=(512, 1, 1),
                                          grid=(t/512+1, s, 1))
#            print size, "Indexes", indexHyperplan_d
            size = math.ceil(size/2.)
#        print "Hyperplans", hyperplans
#        print "Indexes", indexHyperplan_d.get()

        # Compute sieves for each new set
        indexes =  indexHyperplan_d.get()
        bestHyperplan = numpy.array(indexes[:,0])
        sieves = numpy.zeros((s, 2, n), dtype=numpy.uint32)
        sieves_d = gpu.to_gpu(sieves)
#        print bestHyperplan
        bestHyperplan_d = gpu.to_gpu(bestHyperplan)
        sieveKernel(position_d, bestHyperplan_d, sieves_d, block=(1, 512, 1),
                    grid=(s, n/512+1, 1))

        # Compute impurity of new sets
        counts = numpy.zeros((s, 2 , c), dtype=numpy.uint32)
        counts_d = gpu.to_gpu(counts)
        T = numpy.zeros((s, 2), dtype=numpy.uint32)
        T_d = gpu.to_gpu(T)
        setImpurity(cat_d, sieves_d, counts_d, T_d, block=(32, 16, 2),
                    grid=(c/32+1, s/16+1, 1))
        counts = counts_d.get()
        T = T_d.get()
#        print "Counts", counts
        impurity = numpy.zeros((s, 2), dtype=numpy.float64)
        print counts
        for i in range(s):
            for j in range(2):
                impurity[i][j] = 1 - numpy.sum([x*x for x in counts[i][j]])/float(T[i][j]**2)

        sieves = sieves_d.get()

        output = list()
        for i in range(s):
            output.append((hyperplans[i][indexes[i][0]],
                          (sieves[i][0], impurity[i][0], counts[i][0]),
                          (sieves[i][1], impurity[i][1], counts[i][1])))
        return output

    def trainClassifier(self, training_data):
        # Class array
        cat = numpy.array(training_data[:,-1], dtype=numpy.uint32)
        # Samples array
        samples = numpy.array(training_data[:,:-1], dtype=numpy.float64)
        n = samples.shape[0]
        d = samples.shape[1]
        # Number of category TODO compute it from the input
        c = 4
        # Impurity threshold
        p = 0.4

        # Initial split
        hyperplan, (sieve1, impurity1, counts1), (sieve2, impurity2, counts2) =\
            self.findHyperplan(samples, cat, c, numpy.ones((1, n),
                               dtype=numpy.uint32))[0]
        print sieve1, impurity1, sieve2, impurity2
        self.DT = DecisionTree()
        self.DT.hyperplan = hyperplan
        self.length = 3
        queue = list()
        if impurity1 > p:
            queue.append((sieve1, self.DT, "L"))
        else:
            node = DecisionTree()
            node.leaf = True
            node.count = counts1
            self.DT.leftChild = node
        if impurity2 > p:
            queue.append((sieve2, self.DT, "R"))
        else:
            node = DecisionTree()
            node.leaf = True
            node.count = counts2
            self.DT.rightChild = node

        while len(queue) > 0:

            # Split all the sets in parallel
            s = len(queue)
            sieve = numpy.zeros((s, n), dtype=numpy.uint32)
            for i in range(s):
                sieve[i] = queue[i][0]
            print "Sieve"
            print sieve

            splits = self.findHyperplan(samples, cat, c, sieve)

            # Browse results
            # The ultimate goal is to write a kernel able to build the tree
            # structure instead of doing it on the CPU
            newQueue = list()
            print '-'*20
            print s
            for i, (hyperplan, (sieve1, impurity1, counts1), (sieve2, impurity2, counts2)) in enumerate(splits):
                print impurity1, impurity2
                node = DecisionTree()
                node.hyperplan = hyperplan
                if impurity1 > p:
                    newQueue.append((sieve1, node, "L"))
                else:
                    newNode = DecisionTree()
                    newNode.leaf = True
                    newNode.count = counts1
                    node.leftChild = newNode
                if impurity2 > p:
                    newQueue.append((sieve2, node, "R"))
                else:
                    newNode = DecisionTree()
                    newNode.leaf = True
                    newNode.count = counts2
                    node.rightChild = newNode

                # Adding a link from the parent to the child
                if queue[i][2] == "L":
                    queue[i][1].leftChild = node
                else:
                    queue[i][1].rightChild = node

                self.length += 2
            queue = newQueue

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

