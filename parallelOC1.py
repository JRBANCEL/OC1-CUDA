import numpy
import math

import Cheetah.Template

import pycuda.autoinit
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu

from kNN import Classifier

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
    def trainClassifier(self, training_data):
        # Number of features
        d = training_data.shape[1] - 1
        # Number of training samples
        n = training_data.shape[0]
        # Class array
        cat = numpy.array(training_data[:,-1], dtype=numpy.uint32)
        # Samples array
        samples = numpy.array(training_data[:,:-1], dtype=numpy.float64)

        print "%d samples, %d features" % (n, d)

        #XXX
        c = 2

        impuritySource = loadSource("impurity.c")
        impuritySource.c = c
        impuritySource.d = d
        impuritySource.n = n
        impuritySource.s = d*n
        impurity1Kernel = cudaCompile(str(impuritySource), "impurity1")
        impurity2Kernel = cudaCompile(str(impuritySource), "impurity2")
        impurity3Kernel = cudaCompile(str(impuritySource), "impurity3")

        parallelSplitsSource = loadSource("parallel_splits.c")
        parallelSplitsSource.n = n
        parallelSplitsSource.d = d
        parallelSplitsKernel = cudaCompile(str(parallelSplitsSource),
                                           "parallel_splits")

        # Allocate memory
        splits = numpy.zeros((d*n, d+2), dtype=numpy.float64)
        position = numpy.zeros((d*n, n), dtype=numpy.uint32)
        count = numpy.zeros((n*d, c, 2), dtype=numpy.uint32)
        tl = numpy.zeros(n*d, dtype=numpy.uint32)
        impurity = numpy.zeros(n*d, dtype=numpy.float64)

        samples_d = gpu.to_gpu(samples)
        cat_d = gpu.to_gpu(cat)
        splits_d = gpu.to_gpu(splits)
        position_d = gpu.to_gpu(position)
        count_d = gpu.to_gpu(count)
        tl_d = gpu.to_gpu(tl)
        impurity_d = gpu.to_gpu(impurity)

        # Compute the splits on parallel axis
        parallelSplitsKernel(samples_d, splits_d, block=(512/d, d, 1), grid=(n, 1, 1)) #XXX Fix gridsize

        # Compute impurity of the splits
        print splits_d.get()

        impurity1Kernel(samples_d, splits_d, position_d, block=(n, n*d, 1), grid=(1, 1, 1))
        print position_d.get()
        impurity2Kernel(cat_d, count_d, tl_d, position_d, block=(c+1, n*d, 1), grid=(1, 1, 1))
        print count_d.get(), tl_d.get()
        impurity3Kernel(count_d, tl_d, impurity_d, block=(n*d, 1, 1), grid=(1, 1, 1))

        #TODO Implement that on GPU
        impurity = impurity_d.get()
        minimum = impurity[0]
        for index, imp in enumerate(impurity):
            if imp < minimum:
                hyperplan = index
                minimum = imp

        print "Best Split", splits_d.get()[hyperplan]


        return
        blocksize = (32, 16, 1)
        gridsize  = (int(math.ceil(width/32)), int(math.ceil(height/16)))
        # Step 1
        E_kernel(im_d, E_d, np.int32(width), block=blocksize, grid=gridsize)
        # Number of features
        d = training_data.shape[1] - 1
        # Number of training samples
        n = training_data.shape[0]
        # Class array
        cat = training_data[:,-1]
        # Sample array
        sample = training_data[:,:-1]

    def classifyInstance(self,instance):
    #takes a 1D numpy float array and returns a class
        pass
    def isTrained(self):
    #boolean response of if the classifier is ready to be used
        pass

if __name__ == "__main__":
#    sample = numpy.array([
#                      [1.1, 1.1, 1],
#                      [1, 2, 1],
#                      [2.1, 1, 0],
#                      [2, 2.1, 0],
#                      [3, 1.9, 0],
#                      [1, 4, 0],
#                      [0.73, 4, 1],
#                      [1, -1, 2],
#                      [1, 1, 2],
#                      [1, 2, 2],
#                      ])
    sample = numpy.array([
                        [1, 3, 0],
                        [1.5, 2.5, 0],
                        [1, 0, 1],
                        [2.5, 2, 1]
                         ])
    classifier = ParallelOC1()
    classifier.trainClassifier(sample)
