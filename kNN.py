import numpy
from numpy import *


#####-----Raw KNN Classifier Function-----#####

def kNN(k,data,numClasses,inputs):
#  inputs = instance to classify
#  numClasses = classes of the data (3 classes -> [0,1,2])
#  data = samples / training data
#  k = integer number og neighbors to use

    nData = shape(data)[0]          #num rows of training data
    class_idx = shape(data)[1]-1    #class label stored in last column of training data
    
    #compute distances
    distances = zeros(nData)
    for n in range(nData): distances[n] = sum((inputs-data[n,0:class_idx])**2)
    
    #get indices of the nearest neighbors in ascending order
    indices = argsort(distances)
    
    #get class 
    if numClasses <= 1: return (numpy.float32(0))
    else:
        counts = zeros(numClasses)
        for i in range(k):
            counts[data[indices[i],class_idx]] += 1
        return (numpy.float32(counts.argmax()))

        
#####-----Classifier API-----######
        
class Classifier(object):
#interface for classifier objects
    def trainClassifier(self,training_data):
    #training data is a 2D numpy float array with the last column holding class vals
        pass
    def classifyInstance(self,instance):
    #takes a 1D numpy float array and returns a class
        pass
    def isTrained(self):
    #boolean response of if the classifier is ready to be used
        pass


#####------KNN implemented in terms of Classifier------######
        
class KNN(Classifier):
    k = 0
    num_classes = 0
    num_features = 0
    data = numpy.zeros(0)
    #initialize with the k, number of classes and number of features per vector(not including the class)
    def __init__(self,k,num_classes,num_features):
        self.k = k
        self.num_classes = num_classes
        self.num_features = num_features
    #training is trivial in kNN - just load the examples    
    def trainClassifier(self,training_data):
        self.data = training_data
    #flag as trained once more data than k    
    def isTrained(self):
        return (len(self.data) > self.k)
    #classify a num_features length instance    
    def classifyInstance(self,instance):
        nData = shape(self.data)[0]          #num rows of training data
        class_idx = self.num_features        #class label stored in last column of training data
        #compute distances
        distances = zeros(nData)
        for n in range(nData): distances[n] = sum((instance-self.data[n,0:class_idx])**2)
        #get indices of the nearest neighbors in ascending order
        indices = argsort(distances)
        #get class 
        if self.num_classes <= 1: return (numpy.float32(0))
        else:
            counts = zeros(self.num_classes)
            for i in range(self.k):
                counts[self.data[indices[i],class_idx]] += 1
            return (numpy.float32(counts.argmax()))
        

######---Test the fxns---#####        

ins = numpy.array([[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[1,0,1,1,1],[1,0,1,1,1],[1,0,1,1,1]]).astype(numpy.float32)
data1 = numpy.array([1,1,1,0]).astype(numpy.float32)
data2 = numpy.array([1,0,1,1]).astype(numpy.float32)
cKNN = KNN(3,2,4)
cKNN.trainClassifier(ins)
cl1 = cKNN.classifyInstance(data1)
cl2 = cKNN.classifyInstance(data2)
print cl1,cl2
class1 = kNN(3,ins,2,data1)
class2 = kNN(3,ins,2,data2)
print class1, class2
