import numpy as np
import random
from kNN import Classifier

R = 5
Pstag = 0.7

class SerialOC1(Classifier):
    def trainClassifier(self, training_data):
        # Number of features
        d = training_data.shape[1] - 1
        # Number of training samples
        n = training_data.shape[0]
        # Class array
        cat = training_data[:,-1]
        # Sample array
        sample = training_data[:,:-1]

        # Find the best axis parallel split
        impurityMin = 100000
        featureMin = 0
        splitMin = 0
        # For each feature
        for feature in range(d):
            # For each split
            sortedSample = sample[:,feature].copy()
            sortedSample.sort()
            for split in [(sortedSample[i]+sortedSample[i+1])/2.\
                          for i in range(n-1)]:
                # Compute impurity
                Tl = Tr = 0
                Lr = np.zeros(n)
                Ll = np.zeros(n)
                for index, point in enumerate(sample[:,feature]):
                    if point < split:
                        Tl += 1
                        Ll[cat[index]] += 1
                    else:
                        Tr += 1
                        Lr[cat[index]] += 1
                GiniL = 1 - sum([(l/float(Tl))**2 for l in Ll])
                GiniR = 1 - sum([(r/float(Tr))**2 for r in Lr])
                impurity = (Tl*GiniL + Tr*GiniR)/float(n)
                if impurity < impurityMin:
                    impurityMin = impurity
                    featureMin = feature
                    splitMin = split
        print "Best Parallel Split: ", featureMin, splitMin

        R = 0
        coeff = np.zeros(d + 1)
        Pmove = Pstag
        # Main Loop
        while R < 5:
            # Choose a random hyperplan
            if R == 0:
                coeff[featureMin] = 1
                coeff[d] = -splitMin
            else:
                pass

            # Perturb coefficients in sequence
            for m in range(d):
                # Compute U_j
                U = np.zeros(n)
                for j in range(n):
                    V = coeff[-1] + sum([coeff[i]*sample[j][i] for i in range(d)])
                    U[j] = (coeff[m]*sample[j][m] - V)/float(sample[j][m])
                Usorted = U.copy()
                Usorted.sort()
                # Find best univariate split for sorted U_j
                impurityMin = 100000
                splitMin = 0
                for split in [(Usorted[i]+Usorted[i+1])/2.\
                              for i in range(n-1)]:
                    # Compute impurity
                    Tl = Tr = 0
                    Lr = np.zeros(n)
                    Ll = np.zeros(n)
                    for index, point in enumerate(U):
                        if point < split:
                            Tl += 1
                            Ll[cat[index]] += 1
                        else:
                            Tr += 1
                            Lr[cat[index]] += 1
                    GiniL = 1 - sum([(l/float(Tl))**2 for l in Ll])
                    GiniR = 1 - sum([(r/float(Tr))**2 for r in Lr])
                    impurity = (Tl*GiniL + Tr*GiniR)/float(n)
                    if impurity < impurityMin:
                        impurityMin = impurity
                        splitMin = split
                H1 = coeff.copy()
                H1[m] = splitMin
                print coeff, H1

                # Compare Impurity
                Tl = Tr = 0
                Lr = np.zeros(n)
                Ll = np.zeros(n)
                for index in range(n):
#                    print "Point: %d, Value: %f" % (index, sum([coeff[i]*sample[index][i] for i in range(d)]) + coeff[d])
                    if sum([coeff[i]*sample[index][i] for i in range(d)]) + coeff[d] > 0:
                        Tl += 1
                        Ll[cat[index]] += 1
                    else:
                        Tr += 1
                        Lr[cat[index]] += 1
#                    print Tl, Tr
                GiniL = 1 - sum([(l/float(Tl))**2 for l in Ll])
                GiniR = 1 - sum([(r/float(Tr))**2 for r in Lr])
                impurityH = (Tl*GiniL + Tr*GiniR)/float(n)

                Tl = Tr = 0
                Lr = np.zeros(n)
                Ll = np.zeros(n)
                for index in range(n):
                    if H1[m] > U[index]:
                        Tl += 1
                        Ll[cat[index]] += 1
                    else:
                        Tr += 1
                        Lr[cat[index]] += 1
                GiniL = 1 - sum([(l/float(Tl))**2 for l in Ll])
                GiniR = 1 - sum([(r/float(Tr))**2 for r in Lr])
                impurityH1 = (Tl*GiniL + Tr*GiniR)/float(n)
                print "H %f, H1 %f" % (impurityH, impurityH1)
                if impurityH1 < impurityH:
                    coeff = H1
                    Pmove = Pstag
                elif impurityH1 ==  impurityH:
                    if random.random() < Pmove:
                        coeff = H1
                    Pmove -= 0.1*Pstag
                else:
                    break
                print m

            R += 1



    def classifyInstance(self,instance):
    #takes a 1D numpy float array and returns a class
        pass
    def isTrained(self):
    #boolean response of if the classifier is ready to be used
        pass

if __name__ == "__main__":
    sample = np.array([
                      [1.1, 1.1, 2],
                      [1, 2, 2],
                      [2.1, 1, 1],
                      [2, 2.1, 1],
                      [3, 1.9, 1],
                      [1, 4, 1],
                      ])
    classifier = SerialOC1()
    classifier.trainClassifier(sample)
