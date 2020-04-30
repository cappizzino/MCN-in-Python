from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
    
class pairwiseComparison:
    def __init__ (self, mode="auto"):
        self.mode = mode
        self.maxValue = 0

    def evaluateSimilarity(self, inputFeatures):
       
        S = np.zeros((inputFeatures.shape[0],inputFeatures.shape[0]), dtype=float)
        self.maxValue = 0
        
        for idx1 in range(inputFeatures.shape[0]):
            v1 = inputFeatures[idx1,:]
            for idx2 in range(idx1,inputFeatures.shape[0]):
                v2 = inputFeatures[idx2,:]
                #dist = euclidean_distances(v1.reshape(1, -1),v2.reshape(1, -1))
                dist = cosine_similarity(v1.reshape(1, -1),v2.reshape(1, -1))
                #dist = hamming(v1,v2)
                if dist != 0:
                    S[idx1,idx2] = dist
                    if dist > self.maxValue:
                        self.maxValue = dist
                else:
                    S[idx1,idx2] = 0
        return S