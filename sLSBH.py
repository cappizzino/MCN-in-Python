import numpy as np
    
class randomProject:
    def __init__(self, hash_size, inp_dimensions, s):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.s = s
        self.projections = np.random.normal(0,1,(self.hash_size, inp_dimensions))
        
    def generate_density_binary(self, inp_vector):
        bools = (np.dot(self.projections, inp_vector) > 0).astype('int')
        return bools

    def generate_sparsified_binary(self, inp_vector):
        value = np.dot(self.projections, inp_vector)
        n = int((self.s/100)*self.hash_size)
        if n == 0:
            n = 1
        elif self.s >= 100:
            n = self.hash_size
        largest = value[np.argsort(value)[-n:]]
        result = np.argpartition(value, n-1)
        smallest = value[result[:n]]
        z1 = np.zeros(self.hash_size).astype('int')
        z2 = np.zeros(self.hash_size).astype('int')

        for i in range(self.hash_size):
            for j in range(largest.shape[0]):
                if value[i] == largest[j]:
                    z1[i] = 1
                    break
                else:
                    z1[i] = 0
        for i in range(self.hash_size):
            for j in range(largest.shape[0]):
                if value[i] == smallest[j]:
                    z2[i] = 1
                    break
                else:
                    z2[i] = 0
        z = np.concatenate((z1, z2), axis=0)
        return z

#m = 1024
#example = np.random.normal(0,1,(64896,))
#print(example.shape)

#proj = randomProject(hash_size=m, inp_dimensions=example.shape[0], s=4)

#out = proj.generate_density_binary(example)
#print (out)

#out = proj.generate_sparsified_binary(example)
#print (np.dot(out,out))