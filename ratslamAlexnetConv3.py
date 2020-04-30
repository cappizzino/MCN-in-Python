import os, os.path
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import torchvision
from PIL import Image

from miniColumn import MCN
from sLSBH import randomProject
from comparison import pairwiseComparison

class Params:
    pass

def main():
    
    # Load dataset
    DIR = "./dataset/ratSlam"
    numberFiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

    # Load CNN
    original_model = models.alexnet(pretrained=True)
    class AlexNetConv3(nn.Module):
                def __init__(self):
                    super(AlexNetConv3, self).__init__()
                    self.features = nn.Sequential(
                        # stop at conv3
                        *list(original_model.features.children())[:7]
                    )
                def forward(self, x):
                    x = self.features(x)
                    return x
  
    model = AlexNetConv3()
    model.eval()

    # Create SDR and random project
    sdrSize = 1024
    sdr = np.zeros((sdrSize), dtype=int)
    proj = randomProject(hash_size=sdrSize, inp_dimensions=64896, s=2)

    loop = 0

    while False:

        # sample execution (requires torchvision)
        input_image = Image.open(DIR + "/frame%d.jpg" % loop)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0) 

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        output_conc = torch.flatten(output)
        output_conc = (output_conc.data).numpy()

        #out = proj.generate_density_binary(output_conc)
        out = proj.generate_sparsified_binary(output_conc)
       
        f=open("./data/sdr/SDR.txt",'ab')
        np.savetxt(f,[out], delimiter=',', fmt='%i')
        f.close()

        print(loop)
        if loop == 200: break
        #if loop == (numberFiles-1): break
        loop += 1
    
    
    #D = np.loadtxt("./data/sdr/SDR.txt", dtype='i', delimiter=',')
    D = np.loadtxt("./data/sdr/seq_multi_loop_noise05_al5.txt", dtype='i', delimiter=',')
    #D = np.loadtxt("./data/sdr/seq_multi_loop_noNoise.txt", dtype='i', delimiter=',')

    # simple HTM parameters
    params = Params()
    params.maxPredDepth = 0
    params.probAdditionalCon = 0.05
    params.nCellPerCol = 32
    params.nInConPerCol = int(round(np.count_nonzero(D) / D.shape[0])) 
    params.minColumnActivity = int(round(0.25*params.nInConPerCol))
    params.nColsPerPattern = 10
    params.kMin = 1

    print(params.nInConPerCol)
    print(params.minColumnActivity)

    print ('Simple HTM')
    htm = MCN('htm',params)
    # number of used images
    n = 200
    
    # run HTM
    t = time.time()

    outputSDR = []
    max_index = []

    for i in range (min(n,D.shape[0])):
        print('\n-------- ITERATION %d ---------' %i)
        # skip empty vectors
        if np.count_nonzero(D[i,:]) == 0:
            print('empty vector, skip\n')
            continue
        htm.compute(D[i,:])
        max_index.append(max(htm.winnerCells))
        outputSDR.append(htm.winnerCells)

    elapsed = time.time() - t
    print( "Elapsed time: %f seconds\n" %elapsed)

    # create output SDR matrix from HTM winner cell output
    M = np.zeros((len(outputSDR),max(max_index)+1), dtype=int)

    for i in range(len(outputSDR)):
        for j in range(len(outputSDR[i])):
            winner = outputSDR[i][j]
            M[i][winner] = 1

    pairwise = pairwiseComparison(mode="auto")
 
    plt.figure()

    plt.subplot(121)
    S1 = pairwise.evaluateSimilarity(D)
    plt.imshow(S1,vmin=0, vmax=pairwise.maxValue, cmap='gist_gray')
    plt.title('Input descriptors', y=1.02, fontsize=12)

    plt.subplot(122)
    S2 = pairwise.evaluateSimilarity(M)
    plt.imshow(S2,vmin=0, vmax=pairwise.maxValue, cmap='gist_gray')
    plt.title('Winner cell outputs', y=1.02, fontsize=12)

    #plt.figure()
    #plt.imshow(S, vmin=0, vmax=pairwise.maxValue, cmap='gist_gray', interpolation='bicubic')
    #plt.imshow(S2, vmin=0, vmax=pairwise.maxValue, cmap='gist_gray')
    #plt.title('Input descriptors', y=1.02, fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()