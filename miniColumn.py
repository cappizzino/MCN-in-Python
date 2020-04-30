# Software release for the paper:
# "A Sequence-Based Neuronal Model for Mobile Robot Localization"
# Peer Neubert, Subutai Ahmad, and Peter Protzel, Proc. of German 
# Conference on Artificial Intelligence, 2018 
#
# Simplified Higher Order Sequence Memory from HTM.
# No learning in spatial pooler. Starting with an empty set of minicolumns, 
# each time a feedforward pattern is seen, that is not similar to any 
# existing minicolumn, create a new one witch responds to thinsg similar to
# this pattern.
#
import sys
import numpy as np
import random

class MCN:

    def __init__(self, name, params):
        self.name = name
        self.params = params

        # feedforward connections: FF(:,j) is the set of indices into the input
        # SDR of the j-th minicolumn. So the activation of all minicolumns can
        # be computed by sum(SDR(FF))
        self.FF = np.zeros((self.params.nInConPerCol,), dtype=int)
            
        # P(i,j) is the predicted-flag for the i-th cell in the j-th minicolumn
        self.P = np.array([], dtype=int)
        self.prevP = np.array([], dtype=int)

        # idx of all winning cells (in matrix of same size as P)
        self.winnerCells  = []
        self.prevWinnerCells  = []

        # idx of all active cells (in matrix of same size as P)
        self.activeCells  = np.array([], dtype=int)
        self.prevActiveCells  = np.array([], dtype=int)
        
        # flag if this minicolumn bursts
        self.burstedCol = np.array([], dtype=int) 
        self.prevBurstedCol = np.array([], dtype=int)

        # store for each column the number of iterations since last burst
        self.timeSinceLastBurst = np.array([], dtype=int) 

        self.nCols = 0

        self.new = 0

    def __str__(self):
        return self.name

    # reset some internal variables
    def prepareNewIteration(self):
        print('Prepare new iteration\n')

        self.prevWinnerCells = self.winnerCells
        self.prevActiveCells = self.activeCells
        self.prevP = self.P
        self.prevBurstedCol = self.burstedCol

        self.timeSinceLastBurst = self.timeSinceLastBurst + 1

        if self.nCols > 0:
            self.P = np.zeros((self.params.nCellPerCol,self.nCols), dtype=int)
            self.burstedCol = np.zeros((self.nCols,self.nCols), dtype=int)

    # InputConnections ... idx in a potential input SDR ,stored in obj.FF
    # Return index of the ne column 
    def createNewColumn(self, inputConnections):
        indices_t = np.concatenate(inputConnections.transpose())
        self.new = self.new + 1

        if self.nCols == 0:
            for i in range(self.params.nColsPerPattern):
                self.FF = np.c_[self.FF,
                    np.random.choice(indices_t, 
                        size = self.params.nInConPerCol, replace=True)]
            self.FF = np.delete(self.FF, 0, 1)

            self.P = np.zeros((self.params.nCellPerCol,
                self.params.nColsPerPattern), dtype=int)
            self.prevP = np.zeros((self.params.nCellPerCol,
                self.params.nColsPerPattern), dtype=int)

            self.burstedCol = np.zeros((1,self.params.nColsPerPattern), dtype=int)
            self.prevBurstedCol = np.zeros((1,self.params.nColsPerPattern), dtype=int)
            self.timeSinceLastBurst = np.zeros((1,self.params.nColsPerPattern), dtype=int)

            # datastructure that hold the connections for predictions. It is a
            # cell array of matrices.  predictionConnections{i,j} is a two row matrix
            # with each column [idx in P; permanence] <--- now: only first row yet
            self.cop = []
            self.predictionConnections = []

            for i in range(0, self.params.nCellPerCol, 1):
                linha = []
                for j in range(0, self.params.nColsPerPattern*100, 1):
                    linha.append([])
                self.predictionConnections.append(linha)
            #self.cop[:] = self.predictionConnections
            
        else:
            for i in range(self.params.nColsPerPattern):
                self.FF = np.c_[self.FF,
                    np.random.choice(indices_t, 
                        size = self.params.nInConPerCol, replace=True)]

                self.P = np.c_[self.P, np.zeros((self.P.shape[0],1), dtype=int)]
                self.prevP = np.c_[self.prevP, np.zeros((self.prevP.shape[0],1), dtype=int)]

                self.burstedCol = np.c_[self.burstedCol,
                    np.zeros((self.burstedCol.shape[0],1), dtype=int)]
                self.prevBurstedCol = np.c_[self.prevBurstedCol,
                   np.zeros((self.prevBurstedCol.shape[0],1), dtype=int)]
                self.timeSinceLastBurst = np.c_[self.timeSinceLastBurst,0]
            

        self.nCols = self.nCols + self.params.nColsPerPattern
        newColIdx = np.arange(start=self.nCols-self.params.nColsPerPattern, stop=self.nCols)   
        
        print('Created %d new column(s), nCols is %d\n' %(self.params.nColsPerPattern, self.nCols))
        
        return newColIdx

    # Compare inputSDR to all minicolumns to find active minicolumns
    # Search for predicted cells in active minicolumns and activate their predictions
    def compute(self, inputSDR):
        self.prepareNewIteration()
        
        # How similar is the input SDR to the pattern of the minicolumns?
        columnActivity = self.computeColumnActivations(inputSDR)

        # Test: If less than k_min minicolumns are active, missing minicolumns
        # are newly created.
        columnCompare = np.greater_equal(columnActivity, self.params.minColumnActivity)
        k = np.count_nonzero(columnCompare)
        print(k)

        if k >= self.params.kMin:
            activeCols = np.argwhere(columnActivity >= self.params.minColumnActivity)
            activeCols = np.concatenate(activeCols)
        else:
            sdrNonZeroIdx = np.argwhere(inputSDR == 1)
            activeCols = self.createNewColumn(sdrNonZeroIdx)
                
        # Is there an activity above threshold? If yes, activate the most
        # active column, otherwise create a new one and make this the active one.
        #maxActivity = np.amax(columnActivity)
        
        #if maxActivity > self.params.minColumnActivity:
        #    activeCols = np.argwhere(columnActivity > self.params.minColumnActivity)
        #    activeCols = np.concatenate(activeCols)
        #else:
        #    sdrNonZeroIdx = np.argwhere(inputSDR == 1)
        #    activeCols = self.createNewColumn(sdrNonZeroIdx)

        # for each active column:
        # - mark all predicted cells as winnerCells
        # - if there was no predicted cell, chose one and activate all predictions
        # - activate predictions of winnerCells
        self.activeCells = []
        self.winnerCells = []

        for activeCol in activeCols:
            predictedIdx = np.argwhere(self.prevP[:,activeCol]>0)
            #print('predictedIdx: {}'.format(predictedIdx))
 
            if predictedIdx.size == 0:
                # if there are no predicted: burst (predict from all cells
                # and choose one winner cell)
                winnerCell = self.burst(activeCol)
                index = np.ravel_multi_index([winnerCell,activeCol], self.P.shape, order='F')
                self.winnerCells.append(index[0])
            elif predictedIdx.size == 1:
                # if there is only one predicted cell, make this the winner cell
                winnerCell = np.asscalar(predictedIdx)
                self.activatePredictions(winnerCell, activeCol)
                index = np.ravel_multi_index([winnerCell,activeCol], self.P.shape, order='F')
                self.winnerCells.append(index)
            else:
                # if there are multiple predicted cells, make all winner cells  
                #print('Multiple cells were predicted(__):\n')
                for j in predictedIdx:
                    self.activatePredictions(np.asscalar(j),activeCol)
                    index = np.ravel_multi_index([np.asscalar(j),activeCol], self.P.shape, order='F')
                    self.winnerCells.append(index)
        
        # learn predictions
        self.learnPredictions()
        
        # also predict newly learned predictions
        for columnIdx in range(self.nCols):
            #print (self.burstedCol[0,columnIdx])
            if self.burstedCol[0,columnIdx] == 1:
                for i in range(self.P.shape[0]):
                    self.activatePredictions(i, columnIdx)

        print(self.winnerCells)

    # given the set of currently winning cells and the set of previously
    # winning cells, create prediction connection 
    def learnPredictions(self):

        for prevIdx in self.prevWinnerCells:
            for curIdx in self.winnerCells:
                [prevCellIdx, prevColIdx] = np.unravel_index(prevIdx, self.P.shape, order='F')
                [curCellIdx, curColIdx] = np.unravel_index(curIdx, self.P.shape, order='F')
                
                # check whether this minicolumn was bursting
                existingPredConFlag = self.checkExistingPredCon(prevColIdx, curIdx)
                if (not existingPredConFlag) or (random.random()<=self.params.probAdditionalCon):
                    self.predictionConnections[prevCellIdx][prevColIdx].append(curIdx)
                    self.predictionConnections[prevCellIdx][prevColIdx] = self.unique(
                        self.predictionConnections[prevCellIdx][prevColIdx])
                    #print('learned [col, cell]: [{}, {}] --> [
                    # {}, {}]\n'.format(prevColIdx, prevCellIdx, curColIdx, curCellIdx))
                #else:
                    #print('not learned sice there is already a connection\n')
        
    # Check if there already is an predicting connection from the previous
    # column to this active cell. This is used during bursting to prevent
    # learning multiple connections from one column to a single cell. In
    # this case, the new connection should go to a new cell of the current
    # collumn, to indicate the different context.
    def checkExistingPredCon(self, prevColIdx, curCellIdx):
        existingPredConFlag = 0

        for i in range(len(self.predictionConnections)):
            for j in range(len(self.predictionConnections[i][prevColIdx])):
                a = self.predictionConnections[i][prevColIdx]
                if a[j]==curCellIdx:
                    existingPredConFlag = existingPredConFlag + 1
        
        if existingPredConFlag > 0:
            existingPredConFlag = 1

        return existingPredConFlag
        
    # Activate the predictions of all cells and identify the cell with the
    # fewest forward predictions to be the winning cell. winnerCellIdx is the 
    # index of this cell in the minicolumn
    def burst(self, columnIdx):
        self.burstedCol[:,columnIdx] = 1
        self.timeSinceLastBurst[:,columnIdx] = 0
  
        for i in range(self.P.shape[0]):
            self.activatePredictions(i, columnIdx)
        
        # winnerCell: one of the cells with fewest existing forward predictions     
        nForwardPredictionsPerCell = []
        for i in range(self.params.nCellPerCol):
            if self.predictionConnections[i][columnIdx] == 0:
                nForwardPredictionsPerCell.append(0)
            else:
                nForwardPredictionsPerCell.append(len(self.predictionConnections[i][columnIdx]))

        # (slightly) inhibit winning cells from the last iteration a little bit
        for i in range(len(self.prevWinnerCells)):
            if self.prevWinnerCells != []:
                [cellIdx, colIdx] = np.unravel_index(self.prevWinnerCells[i], self.P.shape, order='F')
                if colIdx == columnIdx:
                    nForwardPredictionsPerCell[cellIdx] = nForwardPredictionsPerCell[cellIdx] + self.params.nCellPerCol
                                
        # [~, winnerCellIdx] = min(nForwardPredictionsPerCell);
        candidateIdx = []
        for i in range(len(nForwardPredictionsPerCell)):
            if nForwardPredictionsPerCell[i]==min(nForwardPredictionsPerCell):
                candidateIdx.append(i)
  
        winnerCellIdx = self.resolveTie(candidateIdx)

        return winnerCellIdx

    # randomly select one element to break a tie
    def resolveTie(obj, x):
        x_t = np.array(x)
        idx = np.random.choice(x_t, size = 1, replace=True)
        return idx

    # Increase number of predictions for all cells that are predicted from this cell
    def activatePredictions(self, cellIdx, colIdx):
        predIdx = self.predictionConnections[cellIdx][colIdx]
        if predIdx != []:
            [row, col] = np.unravel_index(predIdx, self.P.shape, order='F')
            self.P[row, col] = self.P[row, col] + 1

    # Evaluate all connections between the input space and columns
    def computeColumnActivations(self, inputSDR):
        columnActivity = inputSDR[self.FF].sum(axis=0)
        return columnActivity

    def unique(self,list1): 
        # function to get unique values 
        unique_list = [] 
        # traverse for all elements 
        for x in list1: 
            # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x) 

        return unique_list