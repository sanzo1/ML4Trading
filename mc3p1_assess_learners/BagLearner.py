import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.verbose=verbose
        self.kwargs=kwargs
        self.bags=bags
        self.boost=boost
        self.learner=learner

    def author(self):
        return 'jshi88'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, Xtrain, Ytrain):
        """
        @summary: Add training data to learner
        @param Xtrain: X values of data to add
        @param Ytrain: the Y training values
        """
        self.learners = []
        for i in range(0, self.bags):
            self.learners.append(self.learner(**self.kwargs))
        for j in self.learners:
            sample = np.random.randint(0, high=Xtrain.shape[0], size=Xtrain.shape[0])
            dataX = Xtrain[sample]
            dataY = Ytrain[sample]
            j.addEvidence(dataX, dataY)
        return self.learners

    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param Xtest: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result = []
        for i in self.learners:
            result.append(i.query(Xtest))
        return np.mean(result, axis=0)