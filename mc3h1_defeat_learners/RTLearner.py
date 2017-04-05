"""
A simple wrapper for RTLearner.
"""

import numpy as np
from random import randint

class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'jshi88'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, Xtrain, Ytrain):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        def build_tree(x, y):
            """
            @summary: Recursively builds the decision tree
            @param Xtrain: X values of data to add
            @param Ytrain: the Y training values
            """
            if x.shape[0] <= self.leaf_size or len(set(y))==1:
                self.tree = np.array([[-1,np.mean(y), np.nan, np.nan]])
                return self.tree

            # randomly select    two samples of data in order to take their mean and get the split value for each node
            split_feature = randint(0,x.shape[1] - 1)
            num_data = int(x.shape[0])
            first_datapoint = randint(0, num_data - 1)
            second_datapoint = randint(0, num_data -1)
            split_value = np.mean([float(x[first_datapoint][split_feature]), float(x[second_datapoint][split_feature])])

            left_indices = []
            for i in range(x.shape[0]):
                if (x[i][split_feature] <= split_value):
                    left_indices.append(i)

            right_indices = []
            for i in range(x.shape[0]):
                if (x[i][split_feature] > split_value):
                    right_indices.append(i)

            #checks to see if there's a bad split where everything is on the right or left side of the tree. If so,
            #resplit until we get a better result
            while len(left_indices) < 1 or len(right_indices) < 1:
                split_feature = randint(0, x.shape[1] - 1)
                num_data = int(x.shape[0])
                first_datapoint = randint(0, num_data - 1)
                second_datapoint = randint(0, num_data - 1)
                split_value = np.mean(
                    [float(x[first_datapoint][split_feature]), float(x[second_datapoint][split_feature])])

                left_indices = []
                for i in range(x.shape[0]):
                    if (x[i][split_feature] <= split_value):
                        left_indices.append(i)

                right_indices = []
                for i in range(x.shape[0]):
                    if (x[i][split_feature] > split_value):
                        right_indices.append(i)


            left_tree_x = np.array([x[i] for i in left_indices])
            left_tree_y = np.array([y[i] for i in left_indices])
            right_tree_x = np.array([x[i] for i in right_indices])
            right_tree_y = np.array([y[i] for i in right_indices])
            left_tree = build_tree(left_tree_x, left_tree_y)
            right_tree = build_tree(right_tree_x, right_tree_y)
            root = [split_feature, split_value, 1, len(left_tree)+1]
            return np.vstack((root, left_tree, right_tree))

        self.tree = build_tree(Xtrain, Ytrain)

    def iterate_tree(self, i, row=0):
        """
        @summary: Traverse through the tree
        @param i: the specific instance in the test set
        """
        feature = int(self.tree[row][0])
        split_value = self.tree[row][1]
        if feature == -1:
            return self.tree[row][1]
        elif i[feature] <= split_value:
            return self.iterate_tree(i, row + int(self.tree[row][2]))
        else:
            return self.iterate_tree(i, row + int(self.tree[row][3]))

    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param Xtest: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result = []
        for i in Xtest:
            result.append(self.iterate_tree(i))
        return np.array(result)

if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
