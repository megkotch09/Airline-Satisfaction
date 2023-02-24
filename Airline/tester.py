from unittest import TestCase

from numpy.testing import assert_, assert_raises
import unittest
import exercise
import pandas as pd
from sklearn.metrics import accuracy_score

TEST_URL = 'https://raw.githubusercontent.com/karkir0003/DSGT-Bootcamp-Material/main/Udemy%20Material/Airline%20Satisfaction/train.csv'
class TestModelPerformance(TestCase):
    def test_trained_model(self):
        #READ IN TEST DATA
        print("starting")
        test_df = pd.read_csv(TEST_URL, index_col=0)
        print("training model")
        
        acc = exercise.train_model()

        #print(acc)
        
        #y_test = test_df["satisfaction"]
        #X_test = test_df.drop(["satisfaction"], axis=1)
        #pred = model.predict(X_test)
        #acc = accuracy_score(y_test, pred) #find accuracy of your model
        #print("Accuracy: {}".format(acc))
        #self.assertTrue(acc >= 0.85) #85% accuracy threshold

        print("OK")

if __name__ == '__main__':
    unittest.main()