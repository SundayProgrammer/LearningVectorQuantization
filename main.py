import pandas as pd
from lvq.lvq_1 import Lvq1
import time
from sklearn.model_selection import KFold
lvq_1 = Lvq1([10, 10], [0, 1], epochs = 1000)

# reading data
dataset_filename = './data/diabetes.csv'
diab = pd.read_csv(dataset_filename)
diab_np = diab.values

P = diab_np[:,0:8]
T = diab_np[:,8]

# splitting data
kf = KFold(n_splits = 10)

i = 0
for train_set, test_set in kf.split(P):
    i += 1
    P_train, test_P = P[train_set], P[test_set]
    T_train, test_T = T[train_set], T[test_set]
    start = time.time()
    lvq_1.train(P_train, T_train)
    end = time.time()
    enlapsed_time = end - start
    print("---------- Test number ", i, " ----------")
    print("Learning process took: ", enlapsed_time)
    print("Accuracy is: ",lvq_1.test(test_P, test_T))

