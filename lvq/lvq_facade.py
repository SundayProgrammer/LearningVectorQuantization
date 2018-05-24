from lvq.lvq_1 import Lvq1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv

"""
    @todo:
    - must 
        split data on datasets with equal percentage of positive and negative
        shuffle data on input (once per 10 laps)
        append produced data to *.csv file
        read produced *.csv file and display mesh
    - should
        add configuration parameters for learning process
"""

class Lvq_facade:
    
    """ Facade for implemented Learning Vector Quantization algorithms 
    
        Parameters
        ----------
        _file_path : path to dataset with data to work on        
    """
    
    def __init__(self, file_path, result_file):
        self._file_path = file_path
        self.result_file = result_file
        
    @property
    def file_path(self):
        return self._file_path
    
    def start_process(self):
        
        """ Function generates data for mesh plot following all given combinations
            to be checked out            
        """
        
        diab = pd.read_csv(self._file_path)
        diab_np = diab.values
        
        P = diab_np[:,0:8]
        T = diab_np[:,8]
        
        P_train, test_P, T_train, test_T = train_test_split(P, T, test_size=0.10)
        
        learning_rate_iter = np.arange(0.01,0.1,0.005).tolist()
        codebooks_number_iter = np.arange(5,105,5).tolist()
        
        print(len(learning_rate_iter))
        print(len(codebooks_number_iter))
        
        try:
            lvq1_net = Lvq1([5,5], [0,1], epochs = 50)
            with open(self.result_file, 'w', newline = '') as csvfile:
                result_writer = csv.writer(csvfile)
                result_writer.writerow(codebooks_number_iter)
                for lr_level in range(len(learning_rate_iter)):
                    # accuracy.append(learning_rate_iter[lr_level])
                    accuracy = []
                    lvq1_net.learning_rate = learning_rate_iter[lr_level]
                    for cn_num in range(len(codebooks_number_iter)):
                        lvq1_net.train(P_train, T_train)
                        accuracy.append(lvq1_net.test(test_P, test_T))
                        lvq1_net.neurons_per_class = codebooks_number_iter[cn_num]
                    result_writer.writerow(accuracy)
        except IOError as io_e:
            print("Something went wrong with file generation\n", io_e)
        except RuntimeError as rt_e:
            print("Something went wrong with loops\n", rt_e)
        else:
            csvfile.close()



