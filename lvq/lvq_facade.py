from lvq.lvq_1 import Lvq1
from lvq.lvq_2 import Lvq2
from lvq.lvq_3 import Lvq3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv

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
    
    def start_process(self, lr_iter, cb_num_iter): #, train_methods, train_periods_len):
        
        """ Function generates data for mesh plot following all given combinations
            to be checked out
            
            Parameters
            ----------
            @todo:  
            train_methods : list with sequence of lvq versions to be applied on
                training set
            train_periods_len : number of epochs to be applied to every method in
                train_methods sequence
        """
        
#         if len(train_methods) != len(train_periods_len):
#             raise exc.InvalidParameterException(err = "\n[Exception]: Length of method sequence must be equal to \n"
#                                                 "number of train periods.", num = str(len(train_methods)))
        
        try:
            lvq3_net = Lvq1([5,5], [0,1], epochs = 50)
            with open(self.result_file, 'w', newline = '') as csvfile:
                result_writer = csv.writer(csvfile)
                result_writer.writerow(cb_num_iter)
                for lr_level in range(len(lr_iter)):
                    # accuracy.append(lr_iter[lr_level])
                    accuracy = []
                    lvq3_net.learning_rate = lr_iter[lr_level]
                    for cn_num in range(len(cb_num_iter)):
                        lvq3_net.train(self.P_train, self.T_train)
                        accuracy.append(lvq3_net.test(self.P_test, self.T_test))
                        lvq3_net.neurons_per_class = cb_num_iter[cn_num]
                    result_writer.writerow(accuracy)
        except IOError as io_e:
            print("Something went wrong with file generation\n", io_e)
        except RuntimeError as rt_e:
            print("Something went wrong with loops\n", rt_e)
        else:
            csvfile.close()
            
    def plot_results(self):
        """ Function for plotting results of start_proces function stored in *.csv
            file            
        """
        
    def read_const_split(self, train_file, test_file):
        """ Function imports saved split to numpy array
            
            Parameters
            ----------
            train_file : csv file name with train part of data
            test_file : csv file name with test part of data
        """
        data_train_set = pd.read_csv(train_file)
        train_set = data_train_set.values
        data_test_set = pd.read_csv(test_file)
        test_set = data_test_set.values
        end_col = test_set.shape[1] - 1
        
        self.P_train = train_set[:,0:end_col-1]
        self.T_train = train_set[:,end_col]
        self.P_test = test_set[:,0:end_col-1]
        self.T_test = test_set[:,end_col]

    def generate_const_split(self, train_file, test_file, file_path):
        """ Function saves data split to csv file to allow conducting experiments on
            the same train and test data
            
            Parameters
            ----------
            train_file : csv file name with train part of data
            test_file : csv file name with test part of data
        """
        data_set = pd.read_csv(file_path)
        data_set_np = data_set.values
        
        train_set, test_set = train_test_split(data_set_np, test_size = 0.10)
        
        with open(train_file, 'w', newline = '') as csvfile:
            train_writer = csv.writer(csvfile)
            for i in range(0, train_set.shape[0]):
                train_writer.writerow(train_set[i,:])
            csvfile.close()
            
        with open(test_file, 'w', newline = '') as csvfile:
            test_writer = csv.writer(csvfile)
            for i in range(0, test_set.shape[0]):
                test_writer.writerow(test_set[i,:])
            csvfile.close()
    
