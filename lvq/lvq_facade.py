from lvq.lvq_1 import Lvq1
from lvq.lvq_2 import Lvq2
from lvq.lvq_3 import Lvq3 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import csv
from symbol import except_clause

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
    
    def start_process(self, lr_iter, cb_num_iter, epoch_num): #, train_methods, train_periods_len):
        
        """ Function generates data for mesh plot following all given combinations
            to be checked out
            
            Parameters
            ----------
            @todo:  
            train_methods : list with sequence of lvq versions to be applied on
                training set
            train_periods_len : number of epochs to be applied to every method in
                train_methods sequence
            epoch_num : number of epoch
        """
        
#         if len(train_methods) != len(train_periods_len):
#             raise exc.InvalidParameterException(err = "\n[Exception]: Length of method sequence must be equal to \n"
#                                                 "number of train periods.", num = str(len(train_methods)))
#         if (len(train_methods) != len(epoch_num)):
#             raise exc.InvalidParameterException(err = "\n[Exception]: Length of method sequence must be equal to \n"
#                                                 "number of train periods.", num = str(len(train_methods)))
#         if (1 != len(epoch_num)):
#                 raise exc.InvalidParameterException(err = "\n[Exception]: Length of method sequence must be equal to \n"
#                                                 "number of train periods.", num = str(len(train_methods)))
#                 
        
        try:
            lvq3_net = Lvq1([10,10], [0,1], epochs = 50)
            with open(self.result_file, 'w', newline = '') as csvfile:
                result_writer = csv.writer(csvfile)
                result_writer.writerow(cb_num_iter)
                for lr_level in range(len(lr_iter)):
                    # accuracy.append(lr_iter[lr_level])
                    accuracy = []
                    lvq3_net.learning_rate = lr_iter[lr_level]
                    for cn_num in range(len(cb_num_iter)):
                        lvq3_net.neurons_per_class = cb_num_iter[cn_num]
                        lvq3_net.train(self.P_train, self.T_train)
                        accuracy.append(lvq3_net.test(self.P_test, self.T_test))
                    result_writer.writerow(accuracy)
        except IOError as io_e:
            print("Something went wrong with file generation\n", io_e)
        except RuntimeError as rt_e:
            print("Something went wrong with loops\n", rt_e)
        else:
            csvfile.close()
    
    def experiment_3(self, file_title, epochs_num):
        
        data_file = pd.read_csv(file_title).values
        n_groups = 3
        lr_35 = data_file[0,:]
        lr_60 = data_file[1,:]
         
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8
         
        r1 = plt.bar(index, lr_35, bar_width,
                         alpha=opacity,
                         color='b',
                         label='0.035')
         
        r2 = plt.bar(index + bar_width, lr_60, bar_width,
                         alpha=opacity,
                         color='g',
                         label='0.06')
        
        plt.xlabel('Codebooks number')
        plt.ylabel('Accuracy')
        plt.title('Learning Accuracy ' + epochs_num)
        plt.xticks(index + bar_width, ('10', '15', '20'))
        plt.legend()
        
        plt.tight_layout()
        plt.show()
            
    def experiment_4(self):
        
        try:
            lvq1_net = Lvq1([100,70], [0,1], epochs = 500)
            lvq2_net = Lvq2([10,10], [0,1], epochs = 10, initialize_codebooks = False)
            lvq3_net = Lvq3([10,10], [0,1], epochs = 20, initialize_codebooks = False)
            with open(self.result_file, 'w', newline = '') as csvfile:
                result_writer = csv.writer(csvfile)
                accuracy = []
                
                lvq1_net.train(self.P_train, self.T_train)
                neuron_weights, lr = lvq1_net.get_ancestry()
                accuracy = np.concatenate((accuracy,lvq1_net.epoch_accuracy),axis=0)
#                 result_writer.writerow(accuracy)
#                 accuracy = []
#                 lvq2_net.neuron_labels = lvq1_net.neuron_labels
#                 lvq2_net.set_ancestry(neuron_weights, 0.01)
#                 lvq2_net.train(self.P_train, self.T_train)
#                 neuron_weights, lr = lvq2_net.get_ancestry()
#                 accuracy = np.concatenate((accuracy,lvq2_net.epoch_accuracy),axis=0)
#                 result_writer.writerow(accuracy)
#                 accuracy = []
#                 lvq3_net.neuron_labels = lvq1_net.neuron_labels
#                 lvq3_net.set_ancestry(neuron_weights, 0.01)
#                 lvq3_net.train(self.P_train, self.T_train, epsilon = 0.1)
#                 accuracy = np.concatenate((accuracy,lvq3_net.epoch_accuracy),axis=0)
                
                print(lvq1_net.test(self.P_test, self.T_test))
                lvq1_net.plot_learning_accuracy()                        
                result_writer.writerow(accuracy)
        except IOError as io_e:
            print("Something went wrong with file generation\n", io_e)
        except RuntimeError as rt_e:
            print("Something went wrong with loops\n", rt_e)
        else:
            csvfile.close()
            
    def plot_results(self):
        """ Function for plotting results of start_proces function stored in csv
            file            
        """
    
    def normalize_data(self,norm_range, norm_file, file_name, without_last = True):
        """ Function normalize given csv file i specified range
        
            Parameters
            ----------
            norm_range : normalization range
            norm_file : file to be normalized 
            file_name : new file name for normalized data
            without_last : do not normalize last column in given data  
        """
        try:
            norm_set = pd.read_csv(norm_file).values
            end_col = norm_set.shape[1] - 1
            if without_last == True:
                P = norm_set[:,0:end_col - 1]
                T = norm_set[:,end_col]
                T = T.reshape(T.shape[0],1)
            else:
                P = norm_set
            minmax_norm = MinMaxScaler(feature_range = (0,norm_range), copy = False)
            minmax_norm.fit(P)
            minmax_norm.transform(P)
            print(P.shape)
            print(T.shape)
            if without_last == True:
                norm_set = np.concatenate((P,T),axis=1)
            else:
                norm_set = P
            with open(file_name, 'w', newline = '') as csvfile:
                norm_writer = csv.writer(csvfile)
                for row in range(0,norm_set.shape[0]):
                    norm_writer.writerow(norm_set[row,:])
                csvfile.close()                
        except IOError as io_e:
            print("Something went wrong with file IO\n", io_e)
        
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
    
