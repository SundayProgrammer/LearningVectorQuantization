import numpy as np
import lvq.errors as exc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

class Lvq1:
    
    """ Implementation of Learning Vector Quantization (LVQ1) algorithm
    
    Parameters
    ----------
        neurons_per_class : list with the number of neurons (or codebooks vectors)
                per class the neural network model will use. 
        class_labels : values to name searched classes
        class_number : number of classes present in classification
        epochs : number of epochs to train algorithm
        learning_rate : initial values of learning rate for algorithm
    """
    
    def __init__(self, neurons_per_class, class_labels, epochs = 50, learning_rate = 0.01):
        
        self.class_number = len(neurons_per_class)
        
        if (self.class_number != len(class_labels)):
            print("Error")
        
        invalid_neuron_counts = sum(1 for neurons in neurons_per_class if neurons < 0 or type(neurons) is not int)
        
        if(invalid_neuron_counts > 0):
            raise exc.InvalidParameterException(err = "\n[Exception]: Invalid neuron number.\n", num = str(parameter))
        elif(epochs < 0):
            raise exc.InvalidParameterException(err = "\n[Exception]: Epochs number must be positive.\n", num = str(epochs))
        elif(learning_rate <= 0 or learning_rate >= 1):
            raise exc.InvalidParameterException(err = "\n[Exception]: Learning rate beyond range (0, 1).\n", num = str(learning_rate))
        
        self.class_labels = class_labels
                
        self.neurons_per_class = neurons_per_class
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._epoch_accuracy = []
    
    def train(self, P, T, k=3, plot_along = False):
        
        """ Function trains the model on given data
        
            Parameters
            ----------
            P : data for prediction
            T : target values
            k : how many neighbors to consider
            plot_along : flag for plotting accuracy for every epoch after training
        """
        
        # neurons initialization
        self.__initialization(P, T, k)
        P, T = shuffle(P, T)        
        training_set = P
        training_labels = T
        
        # kNN algorithm initialization for seeking of best matching unit
        get_nearest_neighbour = NearestNeighbors(n_neighbors=1)
        get_nearest_neighbour.fit(self.neuron_weights)
        
        learning_rate = self.learning_rate
        sample_number =  training_set.shape[0]
        # print("Learning rate is: ", learning_rate)
        lr_max_iterations = 2.0 * float(sample_number * self.epochs)
        
        correctly_predicted_num = 0

        self._epoch_accuracy = []
        
        for i in range(self.epochs):
            # print("Epoch number: ", i)
            correctly_predicted_num = 0
            for index, example in enumerate(training_set):
                # best fit neuron seeking                
                nn_index = get_nearest_neighbour.kneighbors(example.reshape(1,-1), return_distance = False) 
                nn_weights = self.neuron_weights[nn_index]
                nn_label = self.neuron_labels[nn_index]
                
                if(nn_label == training_labels[index]):
                    nn_weights += learning_rate * (example - nn_weights)
                    correctly_predicted_num += 1
                else:
                    nn_weights -= learning_rate * (example - nn_weights)
                
                self.neuron_weights[nn_index] = nn_weights

                learning_rate = self.learning_rate - self.learning_rate * ((i * sample_number + index) / lr_max_iterations)
            self._epoch_accuracy.append(float(correctly_predicted_num / sample_number))
            
        if plot_along == True:
            self.__plot_learning_accuracy()

    def test(self, test_P, test_T):
        
        """ Function returns percentage of correctly predicted labels in test set
        
            Parameters
            ----------
            test_P : data for prediction
            test_T : target values
        
        """
        
        correct_labels = 0
        for index, sample in enumerate(test_P):
            predicted_label = self.__predict_label(sample)
            if (predicted_label  == test_T[index]):
                correct_labels += 1
        
        return correct_labels / float(test_T.shape[0])    
    
    def __predict_label(self, sample):
        
        """ Function returns labels of the nearest neighbor for given sample vector
            
            Parameters
            ----------
            sample : vector for which the best matching neuron is sought
        """
                
        nearest_neighbour = self.__get_nn(sample, self.neuron_weights)
        return self.neuron_labels[nearest_neighbour]
    
    def __get_nn(self, sample, vectors):
        
        """ Function returns index of the nearest neighbor by calculating the
            euclidean distance
            
            Parameters
            ----------
            sample : vector for which the best matching neuron is sought
            vectors : codebooks used in our model
        """
        
        min_distance = euclidean(sample, vectors[0])
        neuron_index = 0
        for i in range(1,vectors.shape[0]):
            dist = euclidean(sample, vectors[i])
            if min_distance > dist:
                neuron_index = i
                min_distance = dist
            
        return neuron_index
        
    def __initialization(self, P, T, k):        
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(P, T)
        
        neurons_number = [neuron for neuron in self.neurons_per_class]
        weights_uninitialized = sum([neuron for neuron in self.neurons_per_class])
        class_number = self.class_number
        neuron_weights = []
        neuron_labels = []
        row_number = P.shape[0]
        
        for ii in range(row_number):
            for jj in range(class_number):
                if (T[ii] == self.class_labels[jj]):
                    if (neurons_number[jj] > 0):
                        knn_label = knn_classifier.predict(P[ii].reshape(1,-1))
                        if (knn_label == T[ii]):
                            neuron_weights.append(P[ii])
                            neuron_labels.append(T[ii])
                            weights_uninitialized -= 1
                            neurons_number[jj] -= 1
                    break
        
        if (weights_uninitialized != 0):
            self.__random_initialization(P.shape[1])
        else:
            self.neuron_weights = np.array(neuron_weights)
            self.neuron_labels = np.array(neuron_labels)
            
    def __random_initialization(self, dimensions):
        neuron_number = [neuron for neuron in range(self.neurons_per_class)]
        neuron_weights = []
        neuron_labels = []
        
        for index, neuron_num in enumerate(neuron_number):
            for jj in range(neuron_num):
                neuron_weights.append(np.random.rand(1, dimensions)[0])
                neuron_labels.append(self.class_labels[index])
        
        self.neuron_weights = np.array(neuron_weights)
        self.neuron_labels = np.array(neuron_labels)
        
    def __plot_learning_accuracy(self):
        
        """ Function for plotting accuracy rate for every epoch during learning
            process
            
            Parameters
            ----------
            correctly_predicted : number of correctly predicted output classes
                in given epoch
            training_set_numerousity : number of instances in training set
        """
        x = [i for i in range(0,len(self._epoch_accuracy))]
        
        plt.plot(x, self._epoch_accuracy, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Interesting Graph\n')
        plt.legend()
        plt.show()
    
    @property    
    def epoch_accuracy(self):
        return np.copy(self._epoch_accuracy)