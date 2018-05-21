import numpy as np
import lvq_exceptions as exc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from scipy.spatial.distance import euclidean

class Lvq1:
    
    """ Implementation of Learning Vector Quantization (LVQ1) algorithm
    
    Parameters
    ----------
        neurons_per_class : list with the number of neurons (or codebooks vectors)
                per class the neural network model will use. 
        class_labels : values to name searched classes
        class_number : number of classes present in classification
    """
    
    def __init__(self, neurons_per_class, class_labels, epochs = 50, learning_rate = 0.01):
        
        self.class_number = len(neurons_per_class)
        
        if (self.class_number != len(class_labels)):
            print("Error")
        
        invalid_neuron_counts = sum(1 for neurons in neurons_per_class if neurons < 0 or type(neurons) is not int)
        
        if(invalid_neuron_counts > 0):
            raise exc.InvalidParameterException("\n[Exception]: Invalid neuron number.\n")
        elif(epochs < 0):
            raise exc.InvalidParameterException("\n[Exception]: Epochs number must be positive.\n")
        elif(learning_rate <= 0 or learning_rate >= 1):
            raise exc.InvalidParameterException("\n[Exception]: Learning rate beyond range (0, 1).\n")
        
        self.class_labels = class_labels
                
        self.neurons_per_class = neurons_per_class
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def train(self, P, T, k=3, p=2, plot_along = False):
        
        """ Function trains the model on given data
        
            Parameters
            ----------
            P : 
            T : 
            k :
            p :
            @TODO:
            plot_along : flag for ploting accuracy for every epoch
        """
        
        # neurons initialization
        self.__initialization(P, T, k, p)
        P, T = shuffle(P, T)        
        training_set = P
        training_labels = T
        
        # knn algorithm initialization for best fit seek
        get_nearest_neighbour = NearestNeighbors(n_neighbors=1)
        get_nearest_neighbour.fit(self.neuron_weights)
        
        learning_rate = self.learning_rate
        sample_number =  training_set.shape[0]
        # print("Learning rate is: ", learning_rate)
        lr_max_iterations = float(sample_number * self.epochs)
        
        for i in range(self.epochs):
            # print("Epoch number: ", i)
            for index, example in enumerate(training_set):
                # best fit neuron seeking
                # print(example.reshape(1,-1))                
                nn_index = get_nearest_neighbour.kneighbors(example.reshape(1,-1), return_distance = False) 
                nn_weights = self.neuron_weights[nn_index]
                nn_label = self.neuron_labels[nn_index]
                
                if(nn_label == training_labels[index]):
                    nn_weights += learning_rate * (example - nn_weights)
                else:
                    nn_weights -= learning_rate * (example - nn_weights)
                
                self.neuron_weights[nn_index] = nn_weights

                learning_rate = self.learning_rate - self.learning_rate * ((i * sample_number + index) / lr_max_iterations)

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
        
        """ Function returns labels of the nearest neighbour for given sample vector
            
            Parameters
            ----------
            sample : vector for which the best matching neuron is sought
        """
                
        nearest_neighbour = self.__get_nn(sample, self.neuron_weights)
        return self.neuron_labels[nearest_neighbour]
    
    def __get_nn(self, sample, vectors):
        
        """ Function returns index of the nearest neighbour by calculating the
            eucidean distance
            
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
        
    def __initialization(self, P, T, k, p):        
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
                        # print(P[ii].reshape(1,-1))
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
        
    def __plot_learning_accuracy(self, correctly_predicted, training_set_numerousity):
        
        """ Function for plotting accuracy rate for every epoch during learning
            process
            
            Parameters
            ----------
            correctly_predicted : number of correctly predicted output classes
                in given epoch
            training_set_numerousity : number of instances in training set
        """
