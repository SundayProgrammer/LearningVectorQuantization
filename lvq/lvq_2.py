from lvq.lvq_1 import Lvq1
import lvq.errors as exc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from array import array

class Lvq2(Lvq1):
    
    def __init__(self, neurons_per_class, class_labels, epochs = 50, learning_rate = 0.01, relative_window_width = 0.3):
        super().__init__(neurons_per_class, class_labels, epochs, learning_rate)
        self.window = (1 - relative_window_width)/(relative_window_width + 1)
    
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
        super().initialization(P, T, k)
        P, T = shuffle(P, T)        
        training_set = P
        training_labels = T
        
        # kNN algorithm initialization for seeking of best matching unit
        get_nearest_neighbour = NearestNeighbors(n_neighbors=2)
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
                # print(example.reshape(1,-1))                
                nn_dist, nn_index = get_nearest_neighbour.kneighbors(example.reshape(1,-1), return_distance = True)
                
                nn_dist = array(nn_dist)
                nn_index = array(nn_index)
                
                nn_weights = [self.neuron_weights[weight] for weight in nn_index]
                nn_label = [self.neuron_labels[codebook_label] for codebook_label in nn_index]
                
                print(nn_label)
                print(nn_weights)
                print(nn_dist)
                print(nn_index)
                               
                dist_0 = nn_dist[0]/nn_dist[1]
                dist_1 = nn_dist[1]/nn_dist[0]
                dist = min(dist_0, dist_1)
                
                if dist < self.window:
                    if(nn_label[:,0] == training_labels[index]):
                        nn_weights[0] += learning_rate * (example - nn_weights[0])
                        nn_weights[1] -= learning_rate * (example - nn_weights[1])
                        correctly_predicted_num += 1
                    else:
                        nn_weights[:,0] -= learning_rate * (example - nn_weights[:,0])
                        nn_weights[:,1] += learning_rate * (example - nn_weights[:,1])
                    
                    self.neuron_weights[nn_index[:,0]] = nn_weights[:,0]
                    self.neuron_weights[nn_index[:,1]] = nn_weights[:,1]

                # learning_rate = self.learning_rate - self.learning_rate * ((i * sample_number + index) / lr_max_iterations)
            self._epoch_accuracy.append(float(correctly_predicted_num / sample_number))
            
        if plot_along == True:
            super().plot_learning_accuracy()
            