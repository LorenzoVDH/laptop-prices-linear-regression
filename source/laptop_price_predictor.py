import numpy as np 
import pandas as pd  
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable 

@dataclass
class LaptopPricePredictor: 
    _learning_rate: float = 0.001  # reduced learning rate to prevent overflow
    _weights: np.ndarray = field(default_factory=lambda: np.random.rand(4).astype(np.float32))  # modified to use float32
    _cost_log: list = field(default_factory=list)

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, value):
        self._weights = value 

    @property
    def cost_log(self):
        return self._cost_log
    
    @property
    def learning_rate(self):
        return self._learning_rate

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, epochs: int = 1000) -> None: 
        '''
            This function trains the model using the provided training data.
            It updates weights and decays the learning rate.
            Early stopping is applied if the cost does not improve by more than a threshold.
        '''
        early_stop_threshold = 1e-8
        for epoch in range(epochs): 
            for i in range(len(X_train)):
                # Make a prediction for the current input data (i)
                processor = X_train['Processor'].iloc[i]
                gpu = X_train['GPU'].iloc[i]
                ram = X_train['RAM (GB)'].iloc[i]
                y_pred: float = self.predict(processor, gpu, ram)
                
                # Compare that prediction to the true output data and calculate the error 
                y_true = y_train.iloc[i]
                error_value = self.compute_error(y_true, y_pred)

                # Calculate the gradient to determine the direction to move the weights 
                gradient = self.calculate_gradient(processor, gpu, ram, error_value)

                # Update the weights based on the gradient and the learning rate
                self.weights = self.update_weights(self.weights, gradient, self.learning_rate)
                
            # At the end of the epoch, evaluate the model using the training data (over the whole dataset)
            cost = self.evaluate(X_train, y_train)
            # and calculate the cost
            self.cost_log.append(cost)

            print(f"Epoch {epoch}/{epochs}, Cost: {cost}")
            
            # Early stopping: if improvement is less than threshold, stop training
            if epoch > 0 and abs(cost - self.cost_log[-2]) < early_stop_threshold:
                print(f"Early stopping at epoch {epoch}, improvement < {early_stop_threshold}")
                break
            
            # Decay the learning rate by 1% after each epoch if it is greater than 0.01, with a minimum threshold
            if self._learning_rate > 0.000001:
                self._learning_rate = self.learning_rate * 0.9 #max(self._learning_rate * 0.5, 1e-6)

            # Shuffle training data after each epoch
            X_train, y_train = self.shuffle_data(X_train, y_train)

    def predict(self, processor: float, gpu: float, ram: float) -> float: 
        '''
            This function predicts the output for the given input data. 
            It uses the weights and bias to come to a prediction.
            Not using the dot product to be more explicit for this demonstration.  

            weight 0 = bias
            weight 1 = Processor
            weight 2 = GPU
            weight 3 = RAM (GB)
        '''
        processor = np.float32(processor)  # cast input
        gpu = np.float32(gpu)              # cast input
        ram = np.float32(ram)              # cast input
        prediction = self.weights[0] + self.weights[1] * processor + self.weights[2] * gpu + self.weights[3] * ram
        return prediction  

    # Added new method to support batch predictions using vectorized calculations.
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        processor = X['Processor'].astype(np.float32).values  # cast column
        gpu = X['GPU'].astype(np.float32).values              # cast column
        ram = X['RAM (GB)'].astype(np.float32).values           # cast column
        return self.weights[0] + self.weights[1] * processor + self.weights[2] * gpu + self.weights[3] * ram

    def compute_error(self, y_true: float, y_pred: float) -> float: 
        '''
            This function calculates the error between the true output and the predicted output. 
            It will not be used for the cost function, but for setting the weights during training.

        '''
        return y_true - y_pred
    
    def calculate_gradient(self, processor: float, gpu: float, ram: float, error: float) -> np.ndarray:
        '''
            This function will multiply each input (processor, gpu, ram) by the error and return a new list 
            with the error multiplied by each input, which is the gradient. It will multiply each 
            input by the error to determine the direction to move the weights. The function will 
            return a 1D Array of the gradients for the bias and weights. 
        '''

        gradient_bias = error # since the bias is multiplied by 1, the gradient is just the error
        gradient_processor = error * processor
        gradient_gpu = error * gpu
        gradient_ram = error * ram
        
        # Create a new array by concatenating the bias and features gradients
        gradient = np.array([gradient_bias, gradient_processor, gradient_gpu, gradient_ram])
        return gradient

    def update_weights(self, weights_to_update: np.ndarray, gradient: np.ndarray, learning_rate: float) -> np.ndarray: 
        '''
            Updates the weights using a clipped gradient.
            Note: The update now adds the gradient to move in the correct direction.
        '''
        # Clip the gradient values
        clipped_gradient = np.clip(gradient, -1.0, 1.0)
        # Add the weighted gradient instead of subtracting it
        new_weights = (weights_to_update + learning_rate * clipped_gradient).astype(np.float32)
        return new_weights
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                 cost_function: Optional[Callable[[pd.Series, np.ndarray], float]] = None) -> float: 
        '''
            This function will evaluate the model at the end of an epoch using the test data. 
            It will return the cost of the epoch using the provided cost function or the default MSE cost function. 
            The cost function is a callable that should take in the true output and the predicted output as y_true and y_pred. 
        '''
        if cost_function is None:
            cost_function = self.mean_square_error
        
        # Use the vectorized predict_batch method instead of predict.
        y_pred = self.predict_batch(X_test)
        cost = cost_function(y_test, y_pred)
        return cost
    
    def mean_square_error(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        '''
            This function calculates the Mean Square Error (average) between the true output and the predicted output.
            This is the cost function that we will use to evaluate the model after each epoch. 
        '''
        num_of_samples = len(y_true) #get the number of samples in the dataset
        result = np.sum((y_true - y_pred)**2) / num_of_samples #squared to make it positive and punish larger values more
        return result

    def shuffle_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        '''
            This function shuffles the training data so that the model does not learn 
            the order of the data. It creates a new random order for the row indices,
            then rearranges X_train and y_train accordingly.
        '''
        # Create a random ordering of indices for the data rows
        random_indices = np.random.permutation(len(X_train))
        # Fix indexing for y_train using square brackets instead of parentheses.
        return X_train.iloc[random_indices].reset_index(drop=True), y_train.iloc[random_indices].reset_index(drop=True)
