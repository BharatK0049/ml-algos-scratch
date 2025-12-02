# Main code implementing Linear Regression Algorithm
import json
import numpy as np
import sys

class LinearRegression:
    def __init__(self, learning_rate = 0.001, epochs = 1000):
        """
        Initialize the Linear Regression Model

        parameters - 
        learning_rate: Control the step size of Gradient Descent
        epochs: The number of iterations on the data
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.history = [] # Storing the parameter values along with the MSE calculated

    def fit(self, X, y):
        """
        Trains the model by getting the optimal weights and bias

        parameters -
        X: shape (n_samples,) or (n_samples, n_features)
        y: shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1) # Makes it 1D

        # Ensuring X is in 2D - (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1) # Setting columns as 1 and automatically setting rows

        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = [] # To reset history when called again, creating a new set of weights and biases
        
        tol = 1e-6 # Tolerance for early stopping

        # calculating gradient descent
        for i in range(self.epochs):
            # Predictions
            y_pred = X.dot(self.weights) + self.bias
            
            # Mean squared error
            mse = 1 / n_samples * np.sum((y - y_pred)**2)

            # derivatives
            dE_dw = - 2 / n_samples * X.T.dot(y - y_pred)
            dE_db = - 2 / n_samples * np.sum((y - y_pred))

            # Updating parameters
            self.weights -= self.lr * dE_dw
            self.bias -= self.lr * dE_db

            self.history.append({
                "epoch" : i,
                "weights" : self.weights.astype(float).tolist(),
                "bias" : float(self.bias),
                "MSE" : float(mse)
            })

            # Early Stopping check
            if i > 0 and abs(self.history[-2]['MSE'] - mse) < tol:
                break
            
    def predict(self, X):
        """
        Returns the predicted values of target with weightss and biases
        X: shape (n_samples,) or (n_samples, n_features)
        """
        X = np.array(X)
        # Ensuring X is in 2D - (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1) # Setting columns as 1 and automatically setting rows

        return X.dot(self.weights) + self.bias


# bridge interface

if __name__ == "__main__":
    try:
        # Read data from Node
        input_data = json.loads(sys.argv[1])

        X = np.array(input_data['x'])
        y = np.array(input_data['y'])
        lr = float(np.array(input_data['lr']))
        epochs = int(input_data['epochs'])

        # Train model
        model = LinearRegression(lr, epochs)
        model.fit(X, y)

        output_payload = {
            "status": "success",
            "final_weights": model.weights.astype(float).tolist(),
            "final_bias": float(model.bias),
            "history": model.history
        }

        print(json.dumps(output_payload))
    
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))