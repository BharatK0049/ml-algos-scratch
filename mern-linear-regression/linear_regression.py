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
        self.weight = None
        self.bias = None
        self.history = [] # Storing the parameter values along with the MSE calculated

    def fit(self, X, y):
        """
        Trains the model by getting the optimal weight and bias
        """
        n = len(X) # Number of samples
        
        self.weight = 0
        self.bias = 0
        self.history = [] # To reset history when called again, creating a new set of weights and biases
        
        tol = 1e-6 # Tolerance for early stopping

        # calculating gradient descent
        for i in range(self.epochs):
            # Predictions
            y_pred = self.weight * X + self.bias
            
            # Mean squared error
            mse = 1 / n * np.sum((y - y_pred)**2)

            # derivatives
            dE_dw = - 2 / n * np.sum(X * (y - y_pred))
            dE_db = - 2/n * np.sum((y - y_pred))

            # Updating parameters
            self.weight -= self.lr * dE_dw
            self.bias -= self.lr * dE_db

            self.history.append({
                "epoch" : i,
                "weight" : float(self.weight),
                "bias" : float(self.bias),
                "MSE" : float(mse)
            })

            # Early Stopping check
            if i > 0 and abs(self.history[-2]['MSE'] - mse) < tol:
                break
            
    def predict(self, X):
        """
        Returns the predicted values of target with weights and biases
        """
        return self.weight * X + self.bias


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
            "final_weights": float(model.weight),
            "final_bias": float(model.bias),
            "history": model.history
        }

        print(json.dumps(output_payload))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))