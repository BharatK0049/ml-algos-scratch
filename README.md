# Linear Regression from Scratch

This is a manual implementation of Linear Regression using **Gradient Descent**.
It does not use Scikit-Learn. It is built using purely **NumPy** to demonstrate the underlying mathematics of optimization.

## The Math
The model minimizes the Mean Squared Error (MSE) cost function:
$$J = \frac{1}{n} \sum (y - \hat{y})^2$$

It optimizes parameters using the update rule:
$$w = w - \alpha \cdot \frac{\partial J}{\partial w}$$

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python linear_regression.py`