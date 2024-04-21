---
layout: post
title: "Linear Regression"
date: 2024-04-21 11:45:00 -0800
categories: jekyll update
excerpt: "A Linear Regression Post"
---

Linear Regression: Linear regression is a statistical method that allows us to study relationships between two continuous (quantitative) variables:

One variable, denoted x, is regarded as the predictor, explanatory, or independent variable.
The other variable, denoted y, is regarded as the response, outcome, or dependent variable.
The goal of linear regression is to model the expected value of y given the value of x.

Model: The linear regression model is defined as y = Wx + b + ε, where W is the weight vector, x is the input vector, b is the bias (intercept), and ε is the error term.

Cost Function: The cost function for linear regression is the Mean Squared Error (MSE), which measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. The MSE is defined as J(W, b) = 1/n Σ(y_i - (Wx_i + b))^2, where n is the number of instances, y_i is the actual value, and (Wx_i + b) is the predicted value.

Gradient: The gradient of the cost function is a vector of the partial derivatives with respect to the weights and bias. The partial derivatives are defined as follows:

∂J/∂W = 2/n Σ -x_i(y_i - (Wx_i + b))
∂J/∂b = 2/n Σ -(y_i - (Wx_i + b))
Gradient Descent: Gradient descent is an optimization algorithm used to minimize the cost function. It iteratively adjusts the parameters (weights and bias) in the direction that most decreases the cost function. The parameters are updated using the rule θ = θ - η * ∇J(θ), where θ represents the parameters, η is the learning rate, and ∇J(θ) is the gradient of the cost function.

Intuition for the Cost Function: The cost function (MSE) measures the average squared difference between the actual and predicted values. It's a measure of the model's performance on the training set. The larger the MSE, the larger the difference between the actual and predicted values, which means the model is performing poorly. The goal of training is to find the parameters (weights and bias) that minimize the MSE.

In summary, linear regression is a simple yet powerful model for predicting a response variable given one or more explanatory variables. It's widely used in both statistics and machine learning.

Let's plot a simple cost function for a linear regression with one variable. We'll use a hypothetical model where the true parameters are W = 2 and b = 3, and we'll plot the cost function for a range of W values.

Here's the Python code using matplotlib and numpy:


```python
import numpy as np
import matplotlib.pyplot as plt

# True parameters
W_true = 2
b_true = 3

# Generate some data
np.random.seed(0)
X = np.random.rand(100, 1)
y = W_true * X + b_true + np.random.rand(100, 1)

# Cost function
def cost(W, X, y):
    return np.mean((y - W*X)**2)

# Values
W_values = np.linspace(-10, 10, 400)
cost_values = [cost(W, X, y) for W in W_values]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(W_values, cost_values)
plt.xlabel('Weight')
plt.ylabel('Cost')
plt.title('Cost function for different weight values')
plt.grid(True)
plt.show()
```


    
![png](linear_regression_files/linear_regression_2_0.png)
    


The code snippet above is plotting the cost function for a range of weight values. It's showing how the cost (mean squared error) changes as the weight parameter changes, assuming a fixed bias. This is a way to visualize how the cost function behaves, and it's useful for understanding why gradient descent can find the parameters that minimize the cost.

The code snippet below implements gradient descent to find the parameters that minimize the cost function. It starts with initial guesses for the weight and bias, then iteratively adjusts them to reduce the cost. The adjustments are made in the direction of steepest descent in the cost function, which is why it's called "gradient descent". The gradients (derivatives of the cost function with respect to the weight and bias) are computed with the lines dw = (1/len(x)) * np.sum((y_pred - y) * x) and db = (1/len(x)) * np.sum(y_pred - y).

In each iteration, it plots the data points and the current best-fit line, so you can see how the line changes as the parameters are updated. The learning rate and number of iterations are hyperparameters that control how quickly and how long the algorithm runs.


```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.randn(100)

# Initialize w and b
w = 0
b = 0

# Set the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 101

# Perform gradient descent to update w and b
for i in range(0, num_iterations, 20):
    # Calculate the predicted values
    y_pred = w * x + b
    
    # Calculate the gradients
    dw = (1/len(x)) * np.sum((y_pred - y) * x)
    db = (1/len(x)) * np.sum(y_pred - y)
    
    # Update w and b
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Plot the data points and the current line
    # Plot the data points
    plt.scatter(x, y, label='Data Points')

    # Plot the line
    plt.plot(x, w * x + b, color='red', label='Line: y = {} * x + {}'.format(w, b))

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.text(-1, 25, 'w = {:.3f}, b = {:.3f}'.format(w, b), fontsize=14, bbox=dict(facecolor='green', alpha=0.5))

    # Show the plot
    plt.show()

```


    
![png](linear_regression_files/linear_regression_4_0.png)
    



    
![png](linear_regression_files/linear_regression_4_1.png)
    



    
![png](linear_regression_files/linear_regression_4_2.png)
    



    
![png](linear_regression_files/linear_regression_4_3.png)
    



    
![png](linear_regression_files/linear_regression_4_4.png)
    



    
![png](linear_regression_files/linear_regression_4_5.png)
    

