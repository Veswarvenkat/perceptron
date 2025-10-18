# Perceptron and MLP Implementations for Logic Gates

This Jupyter notebook explores the implementation of the Perceptron algorithm and a simple Multilayer Perceptron (MLP) using both NumPy and TensorFlow. It demonstrates how these models learn basic logic gates like AND and XOR.

## Project Overview

The notebook contains four main sections:

1.  **Single Perceptron Iteration (NumPy):**
    * Demonstrates one pass (iteration) of the Perceptron learning algorithm using NumPy.
    * Uses the AND gate dataset as an example.
    * Shows the initial weight and bias update based on the first encountered misclassification.

2.  **Perceptron Convergence (NumPy):**
    * Implements the full Perceptron training loop using NumPy.
    * Trains the model on the AND gate dataset until it converges (makes no errors in an epoch).
    * Shows the number of epochs required for convergence and the final learned weights and bias.
    * Tests the converged model on all AND gate inputs.

3.  **Perceptron using TensorFlow:**
    * Re-implements the Perceptron training loop for the AND gate using TensorFlow.
    * Utilizes `tf.Variable` for weights and bias and TensorFlow operations for calculations and updates.
    * Demonstrates convergence and tests the final model.

4.  **MLP for XOR (TensorFlow Keras):**
    * Highlights the limitation of a single Perceptron (it cannot solve non-linearly separable problems like XOR).
    * Builds a simple Multilayer Perceptron (MLP) using `tensorflow.keras.Sequential`.
    * The MLP has one hidden layer with `relu` activation and an output layer with `sigmoid` activation.
    * Trains the MLP to learn the XOR function.
    * Evaluates the trained MLP's accuracy and shows its predictions.
    * Extracts and displays the learned weights and biases from the Keras model.

## Concepts Demonstrated

* **Perceptron Algorithm:** The fundamental learning rule for updating weights and bias based on misclassifications.
* **Activation Function:** Use of a step function for the Perceptron and `relu`/`sigmoid` for the MLP.
* **Linear Separability:** Implicit demonstration of why a single Perceptron works for AND but not for XOR.
* **Convergence:** Training a model until it perfectly classifies the training data.
* **NumPy Implementation:** Basic array manipulation for model building.
* **TensorFlow Implementation:** Using TensorFlow tensors, variables, and operations for the same algorithm.
* **Keras MLP:** Building, compiling, training, and evaluating a simple neural network using the Keras API for a non-linear problem.

---

## Requirements

* `numpy`
* `tensorflow`
* `matplotlib` (Used in the XOR part for potential plotting, although not explicitly used for plotting in the provided code)

Install them using pip:
```bash
pip install numpy tensorflow matplotlib
