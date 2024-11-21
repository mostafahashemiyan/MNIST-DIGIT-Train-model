

# MNIST Convolutional Neural Network (CNN) Model

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is built, compiled, trained, and evaluated using the provided dataset.

## Overview

- **Dataset**: MNIST (Modified National Institute of Standards and Technology) dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).
- **Model Architecture**: A CNN with the following layers:
  - Convolutional layer with 8 filters and a 3x3 kernel.
  - Max pooling layer with a 2x2 window.
  - Two fully connected (dense) hidden layers with 64 units and ReLU activation.
  - Output layer with 10 units and softmax activation for classification.
- **Optimizer**: Adam optimizer with default settings.
- **Loss Function**: Sparse categorical crossentropy.
- **Evaluation**: The model is evaluated using accuracy on the test set.

## Requirements

- Python 3.x
- TensorFlow (2.x)
- NumPy
- Matplotlib
- pandas

You can install the required dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib pandas
```

## Project Structure

- `train_model.py`: Contains the code to train the model.
- `evaluate_model.py`: Contains the code to evaluate the model's performance.
- `get_model.py`: Defines the architecture of the CNN.
- `scale_data.py`: Contains functions for normalizing and scaling the dataset.
- `main.py`: This is where you can run the entire workflow including loading data, preprocessing, building, training, and evaluating the model.
  
## How to Use

### 1. Load and Preprocess the Data

```python
from tensorflow.keras.datasets import mnist
import tensorflow as tf

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Scale the image data to range [0, 1]
train_images, test_images = scale_mnist_data(train_images, test_images)
```

### 2. Build the CNN Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the input shape (28x28 pixels, 1 channel)
input_shape = (28, 28, 1)

# Build the CNN model
model = get_model(input_shape)
```

### 3. Compile the Model

```python
# Compile the model with Adam optimizer, cross-entropy loss, and accuracy metric
compile_model(model)
```

### 4. Train the Model

```python
# Train the model for 5 epochs
history = train_model(model, train_images, train_labels)
```

### 5. Evaluate the Model

```python
# Evaluate the model on the test set
test_loss, test_accuracy = evaluate_model(model, test_images, test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
```

### 6. Plot Learning Curves (Optional)

```python
# Plot the learning curves (accuracy and loss over epochs)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()
```

## Functions

### `scale_mnist_data(train_images, test_images)`
This function scales the image pixel values to the range [0, 1].

### `get_model(input_shape)`
This function constructs and returns the CNN model, with the specified architecture:
- A convolutional layer with 8 filters and a 3x3 kernel.
- A max pooling layer.
- Two dense layers with 64 units each.
- An output layer with 10 units and softmax activation.

### `compile_model(model)`
This function compiles the model using the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the metric.

### `train_model(model, scaled_train_images, train_labels)`
This function trains the model on the training data for 5 epochs, and returns the training history for plotting learning curves.

### `evaluate_model(model, scaled_test_images, test_labels)`
This function evaluates the model on the test set and returns the test loss and accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This `README.md` file provides an overview of the project, explains the structure of the code, and gives clear instructions on how to use the provided functions.
