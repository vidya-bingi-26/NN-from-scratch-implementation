# **Neural Network Implementation from Scratch**

## **Project Overview**
This project demonstrates the implementation of a simple feedforward neural network from scratch in Python without using any in-built deep learning libraries. The network is trained to classify binary data from the Iris dataset (Setosa vs. Versicolor) using gradient descent.

## **Key Features**
- Built a custom neural network from scratch, including forward propagation, backpropagation, and gradient descent optimization.
- Implemented all essential components of a neural network:
  - Weight and bias initialization
  - Sigmoid activation function and its derivative
  - Loss function (Mean Squared Error)
  - Training through backpropagation
  - Prediction for binary classification
- Preprocessed the Iris dataset using normalization for faster convergence.

## **Dataset**
The project uses the Iris dataset, focusing on:
- **Classes:** Setosa (0) and Versicolor (1) (first 100 samples).
- **Features:** Sepal length, sepal width, petal length, and petal width.

## **Technologies Used**
- **Python Libraries:**
  - NumPy: For matrix operations and computations.
  - Matplotlib: For visualizing the training loss.
  - scikit-learn: For dataset loading, splitting, and preprocessing.
- **Neural Network Concepts:**
  - Feedforward computation
  - Sigmoid activation
  - Gradient descent optimization

## **Project Structure**
- **Code File:** The complete Python implementation of the neural network.
- **Visualizations:** Includes a loss plot to track the model's convergence over epochs.
- **Dataset Link:** Iris dataset is loaded directly via `scikit-learn`.

## **How to Run the Code**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### **2. Install Required Dependencies**
```bash
pip install numpy matplotlib scikit-learn
```

### **3. Run the Code**
Run the Python script or Jupyter Notebook:
```bash
python neural_network_from_scratch.py
```
_OR_
Open the `.ipynb` file in Jupyter Notebook and execute all cells.

## **Results**
- Achieved high accuracy on the binary classification task.
- Visualized the training loss over 10,000 epochs to ensure convergence.

### **Sample Output**
```
Input: [0.5 -1.2 0.8 1.1] -> Predicted Output: 1, Actual: 1
Input: [-0.7 0.9 -0.6 -1.0] -> Predicted Output: 0, Actual: 0
...
Accuracy: 95.00%
```

## **Screenshots**
![Loss Curve](plot)

## **Future Improvements**
- Extend to multi-class classification for all three Iris classes.
- Experiment with additional activation functions and optimizers.
- Visualize decision boundaries for better interpretability.

---

### **Author**
Vidya Bingi,
vidyabingi26@gmail.com
