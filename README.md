# Neural Mission Control

An interactive Neural Network learning tool built with NumPy and Streamlit. This project visualizes how a simple multi-layer perceptron (MLP) learns using forward and backward propagation.

## Features

### 1. Neuron Computation
- Visualizes activation functions like ReLU, Sigmoid, and Tanh.
- Shows the math behind weighted sums.

### 2. Network Topology
- A dynamic graph that shows the connections between layers.
- Updates based on the number of hidden layers and neurons you choose.

### 3. Decision Boundaries
- Shows how the network classifies 2D data (Moons, Circles, Blobs).
- Uses a heatmap to show where the model is confident vs uncertain.

### 4. Backpropagation
- Explains the chain rule and how gradients flow backwards.
- Allows for single-step training updates to see small changes.

### 5. Training Loop
- A full optimizer that trains the network live.
- Tracks loss history over epochs.

## Setup

### Requirements
- streamlit
- numpy
- plotly
- scikit-learn

### Installation
```bash
pip install streamlit numpy plotly scikit-learn
```

### How to Run
```bash
streamlit run app.py
```

## Files
- app.py: The main Streamlit dashboard.
- engine.py: The Neural Network logic (NumPy based).
- style.css: Custom CSS for the dark theme.
