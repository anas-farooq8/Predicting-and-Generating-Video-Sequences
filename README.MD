PredRnn Branch

# PredRNN: Spatio-Temporal Predictive Model for Sequence Prediction

This branch contains the implementation of PredRNN, a model designed for spatio-temporal video prediction. The model uses a custom variant of LSTM (Long Short-Term Memory) cells that can capture both spatial and temporal dependencies in video sequences. The model generates future frames based on a sequence of input frames.

This document provides a detailed explanation of the PredRNN architecture, how it captures spatio-temporal patterns, and step-by-step guidance on how to use the model for video frame prediction tasks.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
   - [Spatio-Temporal LSTM Cell](#spatio-temporal-lstm-cell)
   - [PredRNN](#predrnn)
3. [How PredRNN Captures Spatio-Temporal Patterns](#how-predrnn-captures-spatio-temporal-patterns)
4. [Training the Model](#training-the-model)
   - [Loss Function](#loss-function)
   - [Optimizer and Scheduler](#optimizer-and-scheduler)
   - [Early Stopping](#early-stopping)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
   - [Requirements](#requirements)
   - [Running the Model](#running-the-model)

## Overview

PredRNN is a deep learning model designed to predict future frames in a video given an input sequence of frames. The model combines spatial processing through convolutional operations and temporal processing through recurrent LSTM cells.

### Key Innovations

- **Spatio-Temporal LSTM Cells:** These cells process both spatial and temporal features of the input sequence.
- **Multiple Layers of LSTMs:** A multi-layer architecture that allows the model to learn complex spatio-temporal dependencies at different scales.

The model is particularly useful in tasks such as video frame prediction, action recognition, and video generation.

## Model Architecture

### Spatio-Temporal LSTM Cell

At the core of the PredRNN architecture is the Spatio-Temporal LSTM Cell. This custom LSTM cell is designed to handle both spatial and temporal dependencies simultaneously. The cell has the following components:

#### Spatial Convolutions

- The cell uses convolutional layers (with small filters) to extract spatial features from both the input frame and the hidden states.
- These convolutions capture spatial relationships in the image, such as object movement, texture, and structure.

#### Memory Cells

Each Spatio-Temporal LSTM cell maintains two types of memory:

- **Hidden State (h):** The traditional hidden state of an LSTM, which captures temporal information.
- **Memory (m):** An additional memory component to further capture long-term dependencies across time.

#### Gates

- **Input Gate (i):** Controls how much of the new information (spatial features) should be written to the memory.
- **Forget Gate (f):** Controls how much of the previous memory should be retained.
- **Output Gate (o):** Decides how much of the memory is exposed to the next layer.

These gates are computed based on both the input frame and the hidden state of the previous time step, allowing the cell to maintain both spatial and temporal information.

### PredRNN

PredRNN is constructed as a stack of Spatio-Temporal LSTM cells, where each layer processes the output of the previous layer.

#### Multiple Layers of LSTM Cells

- The model uses several layers of Spatio-Temporal LSTM cells stacked on top of each other, enabling it to learn hierarchical spatio-temporal representations.
- The first layer takes the input frames directly, and each subsequent layer processes the output of the previous layer.

#### Frame Prediction

- The final output is produced by a convolutional layer applied to the last layer's hidden state. This convolution generates the predicted frames for the future.

### Output

- The model predicts `output_frames` from an `input_sequence` of frames. The output frames are the model's prediction of what the future frames should look like.

## How PredRNN Captures Spatio-Temporal Patterns

PredRNN is designed to model both spatial and temporal dependencies:

### Spatial Dependencies

- Convolutional layers in the Spatio-Temporal LSTM cells capture local spatial features of each frame. These features could represent things like object locations, shapes, textures, etc.
- The hidden states of the LSTM cells also capture spatial context by encoding information from previous frames.

### Temporal Dependencies

- LSTM gates (input, forget, and output gates) learn temporal dependencies, i.e., how information evolves over time.
- The model processes the sequence of input frames, learning how each frame changes in relation to the previous frames. This allows the model to predict future frames by capturing the temporal progression of the video sequence.

### Spatio-Temporal Memory

- The introduction of an additional memory component (in addition to the hidden state) allows the model to remember long-term temporal dependencies and spatial patterns.
- By stacking multiple layers of Spatio-Temporal LSTM cells, PredRNN can capture both high-level temporal dynamics (like long-term motion) and low-level spatial features (like fine-grained details of each frame).

## Training the Model

### Loss Function

The model uses Mean Squared Error (MSE) loss to measure the pixel-wise error between the predicted frames and the ground truth target frames. Additionally, a perceptual loss is incorporated to improve the visual quality of generated frames. This perceptual loss is calculated by comparing the high-level features of the predicted and target frames using a pre-trained VGG16 model.

#### Total Loss Function

- **MSE Loss:** Measures the pixel-wise differences between predicted and target frames.
- **Perceptual Loss:** Helps the model focus on high-level features, such as texture and structure, rather than just pixel-level accuracy.

### Optimizer and Scheduler

- **Optimizer:** Adam optimizer is used with a learning rate of 2e-4.
- **Scheduler:** A learning rate scheduler (`ReduceLROnPlateau`) is used to reduce the learning rate if the validation loss plateaus.

### Early Stopping

To prevent overfitting, an early stopping mechanism is employed. The model stops training if the validation loss does not improve for a specified number of epochs (patience).

## Evaluation

During the evaluation phase, the model's performance is assessed using two metrics:

- **MSE (Mean Squared Error):** Measures the pixel-wise error between predicted and target frames.
- **SSIM (Structural Similarity Index):** Measures the structural similarity between the predicted and target frames. This helps to capture perceptual quality.

## Usage

### Requirements

Before running the code, make sure you have the following libraries installed:

```bash
pip install torch torchvision torchmetrics tqdm
```

### Running the Model

To use the PredRNN model for video frame prediction, you need to:

1. Prepare a dataset with video sequences.
2. Configure the model by adjusting hyperparameters like the number of hidden layers, the number of hidden units per layer, and the frame resolution.
3. Train the model on the prepared dataset.
4. Evaluate the model's performance using the test set.

```python
# Initialize model configuration
config = Config()

# Initialize and train the model
model = PredRNN(config)
model.train()

# Training loop
for epoch in range(num_epochs):
    # Your training code here

# Evaluate the model
model.eval()
# Evaluation code here
```

## Conclusion

PredRNN is a powerful model for predicting future video frames by capturing both spatial and temporal patterns in a video sequence. Its custom Spatio-Temporal LSTM cells enable it to learn complex patterns at different levels, making it suitable for a wide range of video generation and prediction tasks.
