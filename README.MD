# ConvLSTM Model for Sequence Prediction

This branch contains the implementation of a ConvLSTM (Convolutional Long Short-Term Memory) model for predicting future frames in a sequence of images. ConvLSTM is a variant of LSTM (Long Short-Term Memory) networks, designed specifically to handle spatiotemporal data, where both spatial (2D) and temporal (sequence) information is crucial.

The ConvLSTM model is built using a series of ConvLSTM cells, which apply convolution operations to both the input and the hidden states, making it suitable for video prediction, weather forecasting, and other time-series tasks involving images.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
   - [ConvLSTMModel](#convlstmmodel)
   - [Detailed Explanation of the Architecture](#detailed-explanation-of-the-architecture)
   - [ConvLSTM Cell](#convlstm-cell)
   - [Final Convolution Layer](#final-convolution-layer)
2. [ConvLSTM in Sequence Prediction](#convlstm-in-sequence-prediction)
   - [Structure of ConvLSTM for Sequence Prediction](#structure-of-convlstm-for-sequence-prediction)
   - [How ConvLSTM is Used in Sequence Prediction](#how-convlstm-is-used-in-sequence-prediction)
   - [Example](#example)
3. [Training and Evaluation](#training-and-evaluation)
   - [Loss Functions](#loss-functions)
   - [Learning Rate Scheduler](#learning-rate-scheduler)
   - [Early Stopping](#early-stopping)
4. [Dataset](#dataset)
   - [Input Shape](#input-shape)
   - [Output Shape](#output-shape)
5. [Training](#training)
   - [Key Components](#key-components)
   - [Training Loop](#training-loop)
6. [Evaluation](#evaluation)

---

## Model Architecture

The core of the model consists of a multi-layer ConvLSTM network followed by a final convolutional layer to generate the output frames. Below is the breakdown of the architecture:

### ConvLSTMModel

This is the main model that consists of:

- **ConvLSTM Layers**: A sequence of ConvLSTM cells stacked together, where each layer processes the input sequence and passes the hidden and cell states to the next layer.
- **Final Convolution Layer**: This layer reduces the output of the last ConvLSTM layer to the desired number of output channels (e.g., 3 channels for RGB images).

```python
ConvLSTMModel(
  (conv_lstm): ConvLSTM(
    (cell_list): ModuleList(
      (0): ConvLSTMCell(
        (conv): Conv2d(131, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ConvLSTMCell(
        (conv): Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ConvLSTMCell(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
  )
  (conv): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
)
```

### Detailed Explanation of the Architecture:

#### ConvLSTM Cell:

Each ConvLSTMCell processes both the input sequence and the previous hidden state through convolutions. The ConvLSTM uses 4 gates:

1. **Input Gate (i)**: Determines how much of the current input should be stored in the cell state.
2. **Forget Gate (f)**: Decides what part of the previous cell state should be remembered.
3. **Output Gate (o)**: Determines what part of the cell state will be output as the hidden state.
4. **Candidate Cell State (g)**: A new candidate value that could be added to the cell state.

Each ConvLSTM cell applies a convolutional layer to the concatenated input and hidden state. This convolution allows the model to learn spatial patterns over time.

For example, in the first ConvLSTMCell:

```python
Conv2d(131, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

- The 131 input channels come from the concatenation of the input image sequence (3 channels) and the previous hidden state (128 channels), resulting in 131 input channels.
- The model then learns 512 output channels in the hidden state for this layer.

#### ModuleList:

A ModuleList is used to store multiple ConvLSTMCell layers. Each layer has a different number of input and output channels. In the example:

1. The first layer has 131 input channels (3 input channels + 128 hidden channels) and 512 output channels.
2. The second layer has 192 input channels and 256 output channels.
3. The third layer has 128 input channels and 256 output channels.

#### Final Convolution Layer:

After the ConvLSTM layers process the sequence, the output is passed through a final convolutional layer to generate the desired number of output channels (e.g., 3 for RGB images). This layer reduces the 64 channels (from the last ConvLSTM layer) to 3 channels.

```python
Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
```

The kernel size of `(1, 1)` ensures that the spatial dimensions (height and width) of the output remain the same, while the number of channels is adjusted.

---

## ConvLSTM in Sequence Prediction

ConvLSTM is particularly effective in sequence prediction tasks where both spatial and temporal dependencies are important. In sequence prediction, the model needs to predict future frames in a sequence of images based on the past frames, making ConvLSTM suitable for tasks like video frame prediction, image forecasting, and spatiotemporal data modeling.

### Structure of ConvLSTM for Sequence Prediction:

ConvLSTM combines the power of Convolutional Neural Networks (CNNs), which capture spatial information (e.g., textures, objects in an image), with Long Short-Term Memory (LSTM) networks, which capture temporal dependencies in sequences. This makes ConvLSTM ideal for handling video or image sequences where each frame is not only influenced by the previous frames but also by the structure of the image itself.

### How ConvLSTM is Used in Sequence Prediction:

In the context of sequence prediction, ConvLSTM is used as follows:

1. **Input Sequence**: The model is fed a sequence of input frames (e.g., video frames). The input sequence has a shape of `(batch_size, seq_len, channels, height, width)`, where `seq_len` is the number of frames in the input sequence.

2. **Spatiotemporal Processing**: Each ConvLSTM cell in the network processes one frame at a time, using both the current frame (spatial data) and the previous hidden state (temporal context). The convolutional operations allow the model to learn spatial features, while the LSTM component helps to capture the temporal relationships between frames.

3. **Hidden and Cell States**: The ConvLSTM cells maintain a hidden state and a cell state across time steps, which are passed from one time step to the next. These states help the model remember important temporal patterns over the sequence.

4. **Output Sequence**: After processing the sequence through the ConvLSTM layers, the final output is passed through a `1x1` convolutional layer to map the hidden state of the last ConvLSTM cell to the desired number of output channels (e.g., 3 channels for RGB frames).

5. **Prediction of Future Frames**: The output sequence consists of the predicted frames that the model generates based on the input sequence. This output can be compared with the target sequence using loss functions like Mean Squared Error (MSE) or SSIM (Structural Similarity Index) to evaluate the model's performance.

### Example:

Given a sequence of input frames:

```python
inputs = batch['input']  # Shape: (batch_size, seq_len, channels, height, width)
```

The ConvLSTM model processes this sequence frame by frame, maintaining the hidden state and cell state across time steps, and generates a sequence of predicted frames:

```python
outputs = model(inputs)  # Shape: (batch_size, output_frames, channels, height, width)
```

This output is the predicted sequence of future frames that the model has learned to generate based on the input sequence.

## Training and Evaluation

The model is trained on sequences of input images and evaluates the output frames using **Mean Squared Error (MSE)** loss and **Structural Similarity Index (SSIM)**.

### Loss Functions:

- **MSE Loss**: Measures the difference between predicted and target frames.
- **SSIM**: A perceptual metric that measures the structural similarity between predicted and target frames. This metric is useful for image quality assessments.

### Learning Rate Scheduler:

The `ReduceLROnPlateau` scheduler adjusts the learning rate when the validation loss plateaus, helping the model converge more efficiently.

### Early Stopping:

The `EarlyStopping` class is used to stop training when the validation loss does not improve for a specified number of epochs (patience).

---

## Dataset

The model is designed to work on sequences of images, such as video frames. The input consists of a sequence of frames, and the target is the future sequence of frames that the model should predict.

### Input Shape:

- **Shape**: `(batch_size, seq_len, channels, height, width)`
- **Example**: `(1, 10, 3, 64, 64)` where 10 is the number of input frames, 3 is the number of channels (RGB), and `64x64` is the spatial size.

### Output Shape:

- **Shape**: `(batch_size, output_frames, channels, height, width)`
- **Example**: `(1, 5, 3, 64, 64)` where 5 is the number of frames to predict, 3 is the number of output channels (RGB), and `64x64` is the spatial size.

---

## Training

### Key Components:

- **Optimizer**: Adam optimizer with a learning rate of `1e-3`.
- **Scheduler**: ReduceLROnPlateau scheduler to adjust the learning rate when the validation loss plateaus.
- **Early Stopping**: Stops training when the validation loss does not improve for a certain number of epochs (`patience=2`).
- **Metrics**: MSE loss and SSIM score are used to track the model's performance during training and evaluation.

### Training Loop:

For each epoch:

1. The model is trained on the training set.
2. The model is evaluated on the validation set.
3. The learning rate is adjusted if needed.
4. Early stopping checks if the validation loss has improved.

#### Example Training Code:

```python
# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    # Train the model and compute training loss
    ...

    # Validation phase
    model.eval()
    # Compute validation loss and SSIM score
    ...

    # Scheduler step
    scheduler.step(val_loss)

    # Early stopping check
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

---

## Evaluation

After training, the model is evaluated on a test set. The final evaluation metrics include:

1. **Test Loss**: The average MSE loss on the test set.
2. **Test SSIM**: The average SSIM score on the test set.

#### Example Evaluation Code:

```python
# Evaluation on Test Set
model.eval()
test_loss = 0.0
test_ssim = 0.0
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)

        # Compute SSIM for each frame
        ...
test_loss /= len(test_loader.dataset)
test_ssim /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}, Test SSIM: {test_ssim:.4f}')
```

---

## Conclusion

This ConvLSTM-based model is a powerful tool for sequence prediction tasks involving spatiotemporal data. It can be applied to a variety of applications, such as video frame prediction, image forecasting, and more.
