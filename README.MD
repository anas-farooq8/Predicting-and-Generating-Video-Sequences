# Video Data Processing and Loading Pipeline

## Overview

This branch contains the code for processing, subsetting, scaling, and normalizing video datasets for Predicting and Generating Video Sequences, particularly sequence modeling. The pipeline involves the preparation of video datasets by subsetting specific labels, shuffling, and splitting data into sequences for training, validation, and testing.

Special emphasis is given to the `VideoDataset` class, which provides efficient caching, sequence preparation, and transforms, making the video data ready for deep learning models.

---

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Pipeline Workflow](#pipeline-workflow)
   - [Loading Data](#loading-data)
   - [Data Subsetting](#data-subsetting)
   - [Shuffling and Preprocessing](#shuffling-and-preprocessing)
   - [Label Visualization](#label-visualization)
3. [Video Dataset Class](#video-dataset-class)
   - [Key Features](#key-features)
   - [How it Works](#how-it-works)
4. [Data Loaders](#data-loaders)
   - [Purpose](#purpose)
   - [Batch Loading](#batch-loading)
5. [Video Frame Loading and Visualization](#video-frame-loading-and-visualization)
6. [Code Execution Details](#code-execution-details)

---

## Dataset Description

The dataset comprises video clips labeled with action categories such as `BenchPress`, `BoxingPunchingBag`, `HorseRiding`, `PlayingCello`, and `PlayingGuitar`. Each video is represented by:

- A video file path (`clip_path`).
- A corresponding label (`label`).

The dataset is split into three subsets:

- Training
- Validation
- Testing

---

## Pipeline Workflow

### Loading Data

The pipeline begins by reading CSV files containing metadata for the training, validation, and test sets:

```python
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)
```

### Data Subsetting

Only videos belonging to five specific actions are retained:

```python
PENTA_LABELS = ['BenchPress', 'BoxingPunchingBag', 'HorseRiding', 'PlayingCello', 'PlayingGuitar']
train_penta_df = train_df[train_df['label'].isin(PENTA_LABELS)]
```

### Shuffling and Preprocessing

To ensure randomness and robustness during training, the data is shuffled:

```python
train_penta_df = train_penta_df.sample(frac=1.0).reset_index(drop=True)
```

The `clip_path` column is updated to include the full file path:

```python
train_penta_df['clip_path'] = train_penta_df['clip_path'].apply(lambda path: DATASET_PATH[:-1] + path)
```

### Label Visualization

The distribution of labels is visualized using bar plots for training, validation, and test datasets. Custom color palettes are used to distinguish different action categories visually.

---

## Video Dataset Class

The `VideoDataset` class is the cornerstone of the pipeline. It extends PyTorch’s `Dataset` to handle the complexities of video data efficiently.

### Key Features

- **Caching**: Video data is cached in memory to minimize redundant disk I/O.
- **Sequence Preparation**: Videos are split into overlapping sequences of input and target frames.
- **Augmentation**: Missing frames are handled gracefully by repeating the last available frame.
- **Dynamic Loading**: Videos are resized and transformed on-the-fly based on configuration.

### How it Works

1. **Initialization**:

   - Sequences are precomputed for all videos using a stride-based approach.
   - The `get_total_frames` method determines the frame count for each video.

2. **Video Loading**:

   - Videos are loaded using OpenCV and resized to a specified frame size.
   - Frames are converted to grayscale or RGB based on the `color_mode` parameter.

3. **Sequence Extraction**:

   - Input and target frames are extracted from cached videos based on the sequence index.

4. **Transforms**:
   - Normalization is applied to scale pixel values to a range of `[0, 1]`.

```python
input_seq = input_seq.transpose(0, 3, 1, 2)
input_seq = torch.from_numpy(input_seq).float() / 255.0
```

---

## Data Loaders

The PyTorch `DataLoader` is used to handle batch loading:

```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

### Purpose

- Efficient batch processing for training models.
- Seamless integration with PyTorch’s deep learning ecosystem.

### Batch Loading

Each batch contains:

- `input`: Input frames for the model.
- `target`: Frames the model needs to predict.

```python
for batch in train_loader:
    input_seq = batch['input']
    target_seq = batch['target']
```

---

## Video Frame Loading and Visualization

The `load_video_frames` function extracts a specified number of frames from a video. If a video has fewer frames than required, black frames are generated to pad the sequence.

Frames can be visualized to verify preprocessing:

```python
view_frames(sample_video, frame_step=15)
```

---

## Code Execution Details

1. **Dataset Preparation**:
   - CSV files and videos are assumed to be located in a `data/` directory.
2. **Frame Resizing**:
   - Frames are resized to `64x64` for computational efficiency.
3. **Normalization**:
   - Pixel values are normalized to `[0, 1]`.

---
