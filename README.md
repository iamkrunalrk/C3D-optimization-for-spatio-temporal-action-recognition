# C3D: Action Recognition with 3D Convolutional Networks

This repository implements the **C3D** (Convolutional 3D) model for action recognition, as proposed in the paper ["C3D: Generic Features for Video Analysis"](https://arxiv.org/abs/1412.0767). The C3D model uses 3D convolutions to extract spatial and temporal features from videos and is widely used for action recognition tasks in video analysis.

## Overview

C3D is a deep learning model that applies 3D convolutions on video frames to extract spatiotemporal features, making it particularly effective for video-based action recognition. The architecture is based on 3D convolutional layers that extend the 2D convolutions used in traditional image processing to the temporal domain, providing a robust feature extraction mechanism for video data.

This repository includes a TensorFlow implementation of C3D for video action recognition, supporting training and testing with datasets like **UCF101**.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy
- OpenCV


## Getting Started

### 1. **Dataset Preparation**

This implementation uses the **UCF101 dataset**, which consists of 13,320 video clips in 101 action categories. You need to download and prepare the dataset to train the model.

#### Steps:

1. **Download the UCF101 Dataset:**
   - Download the dataset from the official [UCF101 dataset page](http://crcv.ucf.edu/data/UCF101.php).
   - Extract the dataset into a directory of your choice.

2. **Modify Dataset Path:**
   - Modify the `dataset_path` in `create_dataset.py` to point to your extracted UCF101 dataset.

3. **Convert Videos to TFRecord Format:**
   The model requires the dataset to be in the **TFRecord** format. The `create_dataset.py` script converts the raw video data into TFRecord files.

   Run the following command to create the training and testing TFRecords:

   ```bash
   python3 create_dataset.py
   ```

   This will generate two files: `trainset.tfrecord` and `testset.tfrecord` which contain the processed video data. These files will be used for training and evaluation.

### 2. **Model Training**

Once the dataset is prepared, you can begin training the C3D model.

#### Training the Model:

To train the C3D model on the UCF101 dataset, run the following command:

```bash
python3 train_c3d.py
```

This will:

1. Load the dataset from the generated `trainset.tfrecord` file.
2. Train the C3D model on the dataset, optimizing for action recognition.
3. Save the trained model checkpoints.

#### Monitor Training:

You can monitor the training progress by using **TensorBoard**. Simply run:

```bash
tensorboard --logdir=logs
```

This will allow you to visualize loss curves, accuracy, and other metrics.

### 3. **Model Testing / Action Recognition**

Once the model is trained, you can test it on a single video clip to recognize actions.

#### Testing with a Single Video:

1. **Modify the Video Path:**
   Update the `video_path` variable in `ActionRecognition.py` to point to the video you want to classify.

2. **Run the Action Recognition Script:**

   ```bash
   python3 ActionRecognition.py
   ```

   This script will load the trained model and predict the action label of the provided video.

### 4. **Pre-trained Model**

If you don’t want to train the model from scratch, you can use a pre-trained C3D model. The pre-trained model is available for download from **Baidu Cloud**:

- [Pre-trained model download](https://pan.baidu.com/s/1pD4R0k23HOi_RIrLGZSlOg)

Once you download and extract the pre-trained model, you can load it for testing or fine-tuning on your own dataset.

### 5. **Converting the Model for Serving**

For deploying the trained model, you can convert it into a format suitable for serving in production environments.

#### Convert the Model:

To convert the model to a TensorFlow SavedModel format for deployment:

```bash
python3 convert_model.py
```

This will save the model in the correct format for serving with TensorFlow Serving or other deployment tools.


## Model Architecture

The **C3D (Convolutional 3D)** model is a deep neural network designed specifically for action recognition in videos. The architecture leverages 3D convolutions, which allows the network to learn spatiotemporal features from video frames.

### C3D Architecture Details:

1. **Input Layer:**
   - The model takes video clips of size `16 x 112 x 112 x 3` as input, where:
     - 16 frames per video clip.
     - Each frame is resized to `112x112` with 3 color channels (RGB).
   
2. **Convolutional Layers:**
   - Conv3D layers with kernel size of `3x3x3`, followed by layer normalization and max pooling.

3. **Fully Connected Layers:**
   - Flatten the output of the convolutional layers.
   - Two dense layers of 4096 units each with ReLU activations and dropout to prevent overfitting.

4. **Output Layer:**
   - A softmax output layer with `class_num` units, where `class_num` is the total number of action classes (e.g., 101 for UCF-101).

5. **Loss Function:**
   - The model uses `softmax_cross_entropy` loss during training.

6. **Optimizer:**
   - Adam optimizer with a learning rate of `1e-4`.

7. **Metrics:**
   - During evaluation, the accuracy of the predictions is computed.


## Directory Structure

```
C3D/
│
├── create_dataset.py          # Script for creating TFRecord dataset
├── train_c3d.py               # Script for training the C3D model
├── ActionRecognition.py       # Script for video action recognition
├── convert_model.py           # Script for converting model for serving
├── models/                    # Folder containing model weights and 
├── logs/                      # Folder for TensorBoard logs
└── README.md                  # This file
```

