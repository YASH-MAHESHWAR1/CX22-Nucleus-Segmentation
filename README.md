# MultiResUNet for Nucleus & Cytoplasm Segmentation

## 1. Project Overview
This project implements the **MultiResUNet** architecture for semantic segmentation of cell nuclei and cytoplasm. The model is designed to improve upon the classical U-Net by addressing two specific limitations: multi-scale feature extraction and the semantic gap between encoder and decoder features. The implementation uses **TensorFlow/Keras** and achieves high segmentation accuracy on biomedical image data.

## 2. Mathematical Intuition & Model Architecture

### The Problem with Standard U-Net
1.  **Scale Variation:** Objects in medical images (like nuclei) vary significantly in size. Standard 3x3 convolutions may fail to capture features of varying scales simultaneously.
2.  **Semantic Gap:** In U-Net, features from the early encoder layers (low-level features like edges) are concatenated directly with features from the late decoder layers (high-level semantic features). This mismatch often makes optimization harder.

### The MultiResUNet Solution
This implementation addresses these issues using two key components:

#### A. The MultiRes Block (Multi-Resolution Analysis)
Instead of a simple sequence of 3x3 convolutions, the **MultiRes Block** extracts spatial features from multiple scales simultaneously.
* **Concept:** It approximates 5x5 and 7x7 convolutions using sequences of 3x3 convolutions to reduce computational cost while expanding the receptive field.
* **Implementation:** Inside `MultiRes_block`, the input is passed through three parallel paths:
    1.  A 3x3 convolution.
    2.  A sequence of two 3x3 convolutions (effective 5x5 field).
    3.  A sequence of three 3x3 convolutions (effective 7x7 field).
* **Feature Fusion:** The outputs of these three paths are concatenated combined with a residual connection (1x1 conv) from the input, allowing the model to learn features at different resolutions.

#### B. ResPath (Residual Path)
To bridge the semantic gap between the encoder (contracting path) and decoder (expanding path), this model replaces simple skip connections with **ResPaths**.
* **Concept:** Instead of directly copying features, the skip connection passes the encoder features through a series of convolution blocks.
* **Intuition:** This allows the low-level encoder features to undergo additional non-linear processing, making them semantically closer to the decoder features before concatenation.
* **Structure:** The length of the ResPath chain decreases (4, 3, 2, 1 blocks) as we move deeper into the network, because deeper layers are already semantically closer to the corresponding decoder layers.

## 3. Dataset and Preprocessing

### Data Source
The model processes dataset pickles containing cell images and their corresponding binary masks.
* **Inputs:** Single-channel or RGB microscopy images.
* **Targets:** Binary masks representing Nuclei/Cytoplasm regions.

### Preprocessing Pipeline
1.  **Normalization:** Images are cast to `float32` and normalized to the range `[0, 1]` to facilitate gradient descent convergence.
2.  **One-Hot Encoding:** The segmentation masks are converted to categorical format (one-hot encoded) with `num_classes=2` (Background vs. Foreground).
3.  **Data Augmentation:** To prevent overfitting on the training set, an `ImageDataGenerator` applies on-the-fly transformations including:
    * Rotation (up to 20 degrees)
    * Width/Height Shifts (0.2)
    * Horizontal/Vertical Flips
    * Shear and Zoom

## 4. Training Configuration

### Loss Function
A **Combined Loss** function is used to handle class imbalance and shape optimization:
$$\text{Loss} = \text{Binary Crossentropy} + \text{Dice Loss}$$
* **BCE:** pixel-wise classification accuracy.
* **Dice Loss:** optimizes for the overlap between predicted and ground truth shapes, critical for segmentation tasks.

### Evaluation Metrics
The model performance is tracked using a comprehensive suite of metrics:
* **Dice Coefficient:** The primary metric for segmentation overlap.
* **IoU Score (Jaccard Index):** Intersection over Union.
* **Precision & Recall (TPR):** To monitor false positives and false negatives.
* **Specificity:** Ability to correctly identify background.
* **F1-Score:** Harmonic mean of precision and recall.

### Hyperparameters
* **Optimizer:** Adam (`lr=1e-3`)
* **Batch Size:** 2 (optimized for GPU memory usage with heavy MultiRes blocks)
* **Epochs:** 200 (with early stopping)

### Callbacks
1.  **ModelCheckpoint:** Saves the model weights only when `val_dice_coefficient` improves.
2.  **ReduceLROnPlateau:** Reduces learning rate by factor of 0.5 if validation loss stagnates for 7 epochs.
3.  **EarlyStopping:** Stops training if `val_dice_coefficient` does not improve for 20 epochs.

## 5. Results

The model demonstrates strong convergence capabilities. Based on the training logs provided:

* **Training Performance:**
    * **Dice Coefficient:** ~0.93
    * **Accuracy:** ~99.9%
    * **True Positive Rate:** ~93%
* **Validation Performance:**
    * **Dice Coefficient:** ~0.83 - 0.84
    * **Accuracy:** ~99.7%
    * **Precision:** ~86%

*Analysis:* The model achieves excellent pixel-wise accuracy. The gap between training Dice (0.93) and validation Dice (0.83) indicates the model has high capacity, and the heavy data augmentation pipeline successfully keeps the validation score stable and high.

## 6. How to Run

### Requirements
* Python 3.x
* TensorFlow 2.x
* Keras
* NumPy, Pandas, Matplotlib

### Execution Steps
1.  **Load Data:** Ensure the pickle files (`Images.pkl`, `Masks.pkl`) are located in the specified input directory.
2.  **Initialize Environment:**
    ```python
    import tensorflow as tf
    # Code automatically handles GPU memory growth
    ```
3.  **Build Model:**
    ```python
    model = build_multiresunet(input_shape=(512, 512, 3), num_classes=2)
    ```
4.  **Train:**
    Run the training cell. Weights are automatically saved to `multiresunet_single_normalized.weights.h5`.
5.  **Visualize:**
    Use the provided visualization function to compare Input Images, Ground Truths, and Predicted Masks with overlays.
