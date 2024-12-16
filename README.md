# Image Classification with Deep Learning

This project demonstrates an image classification task using deep learning techniques. The model is trained to classify images into specific categories based on the dataset provided. It uses PyTorch for model training and evaluation and OpenCV for image processing. 

## Project Structure

- **Training Folder**: Contains the subdirectories for different classes of images. Each subdirectory corresponds to a class and contains images for training.
- **Testing Folder**: Contains images that will be tested by the trained model to evaluate its performance.

## Requirements

To run this project, you need to install the following Python libraries:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `opencv-python`
- `random`

You can install these libraries using `pip`:

```bash
pip install torch torchvision numpy matplotlib opencv-python random
```
Dataset Structure
The dataset is organized as follows:

```
Training_folder/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    └── ...
Testing_folder/
    ├── test_image_1.jpg
    ├── test_image_2.jpg
```
Training_folder: Contains subdirectories where each subdirectory represents a different class (e.g., class_1, class_2, etc.).
Testing_folder: Contains images for testing the model.
Training
Ensure the dataset is organized into the appropriate structure in the Training_folder.
Train the model using the code provided in the script (not included in this README).
After training, the model weights will be saved in a file (model.pt).
Testing
Load the trained model.
Provide the path to a test image for classification.
The model will predict the class of the image and display it.
