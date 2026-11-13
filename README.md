# Pokémon Image Classifier

## Overview

A Pokémon image classification application built with PyTorch and ResNet18. The model is trained to recognize all 151 Generation 1 Pokémon from images. Users can upload an image through a Streamlit interface or make predictions through the command line.

The project uses the [Pokémon Generation One dataset](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) from Kaggle.

## Files

- **`train.py`** - Trains the ResNet18 model using the training dataset and evaluates its performance using the test dataset. Allows previously saved model weights to be loaded before training.
- **`predict.py`** - Loads the trained model and predicts the Pokémon shown in an image through the command line. Displays the top five predictions with their confidence scores.
- **`app.py`** - Provides a Streamlit web interface where users can upload an image and view the model's top five predictions.
- **`model.pth`** - Stores the trained model weights.
- **`requirements.txt`** - Lists the Python dependencies required to run the project.

## Saving and Loading the Model

After training, `train.py` prompts the user to save the model weights to `model.pth`. These weights can be loaded later to continue training or make predictions without training the model again.

To load existing weights, choose `Y` when prompted at the beginning of `train.py`. The model architecture is recreated and the saved weights are applied to it.

The `predict.py` and `app.py` scripts automatically load `model.pth` to make predictions.

## Running the Project

Install the required dependencies:

pip install -r requirements.txt

**Train the model:**

py train.py

**Make predictions through the command line:**

py predict.py

Enter the image filename when prompted.

**Launch the Streamlit application:**

streamlit run app.py

Open the local URL displayed in the terminal to use the application.
