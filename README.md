# Comment Toxicity Detector - Readme

## Introduction

The Comment Toxicity Detector is a machine learning model designed to classify toxic comments from non-toxic comments. The model is built using embedding and bidirectional LSTM layers, followed by three fully connected layers. It has been trained on the Jigsaw Toxic Comment Classification Challenge dataset.

## Model Architecture

The model architecture consists of the following layers:

1. **Text Vectorization**: The model uses a text vectorization process for text preprocessing. This step converts the input text comments into numerical representations that can be fed into the neural network.

2. **Embedding Layer**: The text representations are then passed through an embedding layer to capture the semantic relationships between words in the comments.

3. **Bidirectional LSTM Layer**: The bidirectional LSTM layer enables the model to capture contextual information from both the forward and backward directions of the comment sequences.

4. **Fully Connected Layers**: Three fully connected layers are used to further process the extracted features and make predictions.

## Training and Evaluation

The model is trained on the Jigsaw Toxic Comment Classification Challenge dataset. Even though the training is performed for only one epoch, the accuracy results are reported to be pretty good.

The training process is visualized through plots of epoch accuracy and epoch loss, showing how the model's performance evolves over each epoch.

## Interface - Gradio

The trained model is deployed on the Gradio interface, where users can input a comment, and the model will provide the toxicity level detection as output. The Gradio interface makes it easy and intuitive for users to interact with the model and get real-time predictions.

## Directory Structure

The project directory contains the following files:

- `Logs_train.png`: A plot of training logs showing the loss and accuracy metrics during the training process.

- `Predictions.py`: Python script to make predictions using the trained model.

- `README.md`: This readme file explaining the project and its components.

- `Train.py`: Python script for model training on the Jigsaw Toxic Comment Classification Challenge dataset.

- `epoch_accuracy.png`: A plot of epoch accuracy during the training process.

- `epoch_loss.png`: A plot of epoch loss during the training process.

- `evaluation_accuracy_vs_iterations.png`: A plot showing the evaluation accuracy versus iterations.

- `evaluation_loss_vs_iterations.png`: A plot showing the evaluation loss versus iterations.

## Conclusion

The Comment Toxicity Detector model is a powerful tool for classifying toxic comments and promoting a safer and healthier online community. With its accurate predictions and easy-to-use Gradio interface, the model can be integrated into various platforms to moderate and filter harmful comments effectively.
