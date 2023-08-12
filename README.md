# LSTM-Based Action Detection for Translating Filipino Sign Language into Text

![maxresdefault](https://github.com/cielo112/FSL_Detection/assets/113077476/29e0c3a9-531e-4fe3-85ef-6d6acd2d3194)
Photo credits to: FSL Mabuhay Youtube Channel

This repository contains an LSTM-based action detection system designed to translate Filipino Sign Language (FSL) into text. The system employs Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to recognize various sign language gestures and actions and convert them into corresponding textual representations.

**Features**  
Utilizes LSTM networks for accurate sequence learning and action detection.
Preprocessed dataset of Filipino Sign Language gestures with labeled actions.
Training code for LSTM model with customizable hyperparameters.
Inference pipeline for real-time action detection and translation.
Example demo showcasing the model's performance on FSL to text translation.

**Dataset**  
The dataset used for training and evaluation is not provided in this repository due to size limitations. However, you can utilize the data collection script included in the repository to gather data using your local machine and webcam. 

**Results**  
Our LSTM-based action detection system achieves a recognition accuracy of 95% on the test dataset. It demonstrates promising performance in real-time FSL translation to text, contributing to improved communication accessibility for the hearing-impaired. The trained model weights are included in the repository as 'model_weights.h5'.
![output](https://github.com/cielo112/FSL_Detection/assets/113077476/0183302d-baa3-4150-992f-71cc7fe484fc)



**Recommendations**  
The script for data collection as well as the model has only been written and trained to translate common FSL greeting such as 'Magandang Umaga', 'Magandang Hapon', and 'Magandang Gabi'. Futher training data is needed in order to make the model usuable for a wider use case. The reader may also utilize the code as a starting point for creating a model capable of interpreting more FSL actions.
