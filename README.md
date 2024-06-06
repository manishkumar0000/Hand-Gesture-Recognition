# Hand_Gesture_Recognition

We trained a classification model to recognise Hand Gestures from the HAGRID Dataset. We tried out various deep learning architectures like InceptionV3, VGG16, DenseNet and the ResNet family. Amongst all these architectures the DenseNet201 proved to be the most efficient and predicted with an accuracy of nearly 94 percent. 
We had fine-tuned the CNN models with depthwise-convolution layers and with the use of Pooling. We observed the accuracies with a small number of epochs at first and tuned the model accordingly by adding layers and changing the pooling type.

Below is a video that was processed using MediaPipe to detect hands and the cropped hand images were sent to the best performing model for classification among the 19 gesture classes.

The different models we had tried are in the Hagrid_all_models.ipynb.

The preprocessing of the HAGRID Dataset Images are explained in the data_prep.ipynb

The prediction on a video can be done using the pred_video.py file


https://user-images.githubusercontent.com/58504532/192094570-ccca7d15-2ad3-4c9e-bad4-afe74ab7fecd.mp4

