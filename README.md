# SiameseCNN-For-FaceRecognizition
Siamese CNN For Face Recognition tested on ATTDataset. Written using tensorflow 1.0.

Best Architecture:
Siamese of 3 CNN layers connected with a 480 dense layer followed by a 40 dense layer for classification using softmax as activation
function and cross entropy loss.

It current achieves a highest accuracy of 71.5% after epochs = 1000, batch size = 10.

There's a lot of room for hyper-parameter tuning.
