# Handwritten Meitei-Mayek Recognition using CNN
![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/b309a8e6-b327-4046-9afa-28dcb7769ede)

## Introduction
[Meitei Mayek](https://en.wikipedia.org/wiki/Meitei_script), an ancient script used in the Indian state of [Manipur](https://en.wikipedia.org/wiki/Manipur), presents a unique significant challenge in Handwritten text recognition due to its complex characters and limited availability of resources like data in the field. The complexity of Meitei Mayek characters lies in their intricate shapes and varying stroke patterns, making traditional recognition methods ineffective. In this project, we propose a Convolutional Neural Network (CNN) approachimplemented in PyTorch. The main motive behind this project was that Google and other internet companies released good recognition and translation systems for all languages except for our language [Manipuri](https://en.wikipedia.org/wiki/Meitei_language), which drove me towards working on computer vision skills to implement the same for regional languages like ours. Through extensive experimentation and fine-tuning, we achieved an accuracy of 90% on the recognition task.

## Requirements

Platform Used: `Google Colab` (T4 GPU)

Tools and Libraries: `PyTorch`, `PIL`, `pandas`, `numpy`, `matplotlib`

Deployment Framework: `Flask`

## Dataset 

The dataset provided by the paper "[(Deena et al, 2021) On developing complete character set Meitei Mayek handwritten character database](https://link.springer.com/article/10.1007/s00371-020-02032-y)" was used for the project. The dataset consists of a collection of handwritten Meitei Mayek characters, meticulously curated to cover a wide range of variations in writing styles, stroke patterns, and character shapes. Each character in the dataset is annotated with its corresponding label, enabling supervised learning for character recognition tasks. From the huge dataset containing over 1200 images for each class/label of characters, I selected only a few train and test examples due to limitation of my computepower (I have access to CPU and limited GPU provided by Colab).

![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/c9446458-269c-4f72-8df1-45665cbc89c2)

**Dataset Source:** [original source](http://agnigarh.tezu.ernet.in/~sarat/resources.html)

You can also download a smaller dataset [here](https://drive.google.com/drive/folders/1Y3LL42Ppvqq7W1uglq6_yBHvIiZdwPkB?usp=sharing).

**No of labels/classes:** 55

**No of training images used for each label:** 150

**No of testing images used for each label:** 25

## Preprocessing
Before feeding the images into the CNN model, we perform preprocessing steps including resizing, normalization, and augmentation. Resizing ensures that all images are of uniform dimensions suitable for the CNN architecture. The original pixel size of the character images have been modified to 64x64.
![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/9458dc02-8823-4cc7-8228-a07a583753a1)


## Model Architecture
The CNN Model comprises of three convolutional layers, each followed by Rectified Linear Unit (ReLU) activation functions to introduce non-linearity, promoting feature learning. Max-pooling layers with 2x2 kernels and a stride of 2 are interleaved between the convolutional layers for spatial down-sampling, aiding in feature extraction. The architecture maintains spatial dimensions through padding in convolutional layers, preserving crucial information. The final feature map isflattened and fed into two fully connected layers. The first dense layer consists of 512 units, followed by ReLU activation and dropout regularization with aprobability of 0.5 to mitigate overfitting. The second dense layer outputs logitsfor classification into the specified number of classes, finalizing the model's prediction process.
 ![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/2e856321-e56b-414c-b57d-c49d1114754c)

`CNNModel(
 (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (relu1): ReLU()
 (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
ceil_mode=False)
 (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (relu2): ReLU()
 (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
ceil_mode=False)
 (conv3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (relu3): ReLU()
 (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
ceil_mode=False)
 (flatten): Flatten(start_dim=1, end_dim=-1)
 (fc1): Linear(in_features=16384, out_features=512, bias=True)
 (relu4): ReLU()
 (dropout): Dropout(p=0.5, inplace=False)
 (fc2): Linear(in_features=512, out_features=55, bias=True)
)`
## Training
For training the CNNModel, we utilized the Cross Entropy Loss function `nn.CrossEntropyLoss()` and stochastic gradient descent (SGD) optimizer `torch.optim.SGD`. The chosen learning rate for the optimizer was set to 0.01. Early stopping is employed to prevent overfitting, where training is stopped if the validation loss does not improve for a certain number of epochs. Throughout the training process, the loss was monitored to assess the model's performance and convergence. The training and testing loop was repeated for a predetermined number of epochs (50), ensuring the model had sufficient opportunities to learn from the training data and optimize its parameters. By utilizing the defined loss function and optimizer, the CNNModel underwent iterative training, gradually improving its ability to classify handwritten Meitei Mayek characters with increased accuracy.
![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/88c558b9-4eac-412a-ac66-27dfff7b21c3)

## Evaluation
We calculate metrics such as accuracy, precision, recall, and F1-score tomeasure the model's performance. Additionally, we visualize the confusion matrix to understand the distribution of misclassifications among different characters. Our CNN model achieves an accuracy of over 90% on the test set, demonstrating its effectiveness in recognizing handwritten Meitei Mayek characters.
![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/21c57d2c-d13a-4c06-893c-3d3ad60b73ac)
![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/3344e592-a749-4504-ac54-3e3a26048fc3)

## Deployment
Flask, a simple Python web framework, was used to create a website for deploying the model.
![image](https://github.com/galax19ksh/Handwritten-Meitei-Mayek-Recognition/assets/112553872/3acc6095-4ffe-403e-92b4-597b6393c31b)

## Conclusion
In conclusion, we have presented a CNN-based approach for handwritten Meitei Mayek character recognition implemented in PyTorch. The model shows promising results (90%) in accurately recognizing characters from this ancient script. This work contributes towards preserving and digitizing cultural heritage through automated recognition systems.

### Future Work
We plan to expand the model's capabilities beyond Meitei Mayek recognition to include object recognition, scene understanding, and text translation. We will optimize the model for large-scale deployment and explore advanced techniques like Transformer-based architectures (vision transformers) and ensemble learning to improve accuracy and generalization. Additionally, we aim to support other regional languages in Northeastern India to promote linguistic diversity and cultural preservation.


