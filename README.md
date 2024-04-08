# Handwritten Meitei-Mayek Recognition using CNN
## Abstract:
Handwritten character recognition is a challenging task in the field of computer vision and machine learning. Meitei Mayek, an ancient script used in the India state of Manipur, presents a unique significant challenge due to its complex
characters and limited availability of resources and advancements in the field.
In this project, we propose a Convolutional Neural Network (CNN) approach
implemented in PyTorch to address this challenge. The CNN architecture is
designed to effectively learn the intricate features of Meitei Mayek characters
from handwritten samples, achieving an accuracy of 90%.
The complexity of Meitei Mayek characters lies in their intricate shapes and
varying stroke patterns, making traditional recognition methods ineffective.
Additionally, the limited availability of datasets and research resources further
exacerbates the difficulty of this task. However, leveraging the power of deep
learning, particularly CNNs, offers a promising solution to this problem.
The main motive behind this particular project was that Google and other
internet companies released good recognition and translation systems for all
languages except for our language Manipuri, which drove me towards working
on computer vision skills to implement the same for regional languages like
ours. Through extensive experimentation and fine-tuning, we achieved an
impressive accuracy of 90% on the recognition task. This accuracy level
demonstrates the efficacy of our proposed CNN architecture in accurately
deciphering handwritten Meitei Mayek characters. Our approach not only
achieves high accuracy but also showcases the potential of deep learning
techniques in tackling challenges associated with script recognition, especially
for less-resourced languages like Meitei Mayek.

## Introduction:
*Platform Used:* Google Colab (jupyter notebook, T4 GPU)
*Tools and Libraries:* PyTorch, PIL, pandas, numpy, matplotlib,
*Frameworks:* Flask, FastAPI
The Meitei Mayek script, used for writing
the Manipuri language, dates back to the
eleventh century but faced suppression
during the eighteenth century due to the
influence of Hinduism/Brahmanism. This
led to the destruction of sacred
scriptures, known as the "Puya
Meithaba." Efforts were made from 1930
to 1980 to revive the script, with debates
on its character set. In 1980, the
Government of Manipur decided on 27
letters for the script. It was reintroduced
into academic curriculum in 2005–2006.
However, a significant portion of the
population remains unfamiliar with the
script, creating a transition from Bangla to
Meitei Mayek. To facilitate
communication, a transliteration system
from Bangla to Meitei Mayek and vice
versa is crucial. Recent developments
have been made in terms of research papers trying to achieve a similar
objective of creating a benchmark dataset of Meitei Mayek characters and also
building robust recognition and translation models. Here in this project, I
proposed a CNN model having multiple layers to achieve a high performance of
90% accuracy.
## Dataset:
For our project on Handwritten Meitei Mayek recognition using Convolutional
Neural Networks (CNNs) in PyTorch, we utilized the dataset provided by Deena
Hijam et al in their 2020 paper “On developing complete character set Meitei
Mayek handwritten character database”. The dataset consists of a collection of
handwritten Meitei Mayek characters, meticulously curated to cover a wide
range of variations in writing styles, stroke patterns, and character shapes.
Each character in the dataset is annotated with its corresponding label,
enabling supervised learning for character recognition tasks. From the huge
dataset containing over 1200 images for each class/label of characters, I
selected only a few train and test examples due to limitation of my compute
power (I have access to CPU and limited GPU).
**Dataset source:** http://agnigarh.tezu.ernet.in/~sarat/resources.html
**No of labels/classes:** 55
**No of training images used for each label:** 150
**No of testing images used for each label:** 25

## Preprocessing:
Before feeding the images into the CNN model, we perform preprocessing
steps including resizing, normalization, and augmentation. Resizing ensures
that all images are of uniform dimensions suitable for the CNN architecture.
The original pixel size of the character images have been modified to 64x64
during the preprocessing.
*Tools used:* PIL (Python Imaging Library)

## Model Architecture:
The CNN Model presented is tailored for image classification tasks, particularly
suited for applications like handwritten character recognition. It comprises
three convolutional layers, each followed by Rectified Linear Unit (ReLU)
activation functions to introduce non-linearity, promoting feature learning.
Max-pooling layers with 2x2 kernels and a stride of 2 are interleaved between
the convolutional layers for spatial down-sampling, aiding in feature
extraction. The architecture maintains spatial dimensions through padding in
convolutional layers, preserving crucial information. The final feature map is
flattened and fed into two fully connected layers. The first dense layer consists
of 512 units, followed by ReLU activation and dropout regularization with a
probability of 0.5 to mitigate overfitting. The second dense layer outputs logits
for classification into the specified number of classes, finalizing the model's
prediction process.
 
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
## Training:
For training the CNNModel, we utilized the Cross Entropy Loss function
(nn.CrossEntropyLoss()) and stochastic gradient descent (SGD) optimizer
(torch.optim.SGD). The chosen learning rate for the optimizer was set to 0.01.
Early stopping is employed to prevent overfitting, where training is stopped if
the validation loss does not improve for a certain number of epochs.
Throughout the training process, the loss was monitored to assess the model's
performance and convergence. The training and testing loop was repeated for
a predetermined number of epochs (50), ensuring the model had sufficient
opportunities to learn from the training data and optimize its parameters. By
utilizing the defined loss function and optimizer, the CNNModel underwent
iterative training, gradually improving its ability to classify handwritten Meitei
Mayek characters with increased accuracy.

## Evaluation:
We calculate metrics such as accuracy, precision, recall, and F1-score to
measure the model's performance. Additionally, we visualize the confusion
matrix to understand the distribution of misclassifications among different
characters. Our CNN model achieves an accuracy of over 90% on the test set,
demonstrating its effectiveness in recognizing handwritten Meitei Mayek
characters. The precision, recall, and F1-score for individual characters are also
reported, indicating the model's performance on each class.
Sample Prediction:

## Deployment:
Following the successful training of the model, the subsequent step involves
deploying it for accessibility in inference tasks. To accomplish this, we utilized
Flask, a Python web framework, to establish a website for deploying the
model. This encompassed several steps: firstly, setting up a Flask application to
act as the foundation of our web service, entailing the creation of routes to
handle incoming requests and the definition of corresponding functions to
process these requests. Secondly, we seamlessly integrated the trained CNN
model into the Flask application, encompassing the loading of model weights
and architecture, ensuring readiness for inference.

## Conclusion:
In conclusion, we have presented a CNN-based approach for handwritten
Meitei Mayek character recognition implemented in PyTorch. The model
shows promising results in accurately recognizing characters from this ancient
script. This work contributes towards preserving and digitizing cultural heritage
through automated recognition systems.

## Future Work:
In the future, we aim to enhance the deployed model's impact and usability.
This may involve extending its capabilities beyond Meitei Mayek recognition to
broader image classification tasks like object recognition or scene
understanding, making it more versatile. We also plan to integrate translation
features into the website, enabling seamless text transliteration between
Meitei Mayek and other languages for improved accessibility. Optimizing the
model for large-scale deployment to ensure scalability and efficiency under
high-demand scenarios is another priority. Exploring advanced techniques like
Transformer-based architectures or ensemble learning approaches could
further improve recognition accuracy and generalization. Additionally,
expanding the project scope to include support for other regional languages in
the Northeastern Indian subcontinent would promote linguistic diversity and
cultural preservation, enhancing the model's societal impact. These future
endeavors aim to evolve the deployed model into a versatile and indispensable
tool with broad implications across various domains.
