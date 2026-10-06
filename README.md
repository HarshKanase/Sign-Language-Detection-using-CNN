## Sign Language Detection using CNN

This project is a **deep learning-based sign language detection system** designed to recognize and classify hand gestures using **Computer Vision and Convolutional Neural Networks (CNNs)**. The system takes visual input from a camera or image, detects the user's hand, processes the relevant features, and predicts the corresponding sign using a trained deep learning model.

The project combines **MediaPipe, OpenCV, PyTorch, and CNN-based image classification** to create a complete hand-sign recognition pipeline. MediaPipe is used for reliable hand detection and tracking, while the extracted hand information is passed through the trained neural network for gesture classification. This approach helps reduce the effect of irrelevant background information and allows the model to focus primarily on the hand and its gesture.

### Knowledge Distillation

A major component of this project is the implementation of **Knowledge Distillation**. The project uses a larger and more capable **Teacher model** to transfer its learned knowledge to a smaller **Student model**. Instead of training the smaller model only from the original labels, the Student model learns from the predictions and knowledge produced by the Teacher model.

This allows the Student model to achieve a good balance between **accuracy and computational efficiency**, making it more suitable for applications where fast inference and lower resource requirements are important. The repository includes the trained weights for both the Teacher and Student models.

### Project Pipeline

The overall system follows a pipeline similar to:

**Camera/Image Input → Hand Detection → Image/Hand Processing → CNN Model → Sign Classification → Predicted Gesture**

The project contains the complete inference implementation along with the trained model weights, allowing the trained models to be used for prediction without requiring the entire training process again.

### Key Features

* **Real-time hand sign detection** using computer vision.
* **CNN-based deep learning model** for gesture classification.
* **MediaPipe-based hand detection and tracking**.
* **OpenCV integration** for handling image/video input.
* **Knowledge Distillation** using Teacher and Student models.
* Includes pretrained **Teacher and Student model weights**.
* Designed with an emphasis on efficient and practical sign recognition.
* Python and PyTorch-based implementation suitable for further experimentation and model improvements.

### Technologies Used

* **Python**
* **PyTorch**
* **Convolutional Neural Networks (CNN)**
* **MediaPipe**
* **OpenCV**
* **Computer Vision**
* **Deep Learning**
* **Knowledge Distillation**

Overall, this project demonstrates how **deep learning, computer vision, and knowledge distillation** can be combined to build an efficient sign language recognition system. It also provides a foundation for further development toward larger vocabularies, improved recognition accuracy, real-time applications, and more lightweight deployment on resource-constrained devices.

