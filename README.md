Skin Cancer Detection.
This project presents a deep learning-based image classification system to detect and classify various types of skin cancer using dermatoscopic images. Leveraging Convolutional Neural Networks (CNN), this project aims to provide an efficient and accurate tool to assist dermatologists in early diagnosis of skin cancer.

🌐 Project Overview
The classification model is trained on the HAM10000 dataset containing dermatoscopic images of skin lesions. The pipeline includes data preprocessing, augmentation, class balancing using SMOTE, CNN-based model building, and performance evaluation using multiple metrics.

📃 Dataset
Source: HAM10000 - Human Against Machine with 10000 training images
Contains over 10,000 dermatoscopic images categorized into seven skin disease types:
Melanocytic nevi
Melanoma
Benign keratosis-like lesions
Basal cell carcinoma
Actinic keratoses
Vascular lesions
Dermatofibroma
Includes patient metadata like age, sex, and localization
📊 Features
CNN Model Architecture:
Includes Conv2D, MaxPooling2D, Dropout, BatchNormalization, and Dense layers
Data Preprocessing:
Label encoding and one-hot encoding
Handling missing values
SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
Training Optimization:
Image augmentation using ImageDataGenerator
Learning rate scheduling using ReduceLROnPlateau
Evaluation Metrics:
Confusion matrix
ROC-AUC Score
Classification Report
🎓 Technologies Used
Python
TensorFlow/Keras
Scikit-learn
Pandas & NumPy
Matplotlib & Seaborn
SMOTE (from imblearn)
🚀 How to Run the Project
Clone the repository:
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection
Install dependencies:
pip install -r requirements.txt
Download the dataset and place it in the project folder
Run the Jupyter Notebook:
jupyter notebook project.ipynb
🚀 Sample Results
Visualizations of class distributions and metadata
High accuracy in skin cancer classification
ROC-AUC and classification reports validate model performance
🙌 Acknowledgments
Dataset courtesy of Kaggle (HAM10000)
Inspired by applications of AI in dermatology
🚀 Future Enhancements
Deployment as a web or mobile application
Use of advanced architectures like ResNet or EfficientNet
Integration with patient history for context-aware diagnosis
👤 Author
Achyuth Kumar Miryala
Master's in Data Science | University of North Texas
Python | Machine Learning | Deep Learning | Data Analysis

Feel free to connect or raise an issue if you have suggestions or questions!

This project is part of my portfolio to showcase deep learning applications in the healthcare domain.
