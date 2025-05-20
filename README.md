# 🧠 Skin Cancer Detection using Deep Learning

This project presents a deep learning-based image classification system to detect and classify different types of **skin cancer** using dermatoscopic images. Leveraging **Convolutional Neural Networks (CNN)**, the goal is to provide an accurate and scalable solution to assist dermatologists in early diagnosis and treatment planning.

---

## 🌐 Project Overview

This pipeline is built using Python and TensorFlow/Keras, and trained on the **HAM10000** dataset — a large collection of dermatoscopic images labeled across 7 skin disease categories. The end-to-end pipeline includes:

- Data preprocessing & augmentation  
- Class imbalance handling using **SMOTE**  
- CNN model architecture for image classification  
- Performance evaluation with multiple metrics

---

## 📁 Dataset

- **Source**: [HAM10000 - Human Against Machine with 10,000 images](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Images**: ~10,000 dermatoscopic images
- **Classes**:
  - Melanocytic nevi
  - Melanoma
  - Benign keratosis-like lesions
  - Basal cell carcinoma
  - Actinic keratoses
  - Vascular lesions
  - Dermatofibroma
- **Metadata**: Patient age, sex, and lesion localization

---

## 🧠 Model Architecture

The CNN model is built using TensorFlow/Keras and consists of:

- `Conv2D` and `MaxPooling2D` layers  
- `Dropout` for regularization  
- `BatchNormalization` for training stability  
- `Dense` layers for final classification

Model training uses:
- `ImageDataGenerator` for image augmentation  
- `ReduceLROnPlateau` for adaptive learning rate tuning

---

## 📈 Performance Metrics

- **Confusion Matrix**
- **ROC-AUC Score**
- **Classification Report** (Precision, Recall, F1-score)
- Accuracy trends over epochs

---

## ⚙️ Project Workflow

### 🔧 Preprocessing
- Label encoding & one-hot encoding  
- Null value handling in metadata  
- SMOTE oversampling for minority classes

### 🖼️ Image Augmentation
- Horizontal/Vertical flips  
- Rotation, zoom, rescale

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/achyuthkumarmiryala/skin-cancer-detection.git
   cd skin-cancer-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - [Download from Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - Place the data files inside the `data/` directory.

4. **Run the Jupyter notebook**
   ```bash
   jupyter notebook project.ipynb
   ```

---

## 📊 Sample Outputs

- Visualizations of dataset distribution  
- Predicted class labels vs actual  
- ROC curves and per-class analysis

---

## 🚧 Future Enhancements

- Deploy as a web/mobile app using **Streamlit**, **Flask**, or **TensorFlow Lite**  
- Experiment with **ResNet** or **EfficientNet** architectures  
- Integrate clinical metadata for more context-aware predictions

---

## 👤 Author

**Achyuth Kumar Miryala**  
Master’s in Data Science | University of North Texas  
📍 Denton, TX  
📫 achyuthkumar286@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/achyuthkumarmiryala) | [GitHub](https://github.com/achyuthkumarmiryala)

---

## 🙌 Acknowledgments

- Dataset by **Kaggle** – [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
- Research inspired by AI applications in healthcare and dermatology

---

⭐ *If you find this project helpful or interesting, please consider giving it a star!*
