# Cotton Plant Disease Classification Using Transfer Learning

##  Project Overview
This project aims to classify cotton plant leaves into 6 categories: **Target Spot**, **Powdery Mildew**, **Healthy Leaf**, **Bacterial Blight**, **Army Worm**, and **Aphids** using **Transfer Learning** with the **MobileNetV2** model. The model can help detect cotton diseases early and aid in crop management.

## 📊 Dataset
- **Target Spot**
- **Powdery Mildew**
- **Healthy Leaf**
- **Bacterial Blight**
- **Army Worm**
- **Aphids**

Images are taken from a dataset of cotton plant leaves with each class representing a different condition.

## 🔧 Technologies & Libraries Used
- **Python**, **NumPy**, **Pandas** for data processing
- **TensorFlow**, **Keras** for model implementation
- **Matplotlib**, **Seaborn** for data visualization
- **Scikit-learn** for evaluation

## 🔄 Preprocessing Steps
1. Loaded and labeled images for each disease.
2. Resized the images to **224×224 pixels** to match MobileNetV2's input dimensions.
3. Split the data into **training (70%)** and **testing (30%)**.

## 🤖 Model Architecture
1. **Input Layer**: Images resized to **224×224×3**.
2. **MobileNetV2**: Pre-trained model used for feature extraction (weights frozen).
3. **Dense Layers**: For classification after feature extraction.
4. **Output Layer**: **Softmax** activation for multi-class classification.

## 📈 Results & Evaluation
- **Accuracy**: Evaluated model accuracy on test data
- **Confusion Matrix**: Visualized performance
- **Classification Report**: Detailed precision, recall, and F1-score for each class
