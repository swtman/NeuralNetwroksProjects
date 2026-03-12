# Neural Networks & Deep Learning Projects

This repository contains a collection of three comprehensive projects focused on implementing, evaluating, and comparing various machine learning and deep learning architectures. 

Each project includes a Jupyter Notebook containing the code, alongside an in-depth PDF report detailing the theoretical methodology, experimental setups, and performance results.

## 📂 Projects Overview

### 1. MLPs & CNNs (`MLP_CNN/`)
This project explores the core architectures of deep learning for image classification using the **CIFAR-10** dataset
* **Key Topics:** Construction, training, and tuning of Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs).
* **Highlights:** Includes extensive hyperparameter tuning, model evaluation (accuracy & loss tracking), and performance comparisons across different network depths and configurations.

### 2. SVMs(`SVMs/`)
This project is focused in both binary and multiclass image classification usings SVMs and comparing different kernels on the **CIFAR-10** dataset
* **Key Topics:** Support Vector Machines (SVMs)
* **Highlights:** Employs Principal Component Analysis (PCA) for feature extraction and dimensionality reduction before classification. It evaluates the computational efficiency and accuracy trade-offs of using reduced feature spaces.

### 3. Autoencoders (`Autoencoder/`)
Focused on unsupervised learning and generative tasks using the **MNIST** dataset.
* **Key Topics:** Image reconstruction, latent space representation, and "Next Digit Generation."
* **Highlights:** Directly compares the image reconstruction capabilities of deep Autoencoders against traditional PCA. It also explores how varying the sizes of latent dimensions (e.g., 32 vs. 64 vs. 128) impacts the model's accuracy and generation loss.

## 🛠️ Technologies & Frameworks Used
* **Language:** Python
* **Deep Learning:** PyTorch
* **Machine Learning:** Scikit-Learn, cuML
* **Data Manipulation & Visualization:** NumPy, Matplotlib
* **Environment:** Jupyter Notebooks / Google Colab

## 📊 Reports & Presentations
For a deep dive into the experimental results, mathematical background, and performance metrics, please refer to the pdf files located within each respective project's directory. 

Additionally, the `ProjectsPresentation.pptx` file in the root directory provides a visual summary combining the findings of all three projects.