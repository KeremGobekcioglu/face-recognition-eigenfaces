# Face Recognition with Eigenfaces

This project implements a face recognition system using the **Eigenfaces method**, based on Principal Component Analysis (PCA).

It includes:
- Image preprocessing and normalization
- Extraction of Eigenfaces using PCA
- Projection of both training and test images into eigenface space
- Classification using **Euclidean distance** between projected vectors

## 📂 Datasets

### ✅ Included
- `olivetti_faces/`: The publicly available **Olivetti Faces Dataset**.

> 🧾 **Dataset Credit**  
This project uses the **Olivetti Faces Dataset**, originally created by **AT&T Laboratories Cambridge**.  
The dataset is freely available for research purposes and can be accessed through **scikit-learn**.

### ❌ Excluded
- `real_dataset/`: A custom dataset containing photos of the authors and friends was **not included** to protect privacy.
- Recognition results based on that dataset have also been omitted.

## 🚨 Performance Note

- The file `recognition_with_custom_dot_product_implementation.py` contains a slow, manual implementation of PCA-related operations and vector projection.
- It avoids using NumPy vectorization, making it much slower than the optimized version.
- For practical testing, use `recognition_with_np_dot.py`, which is significantly faster and more efficient.

---

## 📄 Project Report

📘 **[Face Recognition Using Eigenfaces (PDF)](./Face%20Recognition%20Using%20Eigenfaces.pdf)**  
The report covers:
- Background on PCA and Eigenfaces
- Dataset preparation and limitations
- Implementation details
- Experimental results and analysis

---

## 👥 Authors

- Kerem Göbekcioğlu  
- Kamil Duru

---

## 📁 File Overview

- `Preprocessing.py`: Loads and prepares face images
- `recognition_with_custom_dot_product_implementation.py`: Manual implementation of classification (slow)
- `recognition_with_np_dot.py`: Fast NumPy-based version
- `olivetti_faces/`: Public dataset folder

---

## 📝 License

MIT License © 2025 Kerem Göbekcioğlu & Kamil Duru
