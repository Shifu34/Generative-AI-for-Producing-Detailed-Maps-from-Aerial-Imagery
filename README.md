
# Generative AI for Producing Detailed Maps from Aerial Imagery

## Project Overview
This project demonstrates the use of Generative AI to automate the transformation of satellite imagery into high-resolution, detailed maps. Leveraging the Pix2Pix conditional Generative Adversarial Network (GAN), the project addresses the manual, labor-intensive process of map generation, providing a scalable and efficient solution for geospatial data analysis.

Applications of this work span critical domains such as:
- Urban Planning
- Disaster Management
- Environmental Monitoring

The web application developed allows users to upload satellite images and receive corresponding map outputs. It was implemented using Flask for the backend.

---

## Features
- **Automated Map Generation**: Converts satellite images into detailed maps using a Pix2Pix GAN model.
- **High Accuracy**: Achieved a Structural Similarity Index (SSIM) score of 0.91 and pixel accuracy of 85%.
- **Real-World Applications**: Addresses challenges like urban mapping, disaster response, and ecological monitoring.
- **Scalable**: Capable of processing large datasets and generating maps for diverse geographic regions.
- **User-Friendly Interface**: Flask-based web app for easy interaction and map generation.

---

## Methodology
### 1. Dataset
- **Sources**: DeepGlobe and SpaceNet datasets.
- **Samples**: 20,000 paired satellite and map images.
- **Preprocessing**:
  - Normalization and resizing to 256×256 pixels.
  - Data augmentation (flipping, rotation, scaling).

### 2. Model Architecture
- **Generator**: U-Net for image synthesis.
- **Discriminator**: PatchGAN for local feature evaluation.
- **Loss Function**: Combination of adversarial loss and L1 loss.

### 3. Training
- **Frameworks and Tools**: TensorFlow/Keras, NumPy, OpenCV, and Matplotlib.
- **Epochs**: Trained for 150 epochs.
- **Evaluation Metrics**:
  - SSIM: Quantifies structural similarity.
  - Pixel Accuracy: Measures pixel-level fidelity.

### 4. Web Application
- **Backend**: Flask for serving the model.
- **Functionality**:
  - Upload satellite images.
  - Generate and display map outputs.

---

## Results
### Quantitative Metrics
- **SSIM**: 0.91
- **Pixel Accuracy**: 85%

### Observations
- High fidelity in urban regions with distinct features (roads, buildings).
- Challenges in dense vegetation or shadowed areas.
- Improved learning dynamics reflected by steady decrease in adversarial loss.

### Comparisons
| Model           | SSIM  | Pixel Accuracy |
|-----------------|-------|----------------|
| Pix2Pix         | 0.91  | 85%            |
| CycleGAN        | 0.85  | 80%            |
| Autoencoder     | 0.78  | 75%            |

---

## System Requirements
- **Hardware**: Minimum 16GB RAM, RTX 3060 GPU.
- **Software**: Python 3.x, Flask, TensorFlow/Keras, NumPy, OpenCV.

---

## Challenges and Solutions
### Data Imbalance
- **Issue**: Underrepresentation of certain regions (e.g., rural areas).
- **Solution**: Applied data augmentation and generated synthetic examples for minority classes.

### Computational Constraints
- **Issue**: High resource requirements for training.
- **Solution**: Utilized GPU optimizations, mixed precision training, and checkpointing.

### Boundary Detection
- **Issue**: Difficulty in capturing fine details like thin roads and water bodies.
- **Solution**: Incorporated edge detection in preprocessing and tailored loss functions for critical regions.

---

## Future Work
- Expand the dataset to include more diverse geographies.
- Experiment with alternative architectures like CycleGAN for unpaired data.
- Integrate additional geospatial features (e.g., elevation, land-use data).

---

## How to Run the Application
1. **Clone the Repository**:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```
2. **Download the Trained Models From Google Drive**:
   [Download Trained Models](https://drive.google.com/drive/folders/1r2WtumEbUOjT1ZPZeKeUt9vSNsHYAw67?usp=sharing)

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask App**:
   ```bash
   python app.py
   ```
5. **Access the Application**:
   Open a browser and go to `http://127.0.0.1:5000/`.

---

## Acknowledgments
This project was completed by:
- **Hasnain Jafri(Report and App)**
- **Shafqat Mehmood(Model Training and Testing)**
- **Waiz(Dataset Preprocessing)**

We thank the creators of the Pix2Pix framework and the contributors of the DeepGlobe and SpaceNet datasets for enabling this work.

---

## References
- Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-image translation with conditional adversarial networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
- Yang, W., Zhao, X., & Zhang, L. (2020). *Learning to see through fog: Generative adversarial networks for high-resolution maps from aerial images*. IEEE Transactions on Geoscience and Remote Sensing.
