# 🦷 Oral Disease Classification

An end-to-end deep learning project for classifying oral diseases from images. Three pretrained architectures were trained and compared, with the best-performing model deployed as an interactive **Streamlit** web app.

## 📌 Overview

The model classifies oral images into **6 categories**:

- Calculus
- Caries
- Gingivitis
- Hypodontia
- Mouth Ulcer
- Tooth Discoloration

## 🧠 Models Compared

Three pretrained CNN architectures were fine-tuned and evaluated on the same test set:

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| ResNet18 | 98.05% | 97.62% | 98.05% |
| **EfficientNet-B0** | **99.61%** | **99.48%** | **99.61%** |
| MobileNetV3-Small | 97.86% | 96.91% | 97.87% |

**EfficientNet-B0** achieved the best performance across all metrics and was selected as the final deployed model.

## 🚀 Deployment

The final model is deployed using **Streamlit**, allowing users to:

- Upload one or multiple oral images (`.jpg`, `.jpeg`, `.png`)
- Get an instant prediction with a confidence score
- View the full probability breakdown across all 6 classes

## 🛠️ Tech Stack

- **PyTorch** / **torchvision** — model training and inference
- **EfficientNet-B0** — final architecture (ImageNet pretrained, fine-tuned)
- **Streamlit** — web app deployment
- **PIL** — image preprocessing

## 📂 Project Structure

```
├── app.py                          # Streamlit application
├── efficientnet_b0_best.pth        # Trained model checkpoint
├── requirements.txt                # Project dependencies
└── README.md
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## 🖼️ How It Works

1. Upload one or more oral images through the web interface.
2. Click **Classify** for the image you want to predict.
3. The model preprocesses the image (resize to 224×224, normalize using ImageNet statistics) and runs inference.
4. The predicted class, confidence score, and full probability distribution are displayed.

## 📊 Results Summary

- EfficientNet-B0 consistently outperformed ResNet18 and MobileNetV3-Small across accuracy, macro F1, and weighted F1.
- The model shows strong, balanced performance across all 6 disease categories.

## 🔮 Future Improvements

- Expand the dataset with more diverse, higher-quality images to further reduce misclassification between visually similar conditions.
- Add Grad-CAM visualizations to explain model predictions.
- Package the app with Docker for easier deployment.

## 🙏 Acknowledgements

- **Training Company:** [Instant Software Solutions](https://www.linkedin.com/company/instantsoftwaresolution/)
- **Instructor:** [Jana Hatem](https://www.linkedin.com/in/janahatem/)
- **Mentor:** [Nourhan Nafea](https://www.linkedin.com/in/nourhan-nafea-6a1773260/)

## 📄 License

This project is for educational purposes as part of an AI Diploma training program.
