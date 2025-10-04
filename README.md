# 🖼 Real vs Fake Face Classifier (VGG16 + Streamlit)

This project is a **Deep Learning app** built with **PyTorch** and **Streamlit** to classify faces as **Real** or **Fake** using **Transfer Learning (VGG16)**. The app is deployed on **Streamlit Cloud** for public access.

---

## 🚀 Features
- Transfer Learning with **VGG16** (PyTorch)
- Classifies images as **Real** or **Fake**
- Interactive **Streamlit UI**
- **Probability visualization** with Plotly
- Simple image upload and instant prediction

---

## 📂 Project Structure
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies
├── vgg16_model_weights.pth # Trained model weights (can be downloaded separately if >100MB)
└── README.md # Project description

---

## ⚙️ Installation & Setup
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/real-vs-fake-face-classifier.git
   cd real-vs-fake-face-classifier
pip install -r requirements.txt
streamlit run app.py
🛠 Tools & Libraries

Python

PyTorch

Torchvision

NumPy

Pillow

Plotly

Streamlit
Example Output

Upload a face image

Get prediction: Real or Fake

See probability distribution in a bar chart
👨‍💻 Author

Muhammad Hamza
