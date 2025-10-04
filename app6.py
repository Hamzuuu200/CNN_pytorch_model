import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import plotly.express as px

# -------------------------------
# 1. Load Model Function
# -------------------------------
@st.cache_resource
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 2)  # 2 classes: real/fake
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# -------------------------------
# 2. Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 3. Prediction Function
# -------------------------------
def predict_image(model, device, image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return predicted.item(), probs

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.set_page_config(page_title="VGG16 Image Classifier", layout="wide")

# Background style
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: black;
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 Model Info")

# 3 colored boxes with spacing
st.sidebar.markdown(
    """
    <div style="background-color:#FF5733; padding:15px; border-radius:10px; margin-bottom:15px;">
    <h4 style="color:white;">📖 Instructions</h4>
    <p style="color:white;">Upload an image of face to check if it is Real or Fake.</p>
    </div>

    <div style="background-color:#28A745; padding:15px; border-radius:10px; margin-bottom:15px;">
    <h4 style="color:white;">⚙ Model Details</h4>
    <p style="color:white;">Architecture: VGG16<br>Training: Transfer Learning<br>Dataset: 10K Images</p>
    </div>

    <div style="background-color:#007BFF; padding:15px; border-radius:10px;">
    <h4 style="color:white;">🛠 Tools Used</h4>
    <p style="color:white;">Python, PyTorch, Pandas, Streamlit, Plotly, Seaborn</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Page
st.title("🖼 VGG16 Image Classifier (Real vs Fake)")
st.write("This deep learning model is built using **Transfer Learning (VGG16)**. "
         "It classifies images as **Real** or **Fake** with trained weights. "
         "Powered by **PyTorch** and deployed with **Streamlit**.")

# Load model
weights_path = r"C:\Users\Ch. Hamza\OneDrive\Desktop\vgg16_model_weights.pth"
model, device = load_model(weights_path)

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    pred, probs = predict_image(model, device, image)
    class_names = ["Real", "Fake"]
    st.subheader(f"✅ Prediction: {class_names[pred]}")

    # Plotly Probability Graph
    fig = px.bar(x=class_names, y=probs,
                 labels={"x": "Class", "y": "Probability"},
                 title="Prediction Probabilities",
                 color=class_names)
    st.plotly_chart(fig)
    if uploaded_file is not None:
          image = Image.open(uploaded_file).convert("RGB")

          img_tensor = transform(image).unsqueeze(0).to(device)

          with torch.no_grad(): 
             outputs = model(img_tensor)
             _, predicted = torch.max(outputs, 1)

          classes = ["Real Face", "Fake Face"]
          label = classes[predicted.item()]

          st.image(image, caption=f"Prediction: {label}", use_column_width=True)

