from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import io
import base64
from model import *

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


netG = torch.load('D:\\Datasets\\Detailed map from Aerial Images\\Trained Models\\pix2pix_generator_final.pt')

netG.eval()

# Preprocessing for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)), #All images are resized to a fixed dimension (256x256).
    transforms.ToTensor(), #The resized image is converted into a PyTorch tensor with pixel values in the range [0, 1].
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Check if an image was uploaded
        file = request.files.get("image")
        if not file:
            return render_template("index.html", error="No image uploaded!")
        
        try:
            # Load and preprocess the input image
            input_image = Image.open(file.stream).convert("RGB")
            real_image_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension

            # Generate the output image using the generator
            with torch.no_grad():
                generated_tensor = netG(real_image_tensor)

            # Convert tensors to NumPy arrays
            real_image_np = real_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            generated_image_np = generated_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

            # Rescale images from [-1, 1] to [0, 255]
            real_image_np = ((real_image_np + 1) * 127.5).astype(np.uint8)
            generated_image_np = ((generated_image_np + 1) * 127.5).astype(np.uint8)

            # Calculate SSIM between real and generated images
            ssim_value = ssim(real_image_np, generated_image_np, multichannel=True, win_size=3)

            # Convert images to base64 strings for HTML rendering
            real_image_base64 = image_to_base64(real_image_np)
            generated_image_base64 = image_to_base64(generated_image_np)

            return render_template(
                "result.html",
                real_image=real_image_base64,
                generated_image=generated_image_base64,
                ssim_value=f"{ssim_value:.4f}"
            )
        except Exception as e:
            return render_template("index.html", error=f"Error processing image: {str(e)}")
    return render_template("index.html", error=None)

def image_to_base64(image_np):
    """Convert a NumPy image array to a base64 string."""
    img = Image.fromarray(image_np)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
