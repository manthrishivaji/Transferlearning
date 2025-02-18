import gradio as gr
import requests

API_URL = "http://backend:8000/predict/"  # i changed here to backend , kepping in mind of docker images, or use 127.0.0.1 for local use

def classify_image(image_path):
    """Send image file directly to FastAPI backend and return prediction."""
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        return response.json().get("Predicted Class", "Unknown Class")
    else:
        return f"Error: {response.status_code}, {response.text}"

# Create Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),  # Sends file path instead of loading it
    outputs=gr.Textbox(label="Predicted Class"),
    title="Image Classifier",
    description="Upload an image to classify it using an AlexNet model deployed with FastAPI."
)

# Launch the Gradio App
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)  
