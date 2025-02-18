
import os
from huggingface_hub import hf_hub_download



# create model directory if not exists
os.makedirs("model", exist_ok=True)

# Define model save path
local_model_path = "model/alexnet_model.pth"

# Check if model already exists
if not os.path.exists(local_model_path):
    print("Downloading model from Huggingface Hub...")
    # model_path = hf_hub_download(repo_id="Wolverine001/Alexnet_TransferLearning",filename="Alexnet-finetuned.pth",local_dir="./model")
    model_path = hf_hub_download(repo_id="Wolverine001/Alexnet_TransferLearning",filename="Alexnet-finetuned.pth")
    os.rename(model_path, local_model_path)
    print(f"Model downloaded and saved to {local_model_path}")

else:
    print("Model Already Exists.")