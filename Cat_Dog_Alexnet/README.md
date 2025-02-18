# Cat_Dog_Alexnet: Transfer Learning Project

## Overview
This project applies **transfer learning** using the **AlexNet** model to classify images of cats and dogs. The application is containerized using **Docker** and deployed with **Gradio** as the frontend for easy interaction.

## Project Structure
```
TRANSFERLEARNING/
│── Cat_Dog_Alexnet/
│   │── .gradio/               # Gradio-related files
│   │── backend/               # Backend for model inference
│   │   │── __pycache__/
│   │   │── backend/           # Backend modules
│   │   │── model/             # Model files
│   │   │── .dockerignore      # Docker ignore file for backend
│   │   │── app.py             # API service for model inference
│   │   │── Dockerfile         # Docker setup for backend
│   │   │── model.py           # Model definition and loading
│   │   │── requirements.txt   # Dependencies for backend
│   │── frontend/              # Gradio-based UI
│   │   │── flagged/           # Stores flagged inputs
│   │   │── .dockerignore      # Docker ignore file for frontend
│   │   │── Dockerfile         # Docker setup for frontend
│   │   │── gradio_app.py      # Gradio UI application
│   │   │── requirements.txt   # Dependencies for frontend
│   │── notebook/              # Jupyter Notebooks for experimentation
│   │── .gitignore             # Git ignore file
│   │── docker-compose.yml     # Docker Compose for running the app
│   │── README.md              # Project documentation
```

## Features
✅ Transfer learning with **AlexNet**
✅ **Gradio** frontend for easy model interaction
✅ **Dockerized** deployment with separate backend and frontend services
✅ Uses **Docker Compose** for managing services

## Setup Instructions
### Prerequisites
Ensure you have the following installed:
- [Python 3.9+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Clone the Repository
```sh
git clone https://github.com/your-username/Cat_Dog_Alexnet.git
cd Cat_Dog_Alexnet
```
### First download the model
```sh
cd backend
python model.py
```

### Running with Docker Compose
```sh
docker-compose up --build
```
This command builds and starts both the **backend** and **frontend** containers.

### Access the Application
Once running, you can access the Gradio UI at:
```
http://localhost:7860
```

## API Endpoints
The backend provides the following API endpoint:
```http
POST /predict
```
**Request:**
- Input: Image file (cat or dog)
- Output: Predicted label ("cat" or "dog") with confidence score

## Model Details
- **Architecture:** AlexNet (Pretrained on ImageNet)
- **Fine-Tuned Layers:** Fully connected layers
- **Dataset:** Dogs vs. Cats dataset

## Contribution
Feel free to open issues or submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License.

---
Made with ❤️ by [shivaji]

