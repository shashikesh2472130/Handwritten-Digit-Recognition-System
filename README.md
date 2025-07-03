# Handwritten-Digit-Recognition-System

✍ Handwritten Digit Recognition using CNN and Streamlit
This project is a web-based handwritten digit recognition system built with a trained Convolutional Neural Network (CNN) model and Streamlit. It allows users to either draw a digit or upload a digit image, and the model will predict the digit in real-time.

📌 Features
Draw a digit (0–9) on a canvas and get prediction.
Upload an image of a digit for recognition.
Real-time prediction using a trained CNN model.
Clean and interactive web interface using Streamlit.
🧠 Model Information
Dataset Used: MNIST Handwritten Digits Dataset
Input Size: 28x28 grayscale images
Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Saved Model Format: .h5 (HDF5 format)
Epochs Used for Training: (Specify actual value, e.g., 5 or 10)
🛠 Tech Stack and Libraries
Python 3.x
TensorFlow – Deep learning framework
Keras – High-level neural network API
OpenCV – Image preprocessing
Streamlit – UI for web application
NumPy – Array and matrix operations
streamlit-drawable-canvas – Drawing canvas widget in Streamlit
📁 Folder Structure
handwritten-digit-recognition/ ├── app.py # Main Streamlit app ├── hand_written.h5 # Trained CNN model file ├── requirements.txt # List of dependencies ├── README.md # Project documentation

yaml Copy Edit

📦 Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/handwritten-digit-recognition.git cd handwritten-digit-recognition 2. Create a virtual environment (recommended) python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate 3. Install dependencies Install all required Python libraries using requirements.txt.

pip install -r requirements.txt If requirements.txt is not available, install manually:

pip install streamlit tensorflow opencv-python numpy streamlit-drawable-canvas 🚀 Running the Application Once dependencies are installed and the model (hand_written.h5) is in place:

streamlit run app.py This will start a local server (usually at http://localhost:8501) where you can use the digit recognizer.

🧪 Example Use Case Draw a number (e.g., 5) using the canvas provided.

Click the "Predict" button.

The app will preprocess your drawing and display the predicted digit.

You can also upload an image:

Must be grayscale and preferably 28x28 pixels.

The app will process and show the predicted digit.

📷 Screenshot (Upload a screenshot of your app and embed it here) Example:

🔐 License This project is licensed under the MIT License.

🙋‍♂ Author Your Name GitHub: @yourusername

✅ To Do Add support for color image preprocessing

Improve UI with real-time prediction while drawing

Add training notebook for transparency
