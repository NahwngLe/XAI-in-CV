# Streamlit Image Classification with Grad-CAM & Saliency

This project uses Streamlit, TensorFlow, and various visualization techniques such as Grad-CAM and Saliency to classify images and visualize the model's decision-making process. The app leverages MobileNetV2 for image classification.

## Prerequisites

Make sure you have Python 3.x installed. It is recommended to create a virtual environment to manage dependencies.

### 1. Create a virtual environment (optional but recommended)
For Windows:
```bash
python -m venv venv
```
### 2. Activate Virtual Environment
```bash
venv\Scripts\activate
```

### 3. Install required dependencies
```
pip install -r requirements.txt
```

### 4. Prepare the application
Make sure you have the app.py file ready in your project directory. This file contains the Streamlit app logic for image classification and visualization.

### 5. Running the app
To run the app, use the following command:
```
streamlit run app.py
```

## Project Structure:
```plaintext
├── app.py                  # Main Streamlit application file
├── requirements.txt        # File containing the list of dependencies
└── images/                 # Folder to store uploaded images (if required)
```

## Description
- The app uses **MobileNetV2** from TensorFlow for image classification.
- It employs Grad-CAM and Saliency maps from the tf-keras-vis library to visualize how the model makes predictions based on the input image.
- The Streamlit framework is used to build the web application interface, allowing users to upload an image, view the classification result, and see the visualizations.

## Notes
- The application uses MobileNetV2 as the pretrained model. You can customize the model or change the architecture if needed.
- Make sure the image uploaded is in a supported format (e.g., JPEG, PNG).
