import streamlit as st
import os
from tf_keras_vis.utils.scores import CategoricalScore
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from vis.utils import utils

# Load model
# model = tf.keras.models.load_model('cat_dog_classifier_model.h5')
model = MobileNetV2(weights='imagenet')

# # Swap last layer with linear layer
# layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
# model.layers[-1].activation = tf.keras.activations.linear
# model = utils.apply_modifications(model)

dog_prefix = 'n020'  # Mã lớp chó bắt đầu bằng 'n020'
dog_prefix_3 = 'n0210'
dog_prefix_2 = 'n0211'
cat_prefix = 'n021'  # Mã lớp mèo bắt đầu bằng 'n021'

# Streamlit App
st.title("Cat vs Dog Classifier with Saliency Map and GRAD-CAM")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    img = load_img(uploaded_file, target_size=(224, 224))
    x = img_to_array(img)  # Numpy array với shape (224, 224, 3)
    x = np.expand_dims(x, axis=0)  # Thêm batch dimension => (1, 224, 224, 3)
    x = preprocess_input(x)  # Tiền xử lý ảnh theo chuẩn MobileNetV2

    # Display the original image
    # st.image(img, caption="Uploaded Image", use_column_width=True)

    # Model prediction
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Xác định nhãn dự đoán (chỉ "Dog" hoặc "Cat" nếu phù hợp)
    predicted_category = []
    for pred in decoded_preds:
        if pred[0][:4] == dog_prefix or pred[0][:5] == dog_prefix_2 or pred[0][:5] == dog_prefix_3:  # Kiểm tra 4 ký tự đầu tiên
            predicted_category.append("Dog")
        elif pred[0][:4] == cat_prefix:  # Kiểm tra 4 ký tự đầu tiên
            predicted_category.append("Cat")
        else:
            predicted_category.append("Not a Cat or Dog")
    # st.write(decoded_preds)
    # Tạo danh sách 3 dự đoán top đầu với định dạng: Lớp và xác suất
    top_preds = [f"{pred[1]} ({pred[2] * 100:.2f}%) ({predicted_category[i]})" for i, pred in enumerate(decoded_preds)]
    predicted_label = "\n".join(top_preds)
    # Ghi tiêu đề ảnh với 3 dự đoán
    # predicted_label \
    #     = (f'Predictions:\n1. {top_preds[0]} : {predicted_category[0]}'
    #        f'\n2. {top_preds[1]} : {predicted_category[1]}'
    #        f'\n3. {top_preds[2]} : {predicted_category[2]}')
    # Map class index to labels (customize this based on your model)

    # Generate saliency map
    score = CategoricalScore([np.argmax(preds)])
      # Use predicted class for saliency

    # Thay đổi activation của lớp cuối cùng thành tuyến tính để tính gradient
    layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
    model.layers[-1].activation = tf.keras.activations.linear
    model = utils.apply_modifications(model)

    # Tính Saliency Map
    saliency = Saliency(model, clone=False)
    saliency_map = saliency(score, x, smooth_samples=20, smooth_noise=0.2)
    saliency_map = normalize(saliency_map)

    # Tính Grad-Cam
    gradcam = Gradcam(model, clone=False)
    gradcam_map = gradcam(score, x, penultimate_layer=-1)
    gradcam_map = normalize(gradcam_map)


    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    ax[0].imshow(img)
    # ax[0].set_title("Original Image")
    ax[0].set_title(f"{predicted_label}", fontsize=10)
    ax[0].axis("off")

    saliency = ax[1].imshow(saliency_map[0], cmap='viridis')
    cbar = fig.colorbar(saliency, ax=ax[1], fraction=0.046)  # Liên kết thanh màu với saliency map
    cbar.set_label('', rotation=270, labelpad=15)  # Nhãn cho thanh màu
    # ax[1].set_title(f"{predicted_label}", fontsize=10)
    ax[1].set_title("Saliency Map")
    ax[1].axis("off")

    gradcam_overlay = ax[2].imshow(img, alpha=0.6)
    gradcam_overlay = ax[2].imshow(gradcam_map[0], cmap='jet', alpha=0.4)
    cbar_1 = fig.colorbar(gradcam_overlay, ax=ax[2], fraction=0.046)  # Liên kết thanh màu với saliency map
    cbar_1.set_label('', rotation=270, labelpad=15)  # Nhãn cho thanh màu
    ax[2].set_title("Grad-CAM")
    ax[2].axis("off")

    st.pyplot(fig)
