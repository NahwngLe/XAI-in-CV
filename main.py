import shutil
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from vis.utils import utils

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Path đến ảnh test (bạn có thể chỉnh sửa để phù hợp với dữ liệu của mình)
testing_dog = "images/test/dog"
testing_cat = "images/test/cat"

# Load ảnh ngẫu nhiên từ thư mục chó
random_image = random.sample(os.listdir(testing_dog), 1)[0]
img_path = os.path.join(testing_dog, random_image)
img = load_img(img_path, target_size=(224, 224))  # Kích thước chuẩn cho MobileNetV2
print(img_path)
# Chuyển đổi ảnh sang định dạng numpy array
x = img_to_array(img)  # Numpy array với shape (224, 224, 3)
x = np.expand_dims(x, axis=0)  # Thêm batch dimension => (1, 224, 224, 3)
x = preprocess_input(x)  # Tiền xử lý ảnh theo chuẩn MobileNetV2

# Dự đoán lớp cho ảnh đầu vào
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]  # Lấy 3 lớp dự đoán hàng đầu
print("Predictions:", decoded_preds)

dog_prefix = 'n020'  # Mã lớp chó bắt đầu bằng 'n020'
cat_prefix = 'n021'  # Mã lớp mèo bắt đầu bằng 'n021'

# Xác định nhãn dự đoán (chỉ "Dog" hoặc "Cat" nếu phù hợp)
predicted_category = []
for pred in decoded_preds:
    if pred[0][:4] == dog_prefix:  # Kiểm tra 4 ký tự đầu tiên
        predicted_category.append("Dog")
    elif pred[0][:4] == cat_prefix:  # Kiểm tra 4 ký tự đầu tiên
        predicted_category.append("Cat")


# Tạo danh sách 3 dự đoán top đầu với định dạng: Lớp và xác suất
top_preds = [f"{pred[1]} ({pred[2]:.2f})" for pred in decoded_preds]

# Ghi tiêu đề ảnh với 3 dự đoán
predicted_label \
    = (f'Predictions:\n1. {top_preds[0]} : {predicted_category[0]}'
       f'\n2. {top_preds[1]} : {predicted_category[1]}'
       f'\n3. {top_preds[2]} : {predicted_category[2]}')

# Tùy chỉnh lớp đầu ra (Chọn lớp chó và mèo)
from tf_keras_vis.utils.scores import CategoricalScore
score = CategoricalScore([0])  # Không cần mã cụ thể cho saliency ở đây

# Thay đổi activation của lớp cuối cùng thành tuyến tính để tính gradient
layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
model.layers[-1].activation = tf.keras.activations.linear
model = utils.apply_modifications(model)

# Tính Saliency Map
saliency = Saliency(model, clone=False)
saliency_map = saliency(score, x, smooth_samples=20, smooth_noise=0.2)
saliency_map = normalize(saliency_map)

# Hiển thị ảnh gốc và Saliency Map
subplot_args = {
    'nrows': 1,
    'ncols': 2,
    'figsize': (12, 6),  # Kích thước lớn hơn để hiển thị tiêu đề dài
    'subplot_kw': {'xticks': [], 'yticks': []}
}

f, (ax1, ax2) = plt.subplots(**subplot_args)

# Hiển thị ảnh gốc và kết quả dự đoán
ax1.imshow(img)
ax1.set_title(f"{predicted_label}", fontsize=10)  # Kết hợp nhãn "Dog/Cat" và 3 dự đoán

# Hiển thị Saliency Map
ax2.imshow(saliency_map[0], cmap='Reds')
ax2.set_title("Saliency Map")

plt.tight_layout()
plt.show()
