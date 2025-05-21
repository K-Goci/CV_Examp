import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

# Từ điển mô hình
models = {
    "Model VGG": "vgg_model.h5",
    "Model InceptionV3": "tool_recognition_inception_model.h5",
    "Model DenseNet": "tool_recognition_densenet_model.h5"
}

class_labels = ['Bút bi', 'Bút chì', 'Bút máy', 'Gọt', 'Thước', 'Tẩy']

# Load model mặc định
current_model_path = models["Model VGG"]
model = load_model(current_model_path)

def update_model(selected_model_name):
    global model, current_model_path
    current_model_path = models[selected_model_name]
    try:
        model = load_model(current_model_path)
        result_label.config(text="Mô hình đã được cập nhật thành: " + selected_model_name)
    except Exception as e:
        result_label.config(text=f"Lỗi: {str(e)}")

def preprocess_image(img_path):
    # Chọn kích thước đầu vào dựa trên mô hình
    target_size = (299, 299) if current_model_path == models["Model InceptionV3"] else (224, 224)
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((500, 400))  # Kích thước hiển thị phù hợp khung 900x700
    img_tk = ImageTk.PhotoImage(img)
    image_canvas.config(image=img_tk)
    image_canvas.image = img_tk

    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    result_label.config(text=f"Kết quả :\n{predicted_label}")

# Giao diện ứng dụng chính
root = tk.Tk()
root.title("Nhận diện dụng cụ văn phòng phẩm.py")
root.geometry("600x400")
root.resizable(False, False)

# Tiêu đề nhỏ gọn
title_label = tk.Label(root, text="Nhận diện dụng cụ văn phòng phẩm", font=("Courier", 14, "bold"))
title_label.pack(pady=5)

# Menu chọn mô hình
model_frame = tk.Frame(root)
model_frame.pack()
model_var = tk.StringVar(root)
model_var.set("Model VGG")
model_dropdown = tk.OptionMenu(model_frame, model_var, *models.keys(), command=update_model)
model_dropdown.config(font=("Courier", 10))
model_dropdown.pack()

# Main frame
main_frame = tk.Frame(root)
main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# Khung ảnh nhỏ lại
image_box = tk.Frame(main_frame, bd=2, relief="solid", width=300, height=250)
image_box.pack(side=tk.LEFT, padx=5)
image_box.pack_propagate(False)

image_canvas = tk.Label(image_box, text="Nhập hình ảnh", font=("Courier", 10))
image_canvas.pack(expand=True)

# Khung kết quả nhỏ lại
result_frame = tk.Frame(main_frame, bd=2, relief="solid", width=180, height=250)
result_frame.pack(side=tk.LEFT, padx=5)
result_frame.pack_propagate(False)

result_label = tk.Label(result_frame, text="Kết quả :", font=("Courier", 10), anchor="nw", justify="left")
result_label.pack(padx=5, pady=5, anchor="nw")

# Nút thao tác
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

predict_button = tk.Button(button_frame, text="Dự đoán", font=("Courier", 10), command=predict_image)
predict_button.pack(side=tk.LEFT, padx=10)

exit_button = tk.Button(button_frame, text="Thoát", font=("Courier", 10), command=root.quit)
exit_button.pack(side=tk.LEFT, padx=10)

# Chỉnh ảnh hiển thị phù hợp hơn
def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((280, 230))  # Giới hạn nhỏ hơn
    img_tk = ImageTk.PhotoImage(img)
    image_canvas.config(image=img_tk)
    image_canvas.image = img_tk

    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    result_label.config(text=f"Kết quả :\n{predicted_label}")

root.mainloop()