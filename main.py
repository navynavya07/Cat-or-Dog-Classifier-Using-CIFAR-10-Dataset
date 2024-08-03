import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Filter out only cats (class 3) and dogs (class 5) from CIFAR-10 dataset
cat_dog_indices_train = np.where((y_train == 3) | (y_train == 5))[0]
cat_dog_indices_test = np.where((y_test == 3) | (y_test == 5))[0]

x_train = x_train[cat_dog_indices_train]
y_train = y_train[cat_dog_indices_train]
x_test = x_test[cat_dog_indices_test]
y_test = y_test[cat_dog_indices_test]

# Convert labels: cats (3) to 0, dogs (5) to 1
y_train = np.where(y_train == 3, 0, 1)
y_test = np.where(y_test == 3, 0, 1)

# Normalize the images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification (cat or dog)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Streamlit UI
st.title("Cat or Dog Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    st.write(f"Prediction: {label} (Confidence: {prediction:.2f})")

    # For more detailed prediction, show a confidence score
    st.write(f"Confidence score: {prediction:.2f}")
