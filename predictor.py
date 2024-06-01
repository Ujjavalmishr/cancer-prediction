import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

def load_and_predict(image_path, model, classes, img_size=128):
    img = cv2.imread(image_path)

    if img is not None:
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        img = np.reshape(img, (1, img_size, img_size, 3))

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_class_label = classes[predicted_class]
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title(f'Predicted Class: {predicted_class_label}')
        plt.axis('off')
        plt.show()

        return predicted_class_label

    else:
        print(f"Error loading image: {image_path}")
        return None

# Load the saved model
model_path = 'cancer_model.h5'  # Replace with the actual path to your saved model file
model = keras.models.load_model(model_path)
# Assuming you have the classes list
classes = ['lung_aca', 'lung_n', 'lung_scc']

# Load and predict using the function
image_path_to_predict = 'lungn1.jpeg'  # Replace with the actual path to your image
predicted_class = load_and_predict(image_path_to_predict, model, classes)
if predicted_class:
    print(f"Predicted Class: {predicted_class}")

    # Check if the predicted class is 'lung_aca'
    if predicted_class.lower() == 'lung_aca':
        print("Lung cancer (adenocarcinoma) is present.")
    elif predicted_class.lower() == 'lung_scc':
        print("Lung cancer (squamous cell carcinoma) is present.")
    else:
        print("No lung cancer detected in the image.")
