import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 10  # Adjust this based on the number of food categories


# Load and preprocess the dataset
def load_dataset(images_path, labels_file):
    labels_df = pd.read_csv(labels_file)
    images = []
    food_labels = []
    calorie_labels = []

    for index, row in labels_df.iterrows():
        img_path = os.path.join(images_path, row['image'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0  # Normalize the image
        images.append(img)
        food_labels.append(row['category'])
        calorie_labels.append(row['calories'])

    images = np.array(images)
    food_labels = np.array(food_labels)
    calorie_labels = np.array(calorie_labels)
    return images, food_labels, calorie_labels


# Load dataset
images, food_labels, calorie_labels = load_dataset('dataset/images', 'dataset/labels.csv')

# Convert food labels to categorical
food_labels = to_categorical(food_labels, NUM_CLASSES)

# Split into train and test sets
X_train, X_test, y_train_food, y_test_food, y_train_calories, y_test_calories = train_test_split(
    images, food_labels, calorie_labels, test_size=0.2, random_state=42
)

# Build the CNN model for food classification and calorie estimation
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Convolutional base
x = Conv2D(32, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Dense layers
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output for food classification
food_output = Dense(NUM_CLASSES, activation='softmax', name='food_output')(x)

# Output for calorie estimation
calorie_output = Dense(1, activation='linear', name='calorie_output')(x)

# Combine the model
model = Model(inputs=input_img, outputs=[food_output, calorie_output])

# Compile the model
model.compile(
    optimizer='adam',
    loss={'food_output': 'categorical_crossentropy', 'calorie_output': 'mse'},
    metrics={'food_output': 'accuracy', 'calorie_output': 'mae'}
)

# Train the model
history = model.fit(
    X_train,
    {'food_output': y_train_food, 'calorie_output': y_train_calories},
    epochs=EPOCHS,
    validation_data=(X_test, {'food_output': y_test_food, 'calorie_output': y_test_calories}),
    batch_size=BATCH_SIZE
)

# Evaluate the model
loss, food_loss, calorie_loss, food_accuracy, calorie_mae = model.evaluate(X_test,
                                                                           {'food_output': y_test_food,
                                                                            'calorie_output': y_test_calories})
print(f'Test Food Accuracy: {food_accuracy * 100:.2f}%')
print(f'Test Calorie MAE: {calorie_mae:.2f} calories')

# Plot training & validation accuracy values
plt.plot(history.history['food_output_accuracy'])
plt.plot(history.history['val_food_output_accuracy'])
plt.title('Model food classification accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save the model
model.save('food_recognition_calorie_estimation_model.h5')
