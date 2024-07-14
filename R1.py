import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image


def load_data(image_dir):
    images = []
    labels = []

    for label, subdir in enumerate(['Lines', 'NoLines']):
        subdir_path = os.path.join(image_dir, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


image_dir = 'C:/Users/amogh/Amogh N Kotha RNSIT/3rdYear/6th Sem/Mini Project/LaneLineDetection/ImageDirectory'
images, labels = load_data(image_dir)

images = images / 255.0
labels = to_categorical(labels, num_classes=2)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(128, 128, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = build_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

model.load_weights('best_model.h5')

loss, accuracy = model.evaluate(val_generator)
print(f'Test accuracy: {accuracy}')

output_dir_lines = 'C:/Users/amogh/Amogh N Kotha RNSIT/3rdYear/6th Sem/Mini Project/LaneLineDetection/DetectedLines'
output_dir_no_lines = ('C:/Users/amogh/Amogh N Kotha RNSIT/3rdYear/6th Sem/Mini '
                       'Project/LaneLineDetection/DetectedNoLines')

os.makedirs(output_dir_lines, exist_ok=True)
os.makedirs(output_dir_no_lines, exist_ok=True)

predictions = model.predict(X_val)

for i, prediction in enumerate(predictions):
    label = np.argmax(prediction)
    img = (X_val[i] * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)

    if label == 0:
        pil_img.save(os.path.join(output_dir_lines, f'image_{i}.png'), 'PNG', icc_profile=None)
    else:
        pil_img.save(os.path.join(output_dir_no_lines, f'image_{i}.png'), 'PNG', icc_profile=None)
