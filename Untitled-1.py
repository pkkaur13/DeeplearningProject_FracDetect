import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, ELU, Dense, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Nadam

from pathlib import Path
import imghdr
import seaborn as sns

train_directory = "D:\pkDL\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification/train"
validation_directory = "D:\pkDL\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification/val"
test_directory = "D:\pkDL\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification/test"

image_extensions = [".png", ".jpg"]
accepted_image_types = ["bmp", "gif", "jpeg", "png"]

def verify_images(directory, image_extensions, accepted_image_types):
    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            image_type = imghdr.what(file_path)
            if image_type is None:
                print(f"{file_path} is not an image")
            elif image_type not in accepted_image_types:
                print(f"{file_path} is a {image_type}, not accepted by TensorFlow")

print("Checking train directory...")
verify_images(train_directory, image_extensions, accepted_image_types)

print("\nChecking validation directory...")
verify_images(validation_directory, image_extensions, accepted_image_types)

print("\nChecking test directory...")
verify_images(test_directory, image_extensions, accepted_image_types)

import tensorflow as tf
from tensorflow import keras
from collections import Counter

def is_image_valid(image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return True
    except:
        return False

def load_dataset_and_count_labels(directory, image_size=(64, 64), batch_size=32):
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size
    )
    valid_paths = []
    for path in dataset.file_paths:
        if is_image_valid(path):
            valid_paths.append(path)
    
    dataset = tf.data.Dataset.from_tensor_slices(valid_paths)
    dataset = dataset.map(lambda x: (tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), channels=3), image_size), tf.strings.split(x, '/')[-2]))
    labels = []
    for _, label in dataset:
        labels.append(label.numpy())
    
    label_counts = Counter(labels)
    return dataset, label_counts

train_dataset, train_counts = load_dataset_and_count_labels(train_directory)
validation_dataset, validation_counts = load_dataset_and_count_labels(validation_directory)
test_dataset, test_counts = load_dataset_and_count_labels(test_directory)

def format_counts(label_counts):
    return {label.decode('utf-8'): count for label, count in label_counts.items()}

formatted_train_counts = format_counts(train_counts)
formatted_validation_counts = format_counts(validation_counts)
formatted_test_counts = format_counts(test_counts)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.barplot(x=list(formatted_train_counts.keys()), y=list(formatted_train_counts.values()))
plt.title('Train Dataset')
plt.xlabel('Class Labels')
plt.ylabel('Image Count')

plt.subplot(1, 3, 2)
sns.barplot(x=list(formatted_validation_counts.keys()), y=list(formatted_validation_counts.values()))
plt.title('Validation Dataset')
plt.xlabel('Class Labels')
plt.ylabel('Image Count')

plt.subplot(1, 3, 3)
sns.barplot(x=list(formatted_test_counts.keys()), y=list(formatted_test_counts.values()))
plt.title('Test Dataset')
plt.xlabel('Class Labels')
plt.ylabel('Image Count')

plt.tight_layout()
plt.show()

print("Train label counts:")
for label, count in formatted_train_counts.items():
    print(f"{label}: {count}")

print("\nValidation label counts:")
for label, count in formatted_validation_counts.items():
    print(f"{label}: {count}")

print("\nTest label counts:")
for label, count in formatted_test_counts.items():
    print(f"{label}: {count}")

import tensorflow as tf
from tensorflow import keras
import os
from tqdm import tqdm

def collect_valid_images_and_labels(directory):
    valid_images_and_labels = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.jpeg')):
                path = os.path.join(root, file)
                if is_image_valid(path):
                    label = os.path.basename(root)
                    valid_images_and_labels.append((path, label))
    return valid_images_and_labels

def create_dataset(images_and_labels, image_size=(224, 224), batch_size=32):
    paths, labels = zip(*images_and_labels)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    
    def load_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = image / 255.0
        return image, label
    
    dataset = dataset.map(load_preprocess_image)
    
    label_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(set(label for _, label in images_and_labels)), tf.range(len(set(label for _, label in images_and_labels)))),
        default_value=-1
    )
    
    def encode_labels(image, label):
        return image, label_lookup.lookup(label)
    
    dataset = dataset.map(encode_labels)
    dataset = dataset.batch(batch_size)
    
    return dataset, list(set(label for _, label in images_and_labels))

def extract_images_labels(dataset):
    images, labels = [], []
    for image_batch, label_batch in dataset:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

train_images_and_labels = collect_valid_images_and_labels(train_directory)
validation_images_and_labels = collect_valid_images_and_labels(validation_directory)
test_images_and_labels = collect_valid_images_and_labels(test_directory)

train_data, label_list = create_dataset(train_images_and_labels)
validation_data, _ = create_dataset(validation_images_and_labels)
test_data, _ = create_dataset(test_images_and_labels)

train_images, train_labels = extract_images_labels(train_data)
validation_images, validation_labels = extract_images_labels(validation_data)
test_images, test_labels = extract_images_labels(test_data)

print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Validation images shape:", validation_images.shape)
print("Validation labels shape:", validation_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)
print("Label list:", label_list)

import gc
del train_data
del validation_data
del test_data
gc.collect()

from keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

tf.config.optimizer.set_jit(True)

from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
extractor = Model(inputs=densenet_model.input, outputs=densenet_model.get_layer('conv5_block6_concat').output)

for layer in extractor.layers:
    layer.trainable = False

x = extractor.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=extractor.input, outputs=output_layer)

checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), epochs=12, batch_size=250, callbacks=[reduce_lr, checkpoint])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per epoch")
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per epoch")
plt.show()

from keras.models import load_model

loaded_model = load_model("model.keras")
results = loaded_model.evaluate(test_images, test_labels)

print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")

from sklearn.metrics import classification_report, roc_auc_score, precision_score, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

test_predictions = (loaded_model.predict(test_images) > 0.5).astype("int32")

print("Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=label_list))

precision = precision_score(test_labels, test_predictions)
print(f"Precision Score: {precision}")

auc_score = roc_auc_score(test_labels, test_predictions)
print(f"AUC-ROC Score: {auc_score}")

fpr, tpr, thresholds = roc_curve(test_labels, loaded_model.predict(test_images))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
ConfusionMatrixDisplay.from_predictions(test_labels, test_predictions, display_labels=label_list)
plt.title("Confusion Matrix")
plt.show()