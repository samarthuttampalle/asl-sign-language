import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
import os

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = "/content/asl_data/asl_alphabet_train/asl_alphabet_train"

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Step 1: Data Generators with explicit parameters
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False  # Important for ASL - don't flip hands!
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Create generators
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    seed=123,
    color_mode='rgb',  # Explicitly specify RGB
    interpolation='bilinear'
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    seed=123,
    color_mode='rgb',  # Explicitly specify RGB
    interpolation='bilinear'
)

print(f"Found {train_generator.samples} training images")
print(f"Found {val_generator.samples} validation images")
print(f"Number of classes: {train_generator.num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# Step 2: Create Model with explicit input shape
print("Creating model...")

# Create base model with explicit input shape
base_model = EfficientNetB0(
    include_top=False, 
    weights='imagenet', 
    input_shape=(224, 224, 3)  # Explicitly define input shape
)
base_model.trainable = False

# Create the full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary to verify architecture
print("\nModel Summary:")
model.summary()
print(f"\nModel input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Test model with dummy data to ensure it works
print("\nTesting model with dummy input...")
dummy_input = tf.random.normal((1, 224, 224, 3))
dummy_output = model(dummy_input)
print(f"Dummy test successful! Output shape: {dummy_output.shape}")

# Step 3: Training with callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Calculate steps
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, val_generator.samples // BATCH_SIZE)

print(f"\nTraining parameters:")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Batch size: {BATCH_SIZE}")

# Train the model
print("\nStarting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,  # Increased epochs
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
print("\nSaving final model...")
model.save("sign_language_model_fixed.keras")

# Also save in SavedModel format as backup
model.save("sign_language_model_savedmodel", save_format='tf')

print("Training completed!")
print("Models saved as:")
print("- sign_language_model_fixed.keras")
print("- sign_language_model_savedmodel/")

# Print final model info
print(f"\nFinal model input shape: {model.input_shape}")
print(f"Final model output shape: {model.output_shape}")
print(f"Number of classes: {train_generator.num_classes}")

# Save class labels for reference
import json
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)
print("Class labels saved to class_labels.json")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("Training history saved as training_history.png")