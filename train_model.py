import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers

# Paths
train_path = "https://mriimages30709.s3.us-east-1.amazonaws.com/dataset/Training/"
test_path = "https://mriimages30709.s3.us-east-1.amazonaws.com/dataset/Testing/"

# Data generator
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_path, target_size=(150, 150), class_mode='categorical')
test_data = datagen.flow_from_directory(test_path, target_size=(150, 150), class_mode='categorical')

# Load VGG16 without the top (classification) layer, and exclude the fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the convolutional layers of VGG16 to retain pre-trained features
base_model.trainable = False

# Model architecture
model = Sequential([
    base_model,  # Add VGG16 base model
    Flatten(),   # Flatten the output to feed into fully connected layers
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(128, activation='relu'),  # Fully connected layer with 128 units
    Dense(4, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=test_data, epochs=10)

# Save the model
model.save('brain_tumor_vgg16_model1.h5')
