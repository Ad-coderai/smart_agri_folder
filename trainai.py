import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# 1. SETUP PATHS
# Ensure this matches your folder name exactly
DATASET_PATH = r"C:\Users\Aditi\Desktop\smart_agri_folder" 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. DATA AUGMENTATION (Makes the AI smarter with fewer photos)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Uses 20% of your images for testing
)

# Load Training Data
train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load Validation Data
val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. BUILD THE "BRAIN" (MobileNetV2 is fast for laptops)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Keep the pre-trained knowledge frozen

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. START TRAINING
print(f"Training on {train_gen.num_classes} classes...")
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# 5. SAVE THE MODEL
model.save("plant_model.h5")
print("Success! 'plant_model.h5' has been created.")