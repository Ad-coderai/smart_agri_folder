import os
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = BASE_DIR
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "class_indices.json")


def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1.25),
        zoom_range=0.25,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen


def build_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def save_class_labels(generator):
    labels = [None] * len(generator.class_indices)
    for label, index in generator.class_indices.items():
        labels[index] = label

    with open(CLASS_INDEX_PATH, 'w', encoding='utf-8') as fh:
        json.dump(labels, fh, indent=2)


def train():
    train_gen, val_gen = build_data_generators()
    save_class_labels(train_gen)

    model = build_model(train_gen.num_classes)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print(f"Training on {train_gen.num_classes} classes for {EPOCHS} epochs...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Training complete. Best model saved to {MODEL_PATH}")


if __name__ == '__main__':
    train()
