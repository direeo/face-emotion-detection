# model_training.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json

# Config
DATA_DIR = os.path.join('database', 'fer2013')
SAVE_MODEL_PATH = 'face_emotionModel.h5'
LABEL_MAP_PATH = 'label_map.json'
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 25

# Emotion labels from your directory structure
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_dataset(base_dir):
    X = []
    y = []
    
    for label_idx, emotion in enumerate(EMOTION_LABELS):
        emotion_dir = os.path.join(base_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found: {emotion_dir}")
            continue
            
        print(f"Loading {emotion} images...")
        image_count = 0
        for img_file in os.listdir(emotion_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(emotion_dir, img_file)
            try:
                # Load and preprocess image
                img = load_img(img_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label_idx)
                image_count += 1
                if image_count % 100 == 0:
                    print(f"Processed {image_count} images from {emotion}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        # finished processing files for this emotion
        print(f"Finished loading {image_count} images from {emotion}")
                
    return np.array(X), np.array(y)

def preprocess(X):
    # normalize to [0,1]
    return X / 255.0

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Load training data
    print("Loading training data...")
    train_dir = os.path.join(DATA_DIR, 'train')
    X_train, y_train = load_dataset(train_dir)
    
    # Load test data
    print("Loading test data...")
    test_dir = os.path.join(DATA_DIR, 'test')
    X_test, y_test = load_dataset(test_dir)
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("No training or test data found!")
    
    # Preprocess
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=len(EMOTION_LABELS))
    y_test = to_categorical(y_test, num_classes=len(EMOTION_LABELS))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Build and train model
    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(EMOTION_LABELS))
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(SAVE_MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(SAVE_MODEL_PATH)
    print("Model saved to", SAVE_MODEL_PATH)
    
    # Save label map
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(EMOTION_LABELS, f)
    print("Label map saved to", LABEL_MAP_PATH)
    
    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest accuracy: {acc:.4f}, loss: {loss:.4f}")

if __name__ == '__main__':
    main()
