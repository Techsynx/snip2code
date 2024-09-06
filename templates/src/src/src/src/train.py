# src/train.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import create_model

def train_model(train_dir, model_save_path='models/trained_model.h5'):
    model = create_model()
    
    # Data augmentation and preparation
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(train_dir, target_size=(256, 256), subset='training')
    validation_generator = datagen.flow_from_directory(train_dir, target_size=(256, 256), subset='validation')

    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    model.save(model_save_path)

if __name__ == "__main__":
    train_dir = 'path/to/your/dataset'
    train_model(train_dir)
