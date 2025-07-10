# 📁 File: train_model.py
import os
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV3Large
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Cấu hình chung
KICH_THUOC_ANH = 224
BATCH_SIZE = 32
EPOCHS = 10  # 🔻 Giảm số epoch
LEARNING_RATE = 0.001

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tao_data_generators(duong_dan_du_lieu):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(duong_dan_du_lieu, 'train'),
        target_size=(KICH_THUOC_ANH, KICH_THUOC_ANH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(duong_dan_du_lieu, 'validation'),
        target_size=(KICH_THUOC_ANH, KICH_THUOC_ANH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(duong_dan_du_lieu, 'test'),
        target_size=(KICH_THUOC_ANH, KICH_THUOC_ANH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen

def xay_dung_mo_hinh(so_lop):
    base_model = MobileNetV3Large(weights='imagenet', include_top=False,
                                  input_shape=(KICH_THUOC_ANH, KICH_THUOC_ANH, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(so_lop, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def fine_tune_model(model, base_model, so_lop=30):
    base_model.trainable = True
    for layer in base_model.layers[:-so_lop]:
        layer.trainable = False
    model.compile(optimizer=Adam(LEARNING_RATE / 10),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def tao_callbacks():
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                          min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

def train():
    logger.info("Bắt đầu huấn luyện...")
    duong_dan = 'dataset_split'
    train_gen, val_gen, test_gen = tao_data_generators(duong_dan)
    so_lop = len(train_gen.class_indices)

    model, base_model = xay_dung_mo_hinh(so_lop)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=tao_callbacks()
    )

    model = fine_tune_model(model, base_model)
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=tao_callbacks()
    )

    # Lưu mô hình cuối cùng
    model.save("trashnet_model.keras", save_format="keras")
    logger.info("Đã lưu mô hình thành trashnet_model.keras")

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    train()
    print("\n✅ Đã hoàn tất huấn luyện và lưu mô hình.")
