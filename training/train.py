import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.applications import DenseNet121

# Paths
train_dir = r'D:\python project\Corn_Leaf_Disease_Recognition_CNN_DenseNet\data\train'
val_dir = r'D:\python project\Corn_Leaf_Disease_Recognition_CNN_DenseNet\data\val'
model_save_path = r'D:\python project\Corn_Leaf_Disease_Recognition_CNN_DenseNet\model\corn_densenet_model.h5'
log_csv_path = 'training_log.csv'

# Parameters
img_height, img_width = 224, 224
batch_size = 16
num_classes = 4
initial_lr = 0.001
fine_tune_lr = 1e-5
num_epochs_phase1 = 15
num_epochs_phase2 = 10

# Class names
class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

def create_densenet_model(input_shape, num_classes, base_trainable=False, freeze_until=None, learning_rate=0.001):
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    
    base_model.trainable = base_trainable
    if freeze_until:
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=not base_trainable)
    x = GlobalAveragePooling2D(name='global_pool')(x)
    x = Dropout(0.3, name='top_dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
    )
    return model

# Create augmented generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

# Handle class imbalance
y_train = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Callbacks
callbacks = lambda log_file: [
    ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    CSVLogger(log_file, append=False)
]

# =====================
# Phase 1: Top Layers Only
# =====================
print("\n[Phase 1] Training top layers only (base model frozen)...")
model = create_densenet_model((img_height, img_width, 3), num_classes, base_trainable=False, learning_rate=initial_lr)
history1 = model.fit(
    train_generator,
    epochs=num_epochs_phase1,
    validation_data=val_generator,
    callbacks=callbacks('phase1_log.csv'),
    class_weight=class_weight_dict,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
    verbose=1
)

# =====================
# Phase 2: Fine-Tuning
# =====================
print("\n[Phase 2] Fine-tuning base model...")
model = create_densenet_model((img_height, img_width, 3), num_classes, base_trainable=True, freeze_until=400, learning_rate=fine_tune_lr)
print("Reloading weights from phase 1...")
model.load_weights(model_save_path)

history2 = model.fit(
    train_generator,
    epochs=num_epochs_phase2,
    validation_data=val_generator,
    callbacks=callbacks('phase2_log.csv'),
    class_weight=class_weight_dict,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
    verbose=1
)

# =====================
# Final Evaluation
# =====================
print("\n[Evaluation] Loading best model for final evaluation...")
model = tf.keras.models.load_model(model_save_path)

y_pred = model.predict(val_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes[:len(y_pred_classes)]

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_finetuned.png')
plt.show()

print(f"\nBest model saved to: {model_save_path}")
