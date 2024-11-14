import tensorflow as tf

main_dir = "/content/Five_Faces"

IMG_SIZE = 224
BATCH_SIZE = 32

ds = tf.keras.preprocessing.image_dataset_from_directory(
    main_dir,
    labels="inferred",  # Automatically label images based on folder names
    label_mode="int",   # Labels as integers (good for multi-class classification)
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

for images, labels in ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in vgg_base.layers:
    layer.trainable = False

model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(5, activation='softmax') 
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(ds, epochs=10) 

model.save('five_faces_classification_model.h5')
