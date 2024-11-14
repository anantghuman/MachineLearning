import tensorflow as tf

main_dir = "/content/Five_Faces"

IMG_SIZE = 224
BATCH_SIZE = 32

ds = tf.keras.preprocessing.image_dataset_from_directory(
    main_dir,
    labels="inferred",  
    label_mode="int",  
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

for images, labels in ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    
    layers.Dense(512, activation='relu'),
    
    layers.Dense(5, activation='softmax')  
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Print model summary
# model.summary()
