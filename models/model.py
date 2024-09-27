import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

# EfficientNet model
def build_efficientnet_model(num_classes):
    inputs = Input(shape=(224, 224, 3))
    base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    base_model.trainable = True

    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ResNet model
def build_resnet_model(num_classes):
    inputs = Input(shape=(224, 224, 3))
    base_model = ResNet50(include_top=False, input_tensor=inputs, weights='imagenet')
    base_model.trainable = True

    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Vision Transformer model (ViT)
def build_vit_model(num_classes):
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from tensorflow.keras.applications import VGG16

    # Here, we can replace VGG16 with an actual ViT implementation or a library that supports it
    # For demonstration, I will use a placeholder approach
    # Note: You need to implement or import a ViT architecture from a compatible library
    inputs = Input(shape=(224, 224, 3))
    base_model = VGG16(include_top=False, input_tensor=inputs, weights='imagenet')
    base_model.trainable = True

    x = Flatten()(base_model.output)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
