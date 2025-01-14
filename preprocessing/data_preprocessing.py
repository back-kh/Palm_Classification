import os
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants for directories and image parameters
BASE_TRAIN_DIR = "Text-Classification/train_image"  # Directory for training images
BASE_TEST_DIR = "Text-Classification/test_image"    # Directory for test images
TRAIN_LABEL_PATH = "Text-Classification/gt_train.txt"  # Path to training labels
TEST_LABEL_PATH = "Text-Classification/gt_test.txt"    # Path to test labels
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 224, 224, 3  # Image size for input to the model (224x224, RGB)
BATCH_SIZE = 32  # Batch size for training and testing

# Function to load training and testing data
def load_data():
    """
    Load training and testing datasets from their respective label files.
    The files should contain image paths and corresponding text labels, separated by semicolons.
    """
    # Load training data from CSV file
    df_train = pd.read_csv(TRAIN_LABEL_PATH, sep=';', header=None, names=['img_path', 'text']).dropna().reset_index()
    # Load testing data from CSV file
    df_test = pd.read_csv(TEST_LABEL_PATH, sep=';', header=None, names=['img_path', 'text']).dropna().reset_index()
    
    return df_train, df_test

# Function to load class labels from a list file
def load_classes():
    """
    Load the list of class names (labels) from a text file.
    The text file should contain one class name per line.
    """
    with open("Text-Classification/list_class_name.txt", 'r') as file:
        vocab = [i_word for i_word in file.read().split("\n") if i_word != '']  # Remove empty lines
    return {label: idx for idx, label in enumerate(vocab)}  # Map each class label to a unique index

# Function to encode the text labels into numeric values for training
def encode_labels(df_train, df_test, target_dict):
    """
    Encode the textual class labels into numeric labels and prepare the image file paths.
    This function also associates each image path with its corresponding label.
    """
    # Initialize the LabelEncoder from scikit-learn
    le = preprocessing.LabelEncoder()
    le.fit(list(target_dict.keys()))  # Fit the label encoder on the class labels
    target_dict = {key: idx for idx, key in enumerate(le.classes_)}  # Map each label to a unique index
    
    # Encode the text labels into numeric labels for both train and test datasets
    df_train['label'] = df_train.text.apply(lambda x: target_dict[str(x)])  # Apply the encoding to the training set
    df_test['label'] = df_test.text.apply(lambda x: target_dict[str(x)])    # Apply the encoding to the test set
    
    # Update the image paths to include the base directory for training and test images
    df_train['img_path'] = df_train.img_path.apply(lambda x: os.path.join(BASE_TRAIN_DIR, x))
    df_test['img_path'] = df_test.img_path.apply(lambda x: os.path.join(BASE_TEST_DIR, x))
    
    return df_train, df_test  # Return the updated dataframes with encoded labels and full image paths

# Function to create data generators for training and validation
def create_data_generators(df_train, df_test):
    """
    Create Keras ImageDataGenerators for training and validation. This function also applies
    image augmentation techniques and normalization for better model generalization.
    """
    target_dict = load_classes()  # Load the class labels
    df_train, df_test = encode_labels(df_train, df_test, target_dict)  # Encode the labels and update the paths

    # Define a scaling function to normalize pixel values to the range [-1, 1]
    scalar = lambda img: img / 127.5 - 1  # Normalize the image pixels (0-255) to the range [-1, 1]

    # Create the training data generator with augmentation and normalization
    train_gen = ImageDataGenerator(
        preprocessing_function=scalar,  # Apply normalization
        shear_range=0.2,  # Random shearing transformation for data augmentation
        zoom_range=0.2,   # Random zooming transformation for data augmentation
        horizontal_flip=True,  # Random horizontal flipping for data augmentation
        rotation_range=10  # Random rotations (degrees) for data augmentation
    ).flow_from_dataframe(df_train, x_col='img_path', y_col='text',  # Use image paths and text labels
                          target_size=(IMG_HEIGHT, IMG_WIDTH),  # Resize images to the target size (224x224)
                          color_mode='rgb',  # Ensure images are in RGB format
                          class_mode='categorical',  # Use categorical classification
                          batch_size=BATCH_SIZE,  # Set the batch size
                          shuffle=True)  # Shuffle the dataset for training

    # Create the validation data generator (no augmentation, only normalization)
    valid_gen = ImageDataGenerator(preprocessing_function=scalar).flow_from_dataframe(
        df_test, x_col='img_path', y_col='text',  # Use image paths and text labels for the validation set
        target_size=(IMG_HEIGHT, IMG_WIDTH),  # Resize images to the target size
        color_mode='rgb',  # Ensure images are in RGB format
        class_mode='categorical',  # Use categorical classification
        batch_size=BATCH_SIZE,  # Set the batch size
        shuffle=False)  # No shuffling for validation set to maintain correct order

    return train_gen, valid_gen, target_dict  # Return the generators and the class-to-index dictionary
