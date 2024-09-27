import os
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_TRAIN_DIR = "/content/Text-Classification/train_image"
BASE_TEST_DIR = "/content/Text-Classification/test_image"
TRAIN_LABEL_PATH = "/content/Text-Classification/gt_train.txt"
TEST_LABEL_PATH = "/content/Text-Classification/gt_test.txt"
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 224, 224, 3
BATCH_SIZE = 32

def load_data():
    df_train = pd.read_csv(TRAIN_LABEL_PATH, sep=';', header=None, names=['img_path', 'text']).dropna().reset_index()
    df_test = pd.read_csv(TEST_LABEL_PATH, sep=';', header=None, names=['img_path', 'text']).dropna().reset_index()
    return df_train, df_test

def load_classes():
    with open("/content/Text-Classification/list_class_name.txt", 'r') as file:
        vocab = [i_word for i_word in file.read().split("\n") if i_word != '']
    return {label: idx for idx, label in enumerate(vocab)}

def encode_labels(df_train, df_test, target_dict):
    le = preprocessing.LabelEncoder()
    le.fit(list(target_dict.keys()))
    target_dict = {key: idx for idx, key in enumerate(le.classes_)}
    
    df_train['label'] = df_train.text.apply(lambda x: target_dict[str(x)])
    df_test['label'] = df_test.text.apply(lambda x: target_dict[str(x)])
    df_train['img_path'] = df_train.img_path.apply(lambda x: os.path.join(BASE_TRAIN_DIR, x))
    df_test['img_path'] = df_test.img_path.apply(lambda x: os.path.join(BASE_TEST_DIR, x))
    
    return df_train, df_test

def create_data_generators(df_train, df_test):
    target_dict = load_classes()
    df_train, df_test = encode_labels(df_train, df_test, target_dict)

    scalar = lambda img: img / 127.5 - 1  # Scale pixel values

    train_gen = ImageDataGenerator(
        preprocessing_function=scalar,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=10
    ).flow_from_dataframe(df_train, x_col='img_path', y_col='text', 
                          target_size=(IMG_HEIGHT, IMG_WIDTH), 
                          color_mode='rgb', class_mode='categorical', 
                          batch_size=BATCH_SIZE, shuffle=True)
    
    valid_gen = ImageDataGenerator(preprocessing_function=scalar).flow_from_dataframe(
        df_test, x_col='img_path', y_col='text',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb', class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=False)

    return train_gen, valid_gen, target_dict
