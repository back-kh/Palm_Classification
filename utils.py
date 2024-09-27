import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def plot_samples(generator):
    class_dict = generator.class_indices
    class_names = list(class_dict.keys())
    images, labels = next(generator)

    plt.figure(figsize=(20, 20))
    for i in range(min(len(labels), 25)):
        plt.subplot(5, 5, i + 1)
        plt.imshow((images[i] + 1) / 2)
        plt.title(class_names[np.argmax(labels[i])], color='blue', fontsize=16)
        plt.axis('off')
    plt.show()

def plot_training_history(history):
    t_acc, t_loss = history.history['accuracy'], history.history['loss']
    v_acc, v_loss = history.history['val_accuracy'], history.history['val_loss']
    
    epochs = range(1, len(t_acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.plot(epochs, t_loss, 'r', label='Training Loss')
    ax1.plot(epochs, v_loss, 'g', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, t_acc, 'r', label='Training Accuracy')
    ax2.plot(epochs, v_acc, 'g', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.show()

def print_classification_report(test_gen, preds):
    class_dict = test_gen.class_indices
    true_labels = test_gen.labels
    pred_labels = np.argmax(preds, axis=1)
    cm = confusion_matrix(true_labels, pred_labels)
    clr = classification_report(true_labels, pred_labels, target_names=list(class_dict.keys()), zero_division=0)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("Classification Report:\n", clr)
