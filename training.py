from tensorflow.keras.callbacks import EarlyStopping
from utils import plot_training_history, print_classification_report

def train_and_evaluate(model, train_gen, valid_gen, epochs=30):
    history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, 
                        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    
    plot_training_history(history)
    
    preds = model.predict(valid_gen)
    print_classification_report(valid_gen, preds)
    
    return history
