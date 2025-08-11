# training_utils_tf.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

def train_tf_model(model, train_X, train_y, validation_data=None, epochs=100, batch_size=32, callbacks=None, verbose=1):
    """
    Trains a Keras model.
    Args:
        model (keras.Model): The Keras model to train.
        train_X (np.array): Training input sequences.
        train_y (np.array): Training target sequences.
        validation_data (tuple, optional): Data for validation (val_X, val_y).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        callbacks (list, optional): Keras callbacks (e.g., EarlyStopping).
        verbose (int): Verbosity mode for Keras fit (0, 1, or 2).
    Returns:
        keras.callbacks.History: Training history object.
    """
    print(f"[{model.name if hasattr(model, 'name') else 'KerasModel'}] Training started...")
    history = model.fit(
        train_X, 
        train_y, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose
    )
    print(f"[{model.name if hasattr(model, 'name') else 'KerasModel'}] Training finished.")
    return history, model # Return model as it might be modified by callbacks like ModelCheckpoint

def evaluate_tf_model(model, test_X, test_y, batch_size=32, verbose=0):
    """
    Evaluates a Keras model.
    Args:
        model (keras.Model): The trained Keras model.
        test_X (np.array): Test input sequences.
        test_y (np.array): Test target sequences.
        batch_size (int): Batch size for evaluation.
        verbose (int): Verbosity mode for Keras evaluate.
    Returns:
        list: List of evaluation metrics (e.g., [loss, mae]).
        np.array: Predictions made by the model on test_X.
    """
    print(f"Evaluating [{model.name if hasattr(model, 'name') else 'KerasModel'}]...")
    eval_results = model.evaluate(test_X, test_y, batch_size=batch_size, verbose=verbose)
    print(f"Evaluation results (loss, metrics): {eval_results}")
    
    predictions = model.predict(test_X, batch_size=batch_size, verbose=verbose)
    return eval_results, predictions