import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, multilabel_confusion_matrix,accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from keras import regularizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import time
import json
from datetime import datetime


def save_confusion_matrix_plot(cm, model_name, class_names, save_dir='saved_models'):
    """Save confusion matrix as a PNG plot"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    print(f"Confusion matrix plot saved: {plot_path}")


def save_model_and_metrics(model, history, model_name, train_time, metrics, save_dir='saved_models_DL', label_encoder=None):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the model with .keras extension
    model_path = os.path.join(save_dir, f'{model_name}_model.keras')
    model.save(model_path)
    
    # Prepare metrics dictionary
    model_metrics = {
        'model_name': model_name,
        'training_time': train_time,
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'best_val_loss': min(history.history['val_loss']),
        'best_val_accuracy': max(history.history['val_accuracy']),
        'epochs_trained': len(history.history['loss']),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'additional_metrics': metrics
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(save_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(model_metrics, f, indent=4)
    
    # Save confusion matrix if available
    if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
        # Save as JSON
        cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.json')
        with open(cm_path, 'w') as f:
            json.dump(metrics['confusion_matrix'], f, indent=4)
        
        # Save as PNG plot if label_encoder is provided
        if label_encoder is not None:
            cm_array = np.array(metrics['confusion_matrix'])
            class_names = label_encoder.classes_
            save_confusion_matrix_plot(cm_array, model_name, class_names, save_dir)
    
    # Save training history plots
    save_training_history_plots(history, model_name, save_dir)
    
    print(f"Model and metrics saved for {model_name}")


def evaluate_model(model, X_test, y_test, model_name, label_encoder, show_plots=True):
    if len(X_test.shape) == 2:  # For MLP
        predictions = model.predict(X_test)
    elif model_name=='CNN' or model_name=='LSTM':  # For CNN and LSTM
        predictions = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
    else:
        predictions = model.predict(X_test)
    
    # Convert predictions to class indices
    y_pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    # Print metrics
    print(f"\nModel: {model_name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
    print("Accuracy:", accuracy)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot confusion matrix only if show_plots is True
    if show_plots:
        # Plot single confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    return {
        'classification_report': report,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist()
    }


def save_training_history_plots(history, model_name, save_dir='saved_models'):
    """Save training history plots as PNG files"""
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'Loss Over Epochs ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(save_dir, f'{model_name}_loss_history.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'Accuracy Over Epochs ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    accuracy_path = os.path.join(save_dir, f'{model_name}_accuracy_history.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plots saved for {model_name}")


def plot_history(history, model_name):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'Loss Over Epochs ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'Accuracy Over Epochs ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def train_models(num_features, custom_dataset=False):
    # Choose the correct dataset file based on custom_dataset flag
    if custom_dataset:
        dataset_filename = f'custom_output{num_features}.csv'
        print(f"📁 Using custom dataset: {dataset_filename}")
    else:
        dataset_filename = f'output{num_features}.csv'
        print(f"📁 Using standard dataset: {dataset_filename}")
    
    df = pd.read_csv(dataset_filename, index_col=0)
    df = df[df['Emotion'] != 'none']
    
    # Filter out classes with too few samples (less than 5 to ensure enough for training)
    emotion_counts = df['Emotion'].value_counts()
    valid_emotions = emotion_counts[emotion_counts >= 5].index.tolist()
    
    if len(valid_emotions) < len(emotion_counts):
        removed_emotions = set(emotion_counts.index) - set(valid_emotions)
        print(f"⚠️  Removing classes with too few samples: {removed_emotions}")
        df = df[df['Emotion'].isin(valid_emotions)]
    
    emotion = df['Emotion']
    
    # Get all feature columns (excluding 'File Path' and 'Emotion')
    feature_columns = [col for col in df.columns if col not in ['File Path', 'Emotion']]
    X = df[feature_columns]
    y = emotion
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values)

    num_classes = len(label_encoder.classes_)
    
    # Print class distribution for debugging
    print(f"Number of classes: {num_classes}")
    print("Class distribution:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    for class_idx, count in zip(unique, counts):
        class_name = label_encoder.classes_[class_idx]
        print(f"  {class_name}: {count} samples")
    
    # Step 3: Split into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape data for CNN and LSTM (if needed)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Class balancing - compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print("Class distribution in training data:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        class_name = label_encoder.classes_[class_idx]
        weight = class_weight_dict[class_idx]
        print(f"  {class_name}: {count} samples (weight: {weight:.3f})")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # Step 5a: MLP Model
    start_time = time.time()
    mlp_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax') 
    ])
    mlp_model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    mlp_history = mlp_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict
    )
    end_time = time.time()
    mlp_time = end_time - start_time
    print(f"Tempo de treinamento MLP: {mlp_time:.2f} segundos")
    mlp_metrics = evaluate_model(mlp_model, X_test, y_test, "MLP", label_encoder, show_plots=False)
    save_model_and_metrics(mlp_model, mlp_history, "MLP", mlp_time, mlp_metrics, label_encoder=label_encoder)

    # Step 5b: CNN Model
    start_time = time.time()
    cnn_model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    cnn_history = cnn_model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict
    )
    end_time = time.time()
    cnn_time = end_time - start_time
    print(f"Tempo de treinamento CNN: {cnn_time:.2f} segundos")
    cnn_metrics = evaluate_model(cnn_model, X_test_cnn, y_test, "CNN", label_encoder, show_plots=False)
    save_model_and_metrics(cnn_model, cnn_history, "CNN", cnn_time, cnn_metrics, label_encoder=label_encoder)

    # Step 5c: LSTM Model
    start_time = time.time()
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    lstm_history = lstm_model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict
    )
    end_time = time.time()
    lstm_time = end_time - start_time
    print(f"Tempo de treinamento LSTM: {lstm_time:.2f} segundos")
    lstm_metrics = evaluate_model(lstm_model, X_test_cnn, y_test, "LSTM", label_encoder, show_plots=False)
    save_model_and_metrics(lstm_model, lstm_history, "LSTM", lstm_time, lstm_metrics, label_encoder=label_encoder)

    # Redimensionar os dados para 3D
    X_train_gru = np.expand_dims(X_train, axis=1)  # (batch_size, time_steps=1, feature_dim=num_features)
    X_test_gru = np.expand_dims(X_test, axis=1)

    # Build GRU model
    start_time = time.time()
    model_gru = Sequential([
        Input(shape=(X_train_gru.shape[1], X_train_gru.shape[2])),
        GRU(64, activation='relu', return_sequences=True),
        Dropout(0.3),
        GRU(32, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model_gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training model with validation split and callbacks
    history_gru = model_gru.fit(
        X_train_gru, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict,
        verbose=1
    )
    end_time = time.time()
    gru_time = end_time - start_time
    print(f"Tempo de treinamento GRU: {gru_time:.2f} segundos")
    gru_metrics = evaluate_model(model_gru, X_test_gru, y_test, "GRU", label_encoder, show_plots=False)
    save_model_and_metrics(model_gru, history_gru, "GRU", gru_time, gru_metrics, label_encoder=label_encoder)
    

    # Final evaluation with plots
    print("\nFinal Model Evaluations:")
    evaluate_model(mlp_model, X_test, y_test, "MLP", label_encoder, show_plots=True)
    evaluate_model(cnn_model, X_test_cnn, y_test, "CNN", label_encoder, show_plots=True)
    evaluate_model(lstm_model, X_test_cnn, y_test, "LSTM", label_encoder, show_plots=True)
    evaluate_model(model_gru, X_test_gru, y_test, "GRU", label_encoder, show_plots=True)