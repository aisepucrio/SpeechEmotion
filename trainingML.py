from pycaret.datasets import get_data
from pycaret.classification import *
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import time
from Utils.emotion_number_dict import number_to_emotion
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

       
def train_models(num_features, custom_dataset=False):
    # Create output directory for models and metrics
    output_dir = f"saved_models_ML"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Choose the correct dataset file based on custom_dataset flag
    if custom_dataset:
        dataset_filename = f'custom_output{num_features}'
        print(f"📁 Using custom dataset: {dataset_filename}")
    else:
        dataset_filename = f'output{num_features}'
        print(f"📁 Using standard dataset: {dataset_filename}")
    
    dataset = get_data(dataset_filename)
    dataset_sem_path = dataset.drop(columns=['File Path'])
    
    # Filter out classes with too few samples (less than 5 to ensure enough for training)
    emotion_counts = dataset_sem_path['Emotion'].value_counts()
    valid_emotions = emotion_counts[emotion_counts >= 5].index.tolist()
    
    if len(valid_emotions) < len(emotion_counts):
        removed_emotions = set(emotion_counts.index) - set(valid_emotions)
        print(f"⚠️  Removing classes with too few samples: {removed_emotions}")
        dataset_sem_path = dataset_sem_path[dataset_sem_path['Emotion'].isin(valid_emotions)]

    # Configure PyCaret for small datasets and class imbalance
    s = setup(dataset_sem_path, 
              target = 'Emotion', 
              session_id = 123, 
              fix_imbalance=True,
              fold=2,  # Use minimum folds for small datasets
              fold_strategy='stratifiedkfold')  # Use stratified k-fold


    # Train all models individually instead of using compare_models()
    # This gives us full control and avoids double training
    print("Training all models individually...")
    
    # Specify exactly which models you want to train and save
    models_to_save = ['lightgbm', 'gbc', 'et', 'rf', 'lda', 'ridge', 'knn', 'lr', 'dt', 'qda', 'ada', 'svm', 'nb', 'dummy']  # All PyCaret models
    
    trained_models = {}
    models_saved = 0
    best_model = None
    best_score = 0
    
    for i, model_name in enumerate(models_to_save, 1):
        print(f"Training model {i}/{len(models_to_save)}: {model_name}...")
        try:
            # Add timeout and progress feedback
            start_time = time.time()
            model = create_model(model_name, verbose=False)
            training_time = time.time() - start_time
            print(f"  ✓ {model_name} trained in {training_time:.2f} seconds")
            
            # Evaluate the model to get its score
            predictions = predict_model(model)
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(predictions['Emotion'], predictions['prediction_label'])
            print(f"  ✓ {model_name} accuracy: {accuracy:.4f}")
            
            # Track the best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
            
            # Save the model
            save_model(model, os.path.join(output_dir, f'{model_name}_{num_features}'))
            print(f"  ✓ {model_name} saved successfully")
            models_saved += 1
            trained_models[model_name] = model
            
        except Exception as e:
            print(f"  ✗ Could not train/save model {model_name}: {e}")
            print(f"  Skipping {model_name} and continuing with next model...")
    
    print(f"Trained and saved {models_saved} models")
    print(f"Best model: {best_model} with accuracy: {best_score:.4f}")
    
    # Save the best model separately
    if best_model is not None:
        try:
            save_model(best_model, os.path.join(output_dir, f'best_model_{num_features}'))
            print(f"Saved best model with accuracy: {best_score:.4f}")
        except Exception as e:
            print(f"Could not save best model: {e}")

    print('=========================================================================================================================')
    if best_model is not None:
        print(evaluate_model(best_model))
    
    # Generate and save confusion matrix with emotion labels for the best model
    if best_model is not None:
        print("Generating confusion matrix for best model...")
        
        # Get predictions and actual values for confusion matrix
        predictions = predict_model(best_model)
        y_true = predictions['Emotion']
        y_pred = predictions['prediction_label']
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get emotion labels
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        
        # Handle both numeric and text labels
        emotion_labels = []
        for label in unique_labels:
            try:
                # If label is numeric, convert to emotion name
                if isinstance(label, (int, float)) or str(label).isdigit():
                    emotion_labels.append(number_to_emotion.get(int(label), str(label)))
                else:
                    # If label is already text, use it as is
                    emotion_labels.append(str(label))
            except (ValueError, TypeError):
                # Fallback to original label if conversion fails
                emotion_labels.append(str(label))
        
        # Create custom confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_labels, yticklabels=emotion_labels)
        
        plt.title(f'Confusion Matrix - Best Model ({num_features} Features)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Emotion', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the customized confusion matrix
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_best_{num_features}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save model metrics
    print("Saving model metrics...")
    
    # Create a summary of all trained models
    model_summary = []
    for model_name, model in trained_models.items():
        try:
            predictions = predict_model(model)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(predictions['Emotion'], predictions['prediction_label'])
            precision = precision_score(predictions['Emotion'], predictions['prediction_label'], average='weighted')
            recall = recall_score(predictions['Emotion'], predictions['prediction_label'], average='weighted')
            f1 = f1_score(predictions['Emotion'], predictions['prediction_label'], average='weighted')
            
            model_summary.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })
        except Exception as e:
            print(f"Could not evaluate {model_name}: {e}")
    
    # Save model summary
    if model_summary:
        summary_df = pd.DataFrame(model_summary)
        summary_df.to_csv(os.path.join(output_dir, f'model_summary_{num_features}.csv'), index=False)
        print(f"Saved model summary: {len(model_summary)} models")
    
    # Generate confusion matrices for all saved models
    print("\nGenerating confusion matrices for all saved models...")
    
    # List of all models we have (best model + saved models)
    all_saved_models = ['best'] + list(trained_models.keys())
    
    for model_name in all_saved_models:
        try:
            print(f"Generating confusion matrix for {model_name}...")
            
            if model_name == 'best':
                # Use the best model we already have
                current_model = best_model
            else:
                # Load the saved model
                current_model = load_model(os.path.join(output_dir, f'{model_name}_{num_features}'))
            
            # Get predictions for this model
            predictions = predict_model(current_model)
            y_true = predictions['Emotion']
            y_pred = predictions['prediction_label']
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Get emotion labels
            unique_labels = sorted(list(set(y_true) | set(y_pred)))
            
            # Handle both numeric and text labels
            emotion_labels = []
            for label in unique_labels:
                try:
                    # If label is numeric, convert to emotion name
                    if isinstance(label, (int, float)) or str(label).isdigit():
                        emotion_labels.append(number_to_emotion.get(int(label), str(label)))
                    else:
                        # If label is already text, use it as is
                        emotion_labels.append(str(label))
                except (ValueError, TypeError):
                    # Fallback to original label if conversion fails
                    emotion_labels.append(str(label))
            
            # Save confusion matrix as CSV
            cm_df = pd.DataFrame(cm, index=emotion_labels, columns=emotion_labels)
            cm_df.to_csv(os.path.join(output_dir, f'confusion_matrix_{model_name}_{num_features}.csv'), index=True)
            print(f"  ✓ Confusion matrix CSV saved for {model_name}")
            
            # Create custom confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=emotion_labels, yticklabels=emotion_labels)
            
            plt.title(f'Confusion Matrix - {model_name.upper()} ({num_features} Features)', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
            plt.ylabel('Actual Emotion', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the confusion matrix
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}_{num_features}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Confusion matrix PNG saved for {model_name}")
            
        except Exception as e:
            print(f"  ✗ Could not generate confusion matrix for {model_name}: {e}")
    
    print(f"Generated confusion matrices for {len(all_saved_models)} models")
    
    print(f"All outputs saved to directory: {output_dir}")
    return best_model
 