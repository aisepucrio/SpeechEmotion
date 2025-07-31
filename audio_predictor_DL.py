import os
import numpy as np
import pandas as pd
import librosa
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import warnings
import argparse
import glob
warnings.filterwarnings('ignore')

# Use the correct feature extraction function
from feature_extraction import extract_features

def extract_features_from_audios_folder(audios_folder='Audios_to_predict', n_mfcc=40):
    """
    Extract features from all audio files in the Audios_to_predict folder
    Always overwrites existing CSV files to ensure compatibility.
    
    Args:
        audios_folder (str): Path to the folder containing audio files
        n_mfcc (int): Number of MFCC coefficients to extract
        
    Returns:
        str: Path to the created CSV file
    """
    if not os.path.exists(audios_folder):
        print(f"Error: '{audios_folder}' folder not found!")
        return None
    
    # Find all audio files
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audios_folder, ext)))
        audio_files.extend(glob.glob(os.path.join(audios_folder, '**', ext), recursive=True))
    
    if not audio_files:
        print(f"No audio files found in '{audios_folder}' folder!")
        return None
    
    # Sort files for consistent processing order
    audio_files.sort()
    
    print(f"Found {len(audio_files)} audio files in '{audios_folder}' folder")
    print(f"Extracting {n_mfcc} MFCC features from each file...")
    print("Note: This will overwrite any existing CSV files to ensure compatibility.")
    print("=" * 80)
    
    # Create header for MFCC features
    header = []
    for i in range(n_mfcc):
        header.extend([
            f'MFCC{i+1}_mean',
            f'MFCC{i+1}_std',
            f'MFCC{i+1}_delta_mean',
            f'MFCC{i+1}_delta_std',
            f'MFCC{i+1}_delta2_mean',
            f'MFCC{i+1}_delta2_std'
        ])
    
    # Add headers for chroma features (12 chroma coefficients)
    for i in range(12):
        header.extend([
            f'Chroma{i+1}_mean',
            f'Chroma{i+1}_std'
        ])
    
    # Add headers for mel spectrogram features (128 mel bands by default)
    for i in range(128):
        header.extend([
            f'Mel{i+1}_mean',
            f'Mel{i+1}_std'
        ])
    
    # Add File Path and Emotion columns
    header = ['File Path', 'Emotion'] + header
    
    # Initialize empty list to store features
    features_list = []
    successful_extractions = 0
    failed_extractions = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}")
        
        try:
            # Extract features using the existing function
            features = extract_features(audio_file, n_mfcc=n_mfcc)
            
            if features is not None:
                # For prediction, we'll use 'unknown' as emotion since we don't have labels
                features_list.append([audio_file, 'unknown'] + features.tolist())
                successful_extractions += 1
                print(f"  ✓ Features extracted successfully")
            else:
                failed_extractions += 1
                print(f"  ✗ Failed to extract features")
                
        except Exception as e:
            print(f"  ✗ Error processing {audio_file}: {str(e)}")
            failed_extractions += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(audio_files)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Success rate: {(successful_extractions/len(audio_files)*100):.1f}%")
    
    if successful_extractions == 0:
        print("No features were extracted. Cannot proceed with predictions.")
        return None
    
    # Convert to DataFrame and save
    df = pd.DataFrame(features_list, columns=header)
    output_csv = f'predictions{n_mfcc}.csv'
    df.to_csv(output_csv, index=False)
    
    print(f"\nFeatures saved to: {output_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Feature dimensions: {len(header) - 2} features per sample")
    
    return output_csv

def predict_emotion_single_audio(audio_path, models, scaler, label_encoder, n_mfcc, show_plots=True):
    """Predict emotion for a single audio file"""
    print(f"Processing audio file: {audio_path}")
    features = extract_features(audio_path, n_mfcc=n_mfcc)
    if features is None:
        print(f"Failed to extract features from {audio_path}")
        return None
    expected_features = len(scaler.mean_)
    if len(features) != expected_features:
        print(f"Feature mismatch! Expected {expected_features} features, got {len(features)}")
        print("This might happen if the number of MFCC coefficients doesn't match your training data")
        return None
    features_scaled = scaler.transform(features.reshape(1, -1))
    predictions = {}
    for model_name, model in models.items():
        if model_name == 'MLP':
            pred = model.predict(features_scaled)
        elif model_name in ['CNN', 'LSTM']:
            pred = model.predict(features_scaled.reshape(1, features_scaled.shape[1], 1))
        else:
            pred = model.predict(features_scaled.reshape(1, 1, features_scaled.shape[1]))
        predicted_class_idx = np.argmax(pred[0])
        predicted_class = label_encoder.classes_[predicted_class_idx]
        confidence = pred[0][predicted_class_idx]
        predictions[model_name] = {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': pred[0].tolist()
        }
    print(f"\nResults for: {os.path.basename(audio_path)}")
    print("=" * 50)
    for model_name, result in predictions.items():
        print(f"{model_name}: {result['class']} (confidence: {result['confidence']:.3f})")
    if show_plots:
        plot_predictions(predictions, os.path.basename(audio_path))
    return predictions

def predict_emotions_batch(audio_files, models, scaler, label_encoder, n_mfcc, show_plots=True):
    """Predict emotions for multiple audio files"""
    all_results = {}
    for audio_file in audio_files:
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file}")
        print(f"{'='*60}")
        result = predict_emotion_single_audio(audio_file, models, scaler, label_encoder, n_mfcc, show_plots=False)
        if result:
            all_results[audio_file] = result
    if show_plots and all_results:
        plot_batch_results(all_results, label_encoder.classes_)
    return all_results

def predict_emotions_from_audios_folder(models, scaler, label_encoder, n_mfcc, show_plots=True, save_results=False):
    """
    Automatically predict emotions for all audio files in the 'Audios_to_predict' folder
    
    Args:
        models: Dictionary of loaded models
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        n_mfcc: Number of MFCC features
        show_plots: Whether to show plots
        save_results: Whether to save results to JSON file (default: False for pipeline integration)
    
    Returns:
        dict: Results for all processed audio files
    """
    audios_folder = 'Audios_to_predict'
    
    if not os.path.exists(audios_folder):
        print(f"Error: '{audios_folder}' folder not found!")
        return None
    
    # Find all audio files in the Audios folder
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audios_folder, ext)))
        audio_files.extend(glob.glob(os.path.join(audios_folder, '**', ext), recursive=True))
    
    if not audio_files:
        print(f"No audio files found in '{audios_folder}' folder!")
        return None
    
    # Sort files for consistent processing order
    audio_files.sort()
    
    print(f"Found {len(audio_files)} audio files in '{audios_folder}' folder")
    print("Starting emotion prediction for all files...")
    print("=" * 80)
    
    # Process all audio files
    all_results = {}
    successful_predictions = 0
    failed_predictions = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}")
        print("-" * 60)
        
        try:
            result = predict_emotion_single_audio(audio_file, models, scaler, label_encoder, n_mfcc, show_plots=False)
            if result:
                all_results[audio_file] = result
                successful_predictions += 1
            else:
                failed_predictions += 1
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            failed_predictions += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(audio_files)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Success rate: {(successful_predictions/len(audio_files)*100):.1f}%")
    
    if successful_predictions > 0:
        # Show batch results plot
        if show_plots:
            plot_batch_results(all_results, label_encoder.classes_)
        
        # Generate summary statistics
        generate_prediction_summary(all_results, label_encoder.classes_)
        
        # Save results if requested (for standalone usage)
        if save_results:
            output_file = f'emotion_predictions_audios_{len(audio_files)}_files.json'
            
            # Convert numpy values to native Python types for JSON serialization
            json_results = {}
            for audio_file, results in all_results.items():
                json_results[audio_file] = {}
                for model_name, result in results.items():
                    json_results[audio_file][model_name] = {
                        'class': str(result['class']),
                        'confidence': float(result['confidence']),
                        'probabilities': [float(p) for p in result['probabilities']]
                    }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=4)
            print(f"\nResults saved to: {output_file}")
            
            # Also save a CSV summary
            save_results_to_csv(all_results, output_file.replace('.json', '.csv'))
    
    return all_results

def generate_prediction_summary(all_results, class_names):
    """Generate and display summary statistics of predictions"""
    if not all_results:
        return
    
    model_names = list(next(iter(all_results.values())).keys())
    
    print(f"\nPREDICTION STATISTICS BY MODEL")
    print("-" * 50)
    
    for model_name in model_names:
        print(f"\n{model_name} Model:")
        class_counts = {cls: 0 for cls in class_names}
        total_confidence = 0
        count = 0
        
        for file_result in all_results.values():
            if model_name in file_result:
                result = file_result[model_name]
                class_counts[result['class']] += 1
                total_confidence += result['confidence']
                count += 1
        
        avg_confidence = total_confidence / count if count > 0 else 0
        
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Predictions by emotion:")
        for emotion, count in class_counts.items():
            percentage = (count / len(all_results)) * 100
            print(f"    {emotion}: {count} ({percentage:.1f}%)")

def save_results_to_csv(all_results, output_file):
    """Save prediction results to CSV format"""
    if not all_results:
        return
    
    model_names = list(next(iter(all_results.values())).keys())
    class_names = list(next(iter(all_results.values()))[model_names[0]]['probabilities'])
    
    # Create DataFrame
    rows = []
    for audio_file, results in all_results.items():
        row = {'audio_file': os.path.basename(audio_file)}
        
        for model_name in model_names:
            if model_name in results:
                result = results[model_name]
                row[f'{model_name}_prediction'] = result['class']
                row[f'{model_name}_confidence'] = result['confidence']
                
                # Add individual class probabilities
                for i, prob in enumerate(result['probabilities']):
                    row[f'{model_name}_{class_names[i]}_prob'] = prob
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"CSV summary saved to: {output_file}")

def plot_predictions(predictions, filename):
    """Plot prediction results for a single file"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of model predictions
    model_names = list(predictions.keys())
    classes = [predictions[model]['class'] for model in model_names]
    confidences = [predictions[model]['confidence'] for model in model_names]
    
    bars = ax1.bar(model_names, confidences, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title(f'Model Predictions - {filename}')
    ax1.set_ylabel('Confidence')
    ax1.set_ylim(0, 1)
    
    # Add class labels on bars
    for bar, class_name in zip(bars, classes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                class_name, ha='center', va='bottom', fontweight='bold')
    
    # Heatmap of all probabilities
    prob_matrix = np.array([predictions[model]['probabilities'] for model in model_names])
    im = ax2.imshow(prob_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(label_encoder.classes_)))
    ax2.set_xticklabels(label_encoder.classes_, rotation=45, ha='right')
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels(model_names)
    ax2.set_title('Probability Distribution Across All Classes')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Probability')
    
    plt.tight_layout()
    plt.show()

def plot_batch_results(all_results, class_names):
    """Plot summary of batch results"""
    # Count predictions by model and class
    model_names = list(next(iter(all_results.values())).keys())
    prediction_counts = {model: {cls: 0 for cls in class_names} for model in model_names}
    
    for file_result in all_results.values():
        for model_name, result in file_result.items():
            prediction_counts[model_name][result['class']] += 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.array([[prediction_counts[model][cls] for cls in class_names] for model in model_names])
    
    im = ax.imshow(data, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_title('Prediction Counts by Model and Emotion Class')
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, data[i, j], ha="center", va="center", color="black" if data[i, j] < data.max()/2 else "white")
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.show()

def create_scaler_from_features(csv_file):
    """
    Create a scaler from the extracted features CSV file
    
    Args:
        csv_file (str): Path to the CSV file with extracted features
        
    Returns:
        tuple: (scaler, label_encoder, feature_columns)
    """
    df = pd.read_csv(csv_file, index_col=0)
    feature_columns = [col for col in df.columns if col not in ['File Path', 'Emotion']]
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(df[feature_columns])
    
    # Create label encoder
    label_encoder = LabelEncoder()
    # For prediction, we'll use a standard set of emotions
    standard_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    label_encoder.fit(standard_emotions)
    
    return scaler, label_encoder, feature_columns

def main():
    parser = argparse.ArgumentParser(description='Predict emotions from audio files using trained models')
    parser.add_argument('--audio', type=str, help='Path to a single audio file')
    parser.add_argument('--directory', type=str, help='Path to directory containing audio files')
    parser.add_argument('--audios-folder', action='store_true', help='Process all audio files in the Audios_to_predict folder')
    parser.add_argument('--model', type=str, choices=['MLP', 'CNN', 'LSTM', 'GRU'], 
                       help='Specific model to use (default: all models)')
    parser.add_argument('--features', type=int, default=40, 
                       help=f'Number of MFCC features (default: 40)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Use the features argument instead of N_MFCC
    n_mfcc = args.features
        
    # Load models and preprocessing
    models = {}
    scaler = None
    label_encoder = None
    
    try:
        # Load models that exist
        models = {}
        
        # Try to load MLP model
        try:
            mlp_model = load_model('saved_models_DL/MLP_model.keras')
            models['MLP'] = mlp_model
            print("✓ MLP loaded")
        except Exception as e:
            print(f"✗ MLP model not found: {e}")
        
        # Try to load CNN model
        try:
            cnn_model = load_model('saved_models_DL/CNN_model.keras')
            models['CNN'] = cnn_model
            print("✓ CNN loaded")
        except Exception as e:
            print(f"✗ CNN model not found: {e}")
        
        # Try to load LSTM model
        try:
            lstm_model = load_model('saved_models_DL/LSTM_model.keras')
            models['LSTM'] = lstm_model
            print("✓ LSTM loaded")
        except Exception as e:
            print(f"✗ LSTM model not found: {e}")
        
        # Try to load GRU model
        try:
            gru_model = load_model('saved_models_DL/GRU_model.keras')
            models['GRU'] = gru_model
            print("✓ GRU loaded")
        except Exception as e:
            print(f"✗ GRU model not found: {e}")
        
        if not models:
            print("Error: No models could be loaded!")
            return
        
        print(f"Loaded {len(models)} model(s): {list(models.keys())}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure all model files exist in the 'saved_models_DL' directory")
        return
    
    # Filter models if specific model requested
    if args.model:
        models = {args.model: models[args.model]}
        print(f"Using only {args.model} model")
    
    # Process audio files
    if args.audios_folder:
        # Process all files in Audios folder
        # Always extract features from audio files (overwrite existing CSV files)
        print("Extracting features from audio files...")
        csv_file = extract_features_from_audios_folder('Audios_to_predict', n_mfcc)
        if csv_file is None:
            print("Failed to extract features. Cannot proceed with predictions.")
            return
        
        # Create scaler from extracted features
        scaler, label_encoder, feature_columns = create_scaler_from_features(csv_file)
        print(f"Created scaler from {csv_file}")
        print(f"Features: {len(feature_columns)}")
        print(f"Classes: {label_encoder.classes_}")
        
        return predict_emotions_from_audios_folder(models, scaler, label_encoder, n_mfcc, not args.no_plot, False)
        
    elif args.audio:
        # Single audio file
        if not os.path.exists(args.audio):
            print(f"Audio file not found: {args.audio}")
            return
        
        result = predict_emotion_single_audio(args.audio, models, scaler, label_encoder, n_mfcc, not args.no_plot)
        
        if args.output and result:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Results saved to {args.output}")
    
    elif args.directory:
        # Directory of audio files
        if not os.path.exists(args.directory):
            print(f"Directory not found: {args.directory}")
            return
        
        # Find all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(args.directory, ext)))
            audio_files.extend(glob.glob(os.path.join(args.directory, '**', ext), recursive=True))
        
        if not audio_files:
            print(f"No audio files found in {args.directory}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        # Limit to first 10 files for testing
        if len(audio_files) > 10:
            print(f"Processing first 10 files (use --directory for all files)")
            audio_files = audio_files[:10]
        
        results = predict_emotions_batch(audio_files, models, scaler, label_encoder, n_mfcc, not args.no_plot)
        
        if args.output and results:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {args.output}")
    
    else:
        # Interactive mode
        print("\nNo arguments provided. Choose an option:")
        print("1. Single audio file")
        print("2. Directory of audio files")
        print("3. All files in Audios_to_predict folder")
        print("4. Example files from dataset")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            audio_path = input("Enter path to audio file: ").strip()
            if os.path.exists(audio_path):
                predict_emotion_single_audio(audio_path, models, scaler, label_encoder, n_mfcc, not args.no_plot)
            else:
                print("File not found!")
        
        elif choice == '2':
            dir_path = input("Enter path to directory: ").strip()
            if os.path.exists(dir_path):
                audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
                audio_files = []
                for ext in audio_extensions:
                    audio_files.extend(glob.glob(os.path.join(dir_path, ext)))
                    audio_files.extend(glob.glob(os.path.join(dir_path, '**', ext), recursive=True))
                
                if audio_files:
                    print(f"Found {len(audio_files)} audio files")
                    if len(audio_files) > 10:
                        print("Processing first 10 files...")
                        audio_files = audio_files[:10]
                    predict_emotions_batch(audio_files, models, scaler, label_encoder, n_mfcc, not args.no_plot)
                else:
                    print("No audio files found!")
            else:
                print("Directory not found!")
        
        elif choice == '3':
            # Process all files in Audios folder
            # Always extract features from audio files (overwrite existing CSV files)
            csv_file = extract_features_from_audios_folder('Audios_to_predict', n_mfcc)
            if csv_file is None:
                print("Failed to extract features. Cannot proceed with predictions.")
                return
            
            # Create scaler from extracted features
            scaler, label_encoder, feature_columns = create_scaler_from_features(csv_file)
            print(f"Created scaler from {csv_file}")
            print(f"Features: {len(feature_columns)}")
            print(f"Classes: {label_encoder.classes_}")
            
            predict_emotions_from_audios_folder(models, scaler, label_encoder, n_mfcc, not args.no_plot, True)
        
        elif choice == '4':
            # Use example files from dataset
            example_files = [
                "Dataset/Crema/1001_DFA_ANG_XX.wav",
                "Dataset/Ravdess/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav",
                "Dataset/Tess/OAF_angry/OAF_back_angry.wav"
            ]
            
            available_files = [f for f in example_files if os.path.exists(f)]
            if available_files:
                print(f"Processing {len(available_files)} example files...")
                predict_emotions_batch(available_files, models, scaler, label_encoder, n_mfcc, not args.no_plot)
            else:
                print("No example files found in dataset!")

if __name__ == "__main__":
    main() 
