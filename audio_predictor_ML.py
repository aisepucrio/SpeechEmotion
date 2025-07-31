import librosa
import numpy as np
import pickle
import os
import sys
from pycaret.classification import load_model, predict_model
import pandas as pd
from Utils.emotion_number_dict import number_to_emotion
from feature_extraction import extract_features
import glob
from datetime import datetime


def create_feature_dataframe(features, n_mfcc=40):
    """
    Create a DataFrame with the extracted features in the same format as training data.
    """
    # Create feature names
    feature_names = []
    
    # MFCC feature names
    for i in range(n_mfcc):
        feature_names.extend([
            f'MFCC{i+1}_mean',
            f'MFCC{i+1}_std',
            f'MFCC{i+1}_delta_mean',
            f'MFCC{i+1}_delta_std',
            f'MFCC{i+1}_delta2_mean',
            f'MFCC{i+1}_delta2_std'
        ])
    
    # Chroma feature names (12 chroma coefficients)
    for i in range(12):
        feature_names.extend([
            f'Chroma{i+1}_mean',
            f'Chroma{i+1}_std'
        ])
    
    # Mel spectrogram feature names (128 mel bands by default)
    for i in range(128):
        feature_names.extend([
            f'Mel{i+1}_mean',
            f'Mel{i+1}_std'
        ])
    
    # Create DataFrame
    df = pd.DataFrame([features], columns=feature_names)
    return df


def predict_emotion(audio_file_path, model_path=None, n_mfcc=40):
    """
    Predict emotion from an audio file using a trained model.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_path (str): Path to the trained model (.pkl file)
        n_mfcc (int): Number of MFCC features (default: 40)
    
    Returns:
        dict: Dictionary containing prediction results
    """
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Use default model if none specified
    if model_path is None:
        model_path = "saved_models_ML/gbc_40"
    
    # Check if model exists (try both with and without .pkl extension)
    model_path_with_ext = model_path if model_path.endswith('.pkl') else f"{model_path}.pkl"
    model_path_without_ext = model_path[:-4] if model_path.endswith('.pkl') else model_path
    
    if not os.path.exists(model_path_with_ext) and not os.path.exists(model_path_without_ext):
        raise FileNotFoundError(f"Model file not found: {model_path_with_ext} or {model_path_without_ext}")
    
    try:
        # Extract features from audio file
        print(f"Extracting features from: {audio_file_path}")
        features = extract_features(audio_file_path, n_mfcc=n_mfcc)
        
        if features is None:
            raise ValueError("Failed to extract features from audio file")
        
        # Load the trained model first to understand its expected format
        print(f"Loading model from: {model_path_without_ext}")
        model = load_model(model_path_without_ext)
        
        # Create DataFrame with features
        feature_df = create_feature_dataframe(features, n_mfcc=n_mfcc)
        
        # Try to make prediction
        print("Making prediction...")
        try:
            prediction = predict_model(model, data=feature_df)
        except KeyError as e:
            # If there's a column mismatch, try to fix it
            if "Unnamed: 0" in str(e):
                print("Fixing DataFrame format...")
                # Add the missing index column
                feature_df.insert(0, 'Unnamed: 0', 0)
                prediction = predict_model(model, data=feature_df)
            else:
                raise
        
        # Extract prediction results
        predicted_label = prediction['prediction_label'].iloc[0]
        prediction_score = prediction['prediction_score'].iloc[0]
        
        # Convert numeric label to emotion name if needed
        if isinstance(predicted_label, (int, float)) or str(predicted_label).isdigit():
            emotion_name = number_to_emotion.get(int(predicted_label), str(predicted_label))
        else:
            emotion_name = str(predicted_label)
        
        # Get confidence scores for all emotions if available
        confidence_scores = {}
        for col in prediction.columns:
            if col.startswith('prediction_score_'):
                emotion_num = col.split('_')[-1]
                try:
                    emotion_num = int(emotion_num)
                    emotion = number_to_emotion.get(emotion_num, f"emotion_{emotion_num}")
                    confidence_scores[emotion] = prediction[col].iloc[0]
                except ValueError:
                    continue
        
        result = {
            'audio_file': audio_file_path,
            'predicted_emotion': emotion_name,
            'confidence': prediction_score,
            'all_confidence_scores': confidence_scores,
            'model_used': model_path_without_ext
        }
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


def predict_multiple_files(audio_files, model_path=None, n_mfcc=40):
    """
    Predict emotions for multiple audio files.
    
    Args:
        audio_files (list): List of audio file paths
        model_path (str): Path to the trained model
        n_mfcc (int): Number of MFCC features
    
    Returns:
        list: List of prediction results
    """
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nProcessing file {i}/{len(audio_files)}: {os.path.basename(audio_file)}")
        try:
            result = predict_emotion(audio_file, model_path, n_mfcc)
            results.append(result)
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            results.append({
                'audio_file': audio_file,
                'error': str(e)
            })
    
    return results


def predict_all_audio_files(audio_folder="Audios_to_predict", model_path=None, n_mfcc=40, audio_extensions=None):
    """
    Automatically predict emotions for all audio files in a specified folder.
    
    Args:
        audio_folder (str): Path to the folder containing audio files (default: "Audios_to_predict")
        model_path (str): Path to the trained model
        n_mfcc (int): Number of MFCC features
        audio_extensions (list): List of audio file extensions to process (default: ['.wav', '.mp3', '.flac', '.m4a'])
    
    Returns:
        list: List of prediction results for all audio files
    """
    if audio_extensions is None:
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    # Check if audio folder exists
    if not os.path.exists(audio_folder):
        raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
    
    # Find all audio files in the folder
    audio_files = []
    for ext in audio_extensions:
        pattern = os.path.join(audio_folder, f"*{ext}")
        audio_files.extend(glob.glob(pattern))
        # Also check for uppercase extensions
        pattern_upper = os.path.join(audio_folder, f"*{ext.upper()}")
        audio_files.extend(glob.glob(pattern_upper))
    
    # Remove duplicates and sort
    audio_files = sorted(list(set(audio_files)))
    
    if not audio_files:
        print(f"No audio files found in {audio_folder} with extensions: {audio_extensions}")
        return []
    
    print(f"Found {len(audio_files)} audio files in {audio_folder}")
    print("Starting emotion prediction for all files...")
    
    # Use the existing predict_multiple_files function
    results = predict_multiple_files(audio_files, model_path, n_mfcc)
    
    # Print summary
    successful_predictions = [r for r in results if 'error' not in r]
    failed_predictions = [r for r in results if 'error' in r]
    
    print(f"\n{'='*50}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful predictions: {len(successful_predictions)}")
    print(f"Failed predictions: {len(failed_predictions)}")
    
    if successful_predictions:
        print(f"\nEmotion distribution:")
        emotion_counts = {}
        for result in successful_predictions:
            emotion = result['predicted_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / len(successful_predictions)) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    if failed_predictions:
        print(f"\nFailed files:")
        for result in failed_predictions:
            print(f"  {os.path.basename(result['audio_file'])}: {result['error']}")
    
    return results


def get_available_models(models_folder="saved_models_ML"):
    """
    Get all available model files in the saved_models_ML folder.
    
    Args:
        models_folder (str): Path to the models folder
    
    Returns:
        list: List of model paths (without .pkl extension)
    """
    if not os.path.exists(models_folder):
        print(f"Models folder not found: {models_folder}")
        return []
    
    model_files = glob.glob(os.path.join(models_folder, "*.pkl"))
    model_paths = []
    
    for model_file in model_files:
        # Remove .pkl extension and get model name
        model_path = model_file[:-4]  # Remove .pkl
        model_name = os.path.basename(model_path)
        model_paths.append(model_path)
        print(f"Found model: {model_name}")
    
    return model_paths


def predict_emotion_all_models(audio_file_path, model_paths=None, n_mfcc=40):
    """
    Predict emotion from an audio file using all available models.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_paths (list): List of model paths to use (if None, uses all available models)
        n_mfcc (int): Number of MFCC features (default: 40)
    
    Returns:
        dict: Dictionary containing prediction results from all models
    """
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Get available models if none specified
    if model_paths is None:
        model_paths = get_available_models()
    
    if not model_paths:
        raise ValueError("No models found in saved_models_ML folder")
    
    # Extract features from audio file (only once)
    print(f"Extracting features from: {audio_file_path}")
    features = extract_features(audio_file_path, n_mfcc=n_mfcc)
    
    if features is None:
        raise ValueError("Failed to extract features from audio file")
    
    # Create DataFrame with features
    feature_df = create_feature_dataframe(features, n_mfcc=n_mfcc)
    
    results = {
        'audio_file': audio_file_path,
        'filename': os.path.basename(audio_file_path),
        'models': {}
    }
    
    # Make predictions with each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"Making prediction with {model_name}...")
        
        try:
            # Load the trained model
            model = load_model(model_path)
            
            # Try to make prediction
            try:
                prediction = predict_model(model, data=feature_df)
            except KeyError as e:
                # If there's a column mismatch, try to fix it
                if "Unnamed: 0" in str(e):
                    print(f"Fixing DataFrame format for {model_name}...")
                    # Add the missing index column
                    feature_df_fixed = feature_df.copy()
                    feature_df_fixed.insert(0, 'Unnamed: 0', 0)
                    prediction = predict_model(model, data=feature_df_fixed)
                else:
                    raise
            
            # Extract prediction results
            predicted_label = prediction['prediction_label'].iloc[0]
            prediction_score = prediction['prediction_score'].iloc[0]
            
            # Convert numeric label to emotion name if needed
            if isinstance(predicted_label, (int, float)) or str(predicted_label).isdigit():
                emotion_name = number_to_emotion.get(int(predicted_label), str(predicted_label))
            else:
                emotion_name = str(predicted_label)
            
            # Get confidence scores for all emotions if available
            confidence_scores = {}
            for col in prediction.columns:
                if col.startswith('prediction_score_'):
                    emotion_num = col.split('_')[-1]
                    try:
                        emotion_num = int(emotion_num)
                        emotion = number_to_emotion.get(emotion_num, f"emotion_{emotion_num}")
                        confidence_scores[emotion] = prediction[col].iloc[0]
                    except ValueError:
                        continue
            
            results['models'][model_name] = {
                'predicted_emotion': emotion_name,
                'confidence': prediction_score,
                'all_confidence_scores': confidence_scores,
                'model_path': model_path
            }
            
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            results['models'][model_name] = {
                'error': str(e),
                'model_path': model_path
            }
    
    return results


def predict_all_audio_files_all_models(audio_folder="Audios_to_predict", model_paths=None, n_mfcc=40, audio_extensions=None, save_csv=False):
    """
    Automatically predict emotions for all audio files in a specified folder using all available models.
    
    Args:
        audio_folder (str): Path to the folder containing audio files (default: "Audios_to_predict")
        model_paths (list): List of specific model paths to use (if None, uses all available)
        n_mfcc (int): Number of MFCC features
        audio_extensions (list): List of audio file extensions to process (default: ['.wav', '.mp3', '.flac', '.m4a'])
        save_csv (bool): Whether to save results to CSV file (default: False for pipeline integration)
    
    Returns:
        dict: Dictionary containing prediction results for all audio files and models
    """
    if audio_extensions is None:
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    # Check if audio folder exists
    if not os.path.exists(audio_folder):
        raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
    
    # Find all audio files in the folder
    audio_files = []
    for ext in audio_extensions:
        pattern = os.path.join(audio_folder, f"*{ext}")
        audio_files.extend(glob.glob(pattern))
        # Also check for uppercase extensions
        pattern_upper = os.path.join(audio_folder, f"*{ext.upper()}")
        audio_files.extend(glob.glob(pattern_upper))
    
    # Remove duplicates and sort
    audio_files = sorted(list(set(audio_files)))
    
    if not audio_files:
        print(f"No audio files found in {audio_folder} with extensions: {audio_extensions}")
        return {}
    
    print(f"Found {len(audio_files)} audio files in {audio_folder}")
    print("Starting emotion prediction for all files with all models...")
    
    # Get available models if not specified
    if model_paths is None:
        model_paths = get_available_models()
    
    if not model_paths:
        print("No models found!")
        return {}
    
    print(f"Using {len(model_paths)} models: {[os.path.basename(m) for m in model_paths]}")
    
    # Process all audio files with all models
    all_results = {}
    
    for audio_file in audio_files:
        print(f"\nProcessing: {os.path.basename(audio_file)}")
        file_results = predict_emotion_all_models(audio_file, model_paths, n_mfcc)
        all_results[audio_file] = file_results
    
    # Print summary
    successful_predictions = sum(1 for result in all_results.values() if 'error' not in result)
    failed_predictions = len(all_results) - successful_predictions
    
    print(f"\n{'='*50}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total files processed: {len(all_results)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    
    if successful_predictions > 0:
        print(f"\nEmotion distribution across all models:")
        emotion_counts = {}
        for result in all_results.values():
            if 'error' not in result and 'models' in result:
                for model_name, model_result in result['models'].items():
                    if 'error' not in model_result:
                        emotion = model_result['predicted_emotion']
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / (successful_predictions * len(model_paths))) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    if failed_predictions > 0:
        print(f"\nFailed files:")
        for audio_file, result in all_results.items():
            if 'error' in result:
                print(f"  {os.path.basename(audio_file)}: {result['error']}")
    
    # Save to CSV only if requested (for standalone usage)
    if save_csv:
        save_results_to_csv(all_results, model_paths)
    
    return all_results


def save_results_to_csv(results, model_paths, output_dir="."):
    """
    Save prediction results to a CSV file.
    
    Args:
        results (list): List of prediction results
        model_paths (list): List of model paths used
        output_dir (str): Directory to save the CSV file
    
    Returns:
        str: Path to the saved CSV file
    """
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"emotion_predictions_{timestamp}.csv")
    
    # Prepare data for CSV
    csv_data = []
    
    for result in results:
        row = {
            'filename': result.get('filename', os.path.basename(result.get('audio_file', ''))),
            'audio_file': result.get('audio_file', ''),
        }
        
        # Add error column if there was an error
        if 'error' in result:
            row['error'] = result['error']
            # Add empty columns for each model
            for model_path in model_paths:
                model_name = os.path.basename(model_path)
                row[f'{model_name}_emotion'] = ''
                row[f'{model_name}_confidence'] = ''
                row[f'{model_name}_error'] = ''
        else:
            # Add predictions for each model
            for model_path in model_paths:
                model_name = os.path.basename(model_path)
                if model_name in result['models']:
                    model_result = result['models'][model_name]
                    if 'error' in model_result:
                        row[f'{model_name}_emotion'] = ''
                        row[f'{model_name}_confidence'] = ''
                        row[f'{model_name}_error'] = model_result['error']
                    else:
                        row[f'{model_name}_emotion'] = model_result['predicted_emotion']
                        row[f'{model_name}_confidence'] = model_result['confidence']
                        row[f'{model_name}_error'] = ''
                else:
                    row[f'{model_name}_emotion'] = ''
                    row[f'{model_name}_confidence'] = ''
                    row[f'{model_name}_error'] = 'Model not found'
        
        csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)
    
    return csv_filename


def print_prediction_result(result):
    """
    Print prediction results in a formatted way.
    """
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\nAudio File: {os.path.basename(result['audio_file'])}")
    print(f"Predicted Emotion: {result['predicted_emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if result['all_confidence_scores']:
        print("\nConfidence Scores for All Emotions:")
        sorted_scores = sorted(result['all_confidence_scores'].items(), 
                             key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_scores:
            print(f"   {emotion}: {score:.3f}")


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict emotions from audio files using trained ML models')
    parser.add_argument('--all-models', action='store_true', 
                       help='Process all audio files with all available models')
    parser.add_argument('--all', action='store_true', 
                       help='Process all audio files with a single model')
    parser.add_argument('--features', type=int, default=40, 
                       help='Number of MFCC features (default: 40)')
    parser.add_argument('audio_file', nargs='?', help='Path to audio file')
    parser.add_argument('model_path', nargs='?', help='Path to model file')
    
    args = parser.parse_args()
    
    # Check if user wants to process all audio files with all models
    if args.all_models:
        try:
            results = predict_all_audio_files_all_models(n_mfcc=args.features, save_csv=False)
            print(f"\nProcessed {len(results)} audio files with all models successfully!")
            return results
        except Exception as e:
            print(f"Error: {str(e)}")
        return {}
    
    # Check if user wants to process all audio files with single model
    if args.all:
        model_path = args.model_path
        try:
            if model_path:
                # Use single model
                results = predict_all_audio_files(model_path=model_path, n_mfcc=args.features)
            else:
                # Use all models
                results = predict_all_audio_files_all_models(n_mfcc=args.features)
            print(f"\nProcessed {len(results)} audio files successfully!")
        except Exception as e:
            print(f"Error: {str(e)}")
        return
    
    # Original single file processing
    if not args.audio_file:
        print("Usage: python audio_predictor_ML.py <audio_file_path> [model_path]")
        print("       python audio_predictor_ML.py --all [model_path]")
        print("       python audio_predictor_ML.py --all-models")
        print("Example: python audio_predictor_ML.py test_audio.wav")
        print("Example: python audio_predictor_ML.py test_audio.wav saved_models_ML/gbc_40.pkl")
        print("Example: python audio_predictor_ML.py --all")
        print("Example: python audio_predictor_ML.py --all saved_models_ML/gbc_40.pkl")
        print("Example: python audio_predictor_ML.py --all-models")
        return
    
    audio_file = args.audio_file
    model_path = args.model_path
    
    try:
        if model_path:
            # Use single model
            result = predict_emotion(audio_file, model_path, n_mfcc=args.features)
            print_prediction_result(result)
        else:
            # Use all models
            result = predict_emotion_all_models(audio_file, n_mfcc=args.features)
            print_prediction_result_all_models(result)
    except Exception as e:
        print(f"Error: {str(e)}")


def print_prediction_result_all_models(result):
    """
    Print prediction results from all models in a formatted way.
    """
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\nAudio File: {result['filename']}")
    print(f"Full Path: {result['audio_file']}")
    print(f"{'='*50}")
    
    for model_name, model_result in result['models'].items():
        print(f"\nModel: {model_name}")
        if 'error' in model_result:
            print(f"  Error: {model_result['error']}")
        else:
            print(f"  Predicted Emotion: {model_result['predicted_emotion']}")
            print(f"  Confidence: {model_result['confidence']:.3f}")
            
            if model_result['all_confidence_scores']:
                print("  Confidence Scores for All Emotions:")
                sorted_scores = sorted(model_result['all_confidence_scores'].items(), 
                                     key=lambda x: x[1], reverse=True)
                for emotion, score in sorted_scores:
                    print(f"    {emotion}: {score:.3f}")


if __name__ == "__main__":
    main() 
