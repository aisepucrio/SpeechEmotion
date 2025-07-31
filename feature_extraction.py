import librosa
import numpy as np
import csv
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob

def extract_features(file_path, n_mfcc=13):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Apply pre-emphasis filter
        y_emphasized = librosa.effects.preemphasis(y_trimmed)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y_emphasized, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y_emphasized, sr=sr)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_emphasized, sr=sr)
        
        # Calculate statistics for each coefficient
        features = []
        
        # MFCC features
        for mfcc, delta, delta2 in zip(mfccs, delta_mfccs, delta2_mfccs):
            features.extend([
                np.mean(mfcc),
                np.std(mfcc),
                np.mean(delta),
                np.std(delta),
                np.mean(delta2),
                np.std(delta2)
            ])
        
        # Chroma features
        for chroma_coef in chroma:
            features.extend([
                np.mean(chroma_coef),
                np.std(chroma_coef)
            ])
        
        # Mel spectrogram features
        for mel_coef in mel_spec:
            features.extend([
                np.mean(mel_coef),
                np.std(mel_coef)
            ])
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_to_csv(path_to_data_folder, num_features, datasets_to_load):
    # Create header for MFCC features
    header = []
    for i in range(num_features):
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
    
    # Process TESS dataset
    if 'Tess' in datasets_to_load:
        tess_path = os.path.join(path_to_data_folder, 'Dataset', 'Tess')
        if not os.path.exists(tess_path):
            print(f"Warning: TESS dataset path not found: {tess_path}")
        else:
            for emotion in os.listdir(tess_path):
                if emotion != '.DS_Store':
                    emotion_path = os.path.join(tess_path, emotion)
                    for audio_file in os.listdir(emotion_path):
                        if audio_file.endswith('.wav'):
                            file_path = os.path.join(emotion_path, audio_file)
                            # Extract emotion from filename (e.g., "OAF_angry.wav" -> "angry")
                            emotion = audio_file.split('_')[2].split('.')[0].lower()
                            if emotion == 'ps':
                                emotion = 'surprised'
                            features = extract_features(file_path, n_mfcc=num_features)
                            if features is not None:
                                features_list.append([file_path, emotion] + features.tolist())
    
    # Process RAVDESS dataset
    if 'Ravdess' in datasets_to_load:
        ravdess_path = os.path.join(path_to_data_folder, 'Dataset', 'Ravdess', 'audio_speech_actors_01-24')
        if not os.path.exists(ravdess_path):
            print(f"Warning: RAVDESS dataset path not found: {ravdess_path}")
        else:
            for actor_folder in os.listdir(ravdess_path):
                if actor_folder.startswith('Actor_'):
                    actor_path = os.path.join(ravdess_path, actor_folder)
                    for audio_file in os.listdir(actor_path):
                        if audio_file.endswith('.wav'):
                            file_path = os.path.join(actor_path, audio_file)
                            # Extract emotion from filename
                            emotion_code = audio_file.split('-')[2]
                            emotion = {
                                '01': 'neutral',
                                '02': 'calm',
                                '03': 'happy',
                                '04': 'sad',
                                '05': 'angry',
                                '06': 'fear',
                                '07': 'disgust',
                                '08': 'surprised'
                            }.get(emotion_code, 'none')

                            if emotion == 'calm':
                                continue
                            
                            features = extract_features(file_path, n_mfcc=num_features)
                            if features is not None:
                                features_list.append([file_path, emotion] + features.tolist())
    
    # Process CREMA-D dataset
    if 'Crema' in datasets_to_load:
        crema_path = os.path.join(path_to_data_folder, 'Dataset', 'Crema')
        if not os.path.exists(crema_path):
            print(f"Warning: CREMA-D dataset path not found: {crema_path}")
        else:
            for audio_file in os.listdir(crema_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(crema_path, audio_file)
                    # Extract emotion from filename
                    emotion_code = audio_file.split('_')[2]
                    emotion = {
                        'NEU': 'neutral',
                        'HAP': 'happy',
                        'SAD': 'sad',
                        'ANG': 'angry',
                        'FEA': 'fear',
                        'DIS': 'disgust'
                    }.get(emotion_code, 'none')
                    
                    features = extract_features(file_path, n_mfcc=num_features)
                    if features is not None:
                        features_list.append([file_path, emotion] + features.tolist())
    
    # Process SAVEE dataset
    if 'Savee' in datasets_to_load:
        savee_path = os.path.join(path_to_data_folder, 'Dataset', 'Savee')
        if not os.path.exists(savee_path):
            print(f"Warning: SAVEE dataset path not found: {savee_path}")
        else:
            for audio_file in os.listdir(savee_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(savee_path, audio_file)
                    # Extract emotion from filename
                    emotion_code = audio_file[3:5]
                    if emotion_code[1] not in ['a', 'u']:
                        emotion_code = emotion_code[0]
                    emotion = {
                        'a': 'angry',
                        'd': 'disgust',
                        'f': 'fear',
                        'h': 'happy',
                        'n': 'neutral',
                        'sa': 'sad',
                        'su': 'surprised'
                    }.get(emotion_code, 'none')
                    
                    features = extract_features(file_path, n_mfcc=num_features)
                    if features is not None:
                        features_list.append([file_path, emotion] + features.tolist())
    
    if not features_list:
        print("Error: No features were extracted. Please check if the dataset paths are correct.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list, columns=header)
    
    # Save to CSV
    df.to_csv(f'output{num_features}.csv')
    print(f"Features saved to output{num_features}.csv")
    print(f"Total number of samples: {len(df)}")
    print("\nEmotion distribution:")
    print(df['Emotion'].value_counts())

def extract_to_csv_custom(path_to_data_folder, num_features):
    """
    Extract features from audio files in Custom_dataset folder.
    Expected filename format: '<emotion>_name' where <emotion> is extracted as the emotion label.
    
    Args:
        path_to_data_folder (str): Path to the data folder
        num_features (int): Number of MFCC features to extract
    """
    # Create header for MFCC features
    header = []
    for i in range(num_features):
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
    
    # Path to Custom_dataset folder
    custom_dataset_path = os.path.join(path_to_data_folder, 'Custom_dataset')
    
    if not os.path.exists(custom_dataset_path):
        print(f"Error: Custom_dataset folder not found: {custom_dataset_path}")
        return
    
    # Find all audio files recursively in Custom_dataset folder
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac', '*.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        # Search in the main folder and all subfolders
        pattern = os.path.join(custom_dataset_path, '**', ext)
        audio_files.extend(glob.glob(pattern, recursive=True))
        # Also search in the main folder only
        pattern_main = os.path.join(custom_dataset_path, ext)
        audio_files.extend(glob.glob(pattern_main))
    
    # Remove duplicates and sort
    audio_files = sorted(list(set(audio_files)))
    
    if not audio_files:
        print(f"No audio files found in Custom_dataset folder")
        return
    
    print(f"Found {len(audio_files)} audio files in Custom_dataset folder")
    
    # Process each audio file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"Processing file {i}/{len(audio_files)}: {os.path.basename(audio_file)}")
        
        try:
            # Extract emotion from filename
            filename = os.path.basename(audio_file)
            filename_without_ext = os.path.splitext(filename)[0]  # Remove file extension
            
            # Split by underscore and get the first part as emotion
            parts = filename_without_ext.split('_')
            if len(parts) >= 2:
                emotion = parts[0].lower()  # Convert to lowercase for consistency
                print(f"  Extracted emotion: {emotion}")
            else:
                print(f"  Warning: Filename '{filename}' doesn't match expected format '<emotion>_name'")
                emotion = 'unknown'
            
            # Extract features
            features = extract_features(audio_file, n_mfcc=num_features)
            if features is not None:
                features_list.append([audio_file, emotion] + features.tolist())
                print(f"  ✓ Features extracted successfully")
            else:
                print(f"  ✗ Failed to extract features")
                
        except Exception as e:
            print(f"  ✗ Error processing {audio_file}: {str(e)}")
    
    if not features_list:
        print("Error: No features were extracted. Please check if the audio files are valid.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list, columns=header)
    
    # Save to CSV
    output_filename = f'custom_output{num_features}.csv'
    df.to_csv(output_filename)
    print(f"\nFeatures saved to {output_filename}")
    print(f"Total number of samples: {len(df)}")
    print("\nEmotion distribution:")
    print(df['Emotion'].value_counts())
    
    return output_filename