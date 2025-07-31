#!/usr/bin/env python3
"""
Pipeline Orchestrator for Audio Emotion Recognition
This script orchestrates the complete pipeline for training and prediction with multiple options.
"""

import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd
import numpy as np
import glob # Added for loading prediction results

# Import our modules
import feature_extraction
import trainingML
import trainingDL
import audio_predictor_ML
import audio_predictor_DL
import text_transcription_analyzer

class PipelineOrchestrator:
    def __init__(self):
        self.path_to_data_folder = '.\\'
        self.default_datasets = ['Ravdess', 'Crema', 'Tess', 'Savee']
        self.default_features = 40
        
    def train_pipeline(self, num_features=None, ml_only=False, dl_only=False, custom_dataset=False):
        """
        Training pipeline with options for features, ML/DL selection, and dataset type.
        
        Args:
            num_features (int): Number of MFCC features (default: 40)
            ml_only (bool): Train only ML models
            dl_only (bool): Train only DL models
            custom_dataset (bool): Use custom dataset instead of standard datasets
        """
        if num_features is None:
            num_features = self.default_features
            
        print("=" * 60)
        print("🎯 TRAINING PIPELINE")
        print("=" * 60)
        print(f"Features: {num_features}")
        print(f"ML only: {ml_only}")
        print(f"DL only: {dl_only}")
        print(f"Custom dataset: {custom_dataset}")
        print()
        
        # Feature extraction
        if custom_dataset:
            print("📁 Using Custom Dataset")
            filename = f"custom_output{num_features}.csv"
            print("Extracting features from Custom_dataset folder (overwriting existing file)...")
            feature_extraction.extract_to_csv_custom(self.path_to_data_folder, num_features)
            print(f"✅ Features extracted and saved to: {filename}")
        else:
            print("📁 Using Standard Datasets")
            filename = f"output{num_features}.csv"
            if not os.path.exists(filename):
                print("🔍 Extracting features from standard datasets...")
                feature_extraction.extract_to_csv(self.path_to_data_folder, num_features, self.default_datasets)
            else:
                print(f"✅ Features file already exists: {filename}")
        
        # Training models
        if not dl_only:
            print("\n🤖 Training ML Models...")
            try:
                trainingML.train_models(num_features, custom_dataset)
                print("✅ ML training completed")
            except Exception as e:
                print(f"❌ ML training failed: {e}")
        
        if not ml_only:
            print("\n🧠 Training DL Models...")
            try:
                trainingDL.train_models(num_features, custom_dataset)
                print("✅ DL training completed")
            except Exception as e:
                print(f"❌ DL training failed: {e}")
        
        print("\n🎉 Training pipeline completed!")
        
    def predict_pipeline(self, models=None, ml_only=False, dl_only=False, text_only=False, 
                        audio_folder="Audios_to_predict", output_file=None, num_features=40):
        """
        Prediction pipeline with options for model selection and analysis types.
        
        Args:
            models (list): Specific models to use (if None, uses all available)
            ml_only (bool): Use only ML models
            dl_only (bool): Use only DL models
            text_only (bool): Use only text analysis
            audio_folder (str): Folder containing audio files to predict
            output_file (str): Output file name
        """
        print("=" * 60)
        print("🔮 PREDICTION PIPELINE")
        print("=" * 60)
        print(f"Audio folder: {audio_folder}")
        print(f"ML only: {ml_only}")
        print(f"DL only: {dl_only}")
        print(f"Text only: {text_only}")
        print(f"Specific models: {models}")
        print(f"Features: {num_features}")
        print()
        
        # Generate output filename with timestamp
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"prediction_results_{timestamp}.json"
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "1.0",
            "audio_folder": audio_folder,
            "predictions": {}
        }
        
        # Text analysis (Gemini transcription + sentiment)
        if not dl_only and not ml_only or text_only:
            print("📝 Running Text Analysis...")
            try:
                # Call the text transcription analyzer's main function and capture results
                original_argv = sys.argv
                sys.argv = ['text_transcription_analyzer.py', audio_folder]
                text_results = text_transcription_analyzer.main()
                sys.argv = original_argv
                
                if text_results:
                    all_results['predictions']['text_analysis'] = {
                        "status": "completed",
                        "results": text_results
                    }
                    print("✅ Text analysis completed and results captured")
                else:
                    all_results['predictions']['text_analysis'] = {
                        "status": "completed",
                        "note": "Text analysis completed but no results returned"
                    }
                    print("✅ Text analysis completed")
            except Exception as e:
                print(f"❌ Text analysis failed: {e}")
                all_results['predictions']['text_analysis'] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # ML prediction
        if not dl_only and not text_only or ml_only:
            print("\n🤖 Running ML Prediction...")
            try:
                # Call the ML predictor's main function and capture results
                original_argv = sys.argv
                sys.argv = ['audio_predictor_ML.py', '--all-models', '--features', str(num_features)]
                ml_results = audio_predictor_ML.main()
                sys.argv = original_argv
                
                if ml_results:
                    all_results['predictions']['ml_prediction'] = {
                        "status": "completed",
                        "results": ml_results
                    }
                    print("✅ ML prediction completed and results captured")
                else:
                    all_results['predictions']['ml_prediction'] = {
                        "status": "completed",
                        "note": "ML prediction completed but no results returned"
                    }
                    print("✅ ML prediction completed")
            except Exception as e:
                print(f"❌ ML prediction failed: {e}")
                all_results['predictions']['ml_prediction'] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # DL prediction
        if not ml_only and not text_only or dl_only:
            print("\n🧠 Running DL Prediction...")
            try:
                # Call the DL predictor's main function and capture results
                original_argv = sys.argv
                sys.argv = ['audio_predictor_DL.py', '--audios-folder', '--features', str(num_features)]
                dl_results = audio_predictor_DL.main()
                sys.argv = original_argv
                
                if dl_results:
                    all_results['predictions']['dl_prediction'] = {
                        "status": "completed",
                        "results": dl_results
                    }
                    print("✅ DL prediction completed and results captured")
                else:
                    all_results['predictions']['dl_prediction'] = {
                        "status": "completed",
                        "note": "DL prediction completed but no results returned"
                    }
                    print("✅ DL prediction completed")
            except Exception as e:
                print(f"❌ DL prediction failed: {e}")
                all_results['predictions']['dl_prediction'] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Save combined results
        if all_results['predictions']:
            self._save_combined_results(all_results, output_file)
            print(f"\n💾 Combined results saved to: {output_file}")
        
        print("\n🎉 Prediction pipeline completed!")
    
    def _save_combined_results(self, all_results, output_file):
        """Save combined results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n✅ Combined results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving combined results: {e}")
    
    def list_available_models(self):
        """List all available trained models."""
        print("=" * 60)
        print("📋 AVAILABLE MODELS")
        print("=" * 60)
        
        # Check ML models
        try:
            ml_models = audio_predictor_ML.get_available_models()
            if ml_models:
                print("\n🤖 ML Models:")
                for model in ml_models:
                    print(f"  • {os.path.basename(model)}")
            else:
                print("\n🤖 ML Models: None found")
        except Exception as e:
            print(f"\n🤖 ML Models: Error checking - {e}")
        
        # Check DL models
        dl_model_paths = ['Saved Models/MLP_model.keras', 'Saved Models/CNN_model.keras', 
                         'Saved Models/LSTM_model.keras', 'Saved Models/GRU_model.keras']
        available_dl_models = []
        for model_path in dl_model_paths:
            if os.path.exists(model_path):
                available_dl_models.append(model_path)
        
        if available_dl_models:
            print("\n🧠 DL Models:")
            for model in available_dl_models:
                print(f"  • {os.path.basename(model)}")
        else:
            print("\n🧠 DL Models: None found")
        
        print("\n💡 To train models, use: python main.py train")

def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Recognition Pipeline Orchestrator')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Pipeline mode')
    
    # Training subparser
    train_parser = subparsers.add_parser('train', help='Training pipeline')
    train_parser.add_argument('--features', type=int, default=40, 
                            help='Number of MFCC features (default: 40)')
    train_parser.add_argument('--ml-only', action='store_true', 
                            help='Train only ML models')
    train_parser.add_argument('--dl-only', action='store_true', 
                            help='Train only DL models')
    train_parser.add_argument('--custom-dataset', action='store_true', 
                            help='Use custom dataset instead of standard datasets')
    
    # Prediction subparser
    predict_parser = subparsers.add_parser('predict', help='Prediction pipeline')
    predict_parser.add_argument('--features', type=int, default=40, 
                              help='Number of MFCC features (default: 40)')
    predict_parser.add_argument('--models', nargs='+', 
                              help='Specific models to use')
    predict_parser.add_argument('--ml-only', action='store_true', 
                              help='Use only ML models')
    predict_parser.add_argument('--dl-only', action='store_true', 
                              help='Use only DL models')
    predict_parser.add_argument('--text-only', action='store_true', 
                              help='Use only text analysis')
    predict_parser.add_argument('--audio-folder', default='Audios_to_predict', 
                              help='Folder containing audio files (default: Audios_to_predict)')
    predict_parser.add_argument('--output', 
                              help='Output file name')
    
    # List models subparser
    subparsers.add_parser('list-models', help='List available trained models')
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.mode == 'train':
        orchestrator.train_pipeline(
            num_features=args.features,
            ml_only=args.ml_only,
            dl_only=args.dl_only,
            custom_dataset=args.custom_dataset
        )
    
    elif args.mode == 'predict':
        orchestrator.predict_pipeline(
            models=args.models,
            ml_only=args.ml_only,
            dl_only=args.dl_only,
            text_only=args.text_only,
            audio_folder=args.audio_folder,
            output_file=args.output,
            num_features=args.features
        )
    
    elif args.mode == 'list-models':
        orchestrator.list_available_models()
    
    else:
        # Show help if no mode specified
        print("🎯 Audio Emotion Recognition Pipeline Orchestrator")
        print("=" * 60)
        print("\n📚 Usage Examples:")
        print("\n🎓 Training:")
        print("  # Default training (40 features, ML + DL, standard datasets)")
        print("  python main.py train")
        print("\n  # Custom training")
        print("  python main.py train --features 60 --ml-only --custom-dataset")
        print("\n🔮 Prediction:")
        print("  # Default prediction (all models, all analysis types)")
        print("  python main.py predict")
        print("\n  # Specific models")
        print("  python main.py predict --models gbc_40 lightgbm_40")
        print("\n  # ML only")
        print("  python main.py predict --ml-only")
        print("\n  # DL only")
        print("  python main.py predict --dl-only")
        print("\n  # Text analysis only")
        print("  python main.py predict --text-only")
        print("\n📋 List Models:")
        print("  python main.py list-models")
        print("\n❓ Help:")
        print("  python main.py --help")

if __name__ == '__main__':
    main() 