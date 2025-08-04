# Audio Emotion Recognition System

A comprehensive audio emotion recognition system that combines Machine Learning (ML), Deep Learning (DL), and Text Analysis to classify emotions from audio files.

## Features

### **Training Pipeline**
- **Default**: 40 MFCC features, ML + DL models, standard datasets (Tess, Ravdess, Savee, CREMA-D)
- **Customizable**: Number of features, ML/DL selection, custom datasets
- **Multiple Models**: MLP, CNN, LSTM, GRU (DL) + Gradient Boosting, LightGBM, Random Forest (ML)

### **Prediction Pipeline**
- **Multi-Modal Analysis**: ML + DL + Text Analysis (Gemini transcription + sentiment)
- **Flexible Model Selection**: Use all models or specific ones
- **Unified Output**: Single JSON file with predictions, confidence scores, and transcriptions

## 📁 Project Structure

```
TCC_final/
├── main.py                          # Main orchestrator
├── feature_extraction.py            # Audio feature extraction
├── trainingML.py                    # ML model training
├── trainingDL.py                    # DL model training
├── audio_predictor_ML.py           # ML prediction
├── audio_predictor_DL.py           # DL prediction
├── text_transcription_analyzer.py   # Text analysis (Gemini)
├── gemini_config.py                # Gemini API configuration
├── requirements.txt                 # Dependencies
├── Dataset/                         # Standard datasets
│   ├── Tess/
│   ├── Ravdess/
│   ├── Savee/
│   └── Crema/
├── Custom_dataset/                 # Custom dataset folder for training
├── Audios_to_predict/              # Audio files for prediction
└── Utils/
    └── emotion_number_dict.py      # Emotion mapping
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/aisepucrio/SpeechEmotion.git
cd TCC_final
```

### 2. Setup Environment

Make sure you have Python 3.11.8 installed.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Gemini API (Optional)

Edit `gemini_config.py` and add your Gemini API key:
```python
GEMINI_API_KEY = "your_actual_api_key_here"
```

### 4. Prepare Data

#### Standard Datasets
Place your datasets in the `Dataset/` folder:
- `Dataset/Tess/` - TESS dataset
- `Dataset/Ravdess/` - RAVDESS dataset  
- `Dataset/Savee/` - SAVEE dataset
- `Dataset/Crema/` - CREMA-D dataset

#### Custom Dataset
Place audio files in `Custom_dataset/` with format: `<emotion>_name.wav`
```
Custom_dataset/
├── happy_meeting1.wav
├── sad_conversation.mp3
├── angry_call.m4a
└── subfolder/
    └── neutral_discussion.wav
```

#### Prediction Files
Place audio files for prediction in `Audios_to_predict/`:
```
Audios_to_predict/
├── test1.wav
├── test2.mp3
└── subfolder/
    └── test3.flac
```

## 🎓 Training

### Default Training
```bash
# Train with default settings (40 features, ML + DL, standard datasets)
python main.py train
```

### Custom Training Options
```bash
# Train with 60 features
python main.py train --features 60

# Train only ML models
python main.py train --ml-only

# Train only DL models  
python main.py train --dl-only

# Use custom dataset
python main.py train --custom-dataset

# Combine options
python main.py train --features 60 --ml-only --custom-dataset
```

## 🔮 Prediction

### Default Prediction
```bash
# Predict with all models and analysis types
python main.py predict
```

### Custom Prediction Options
```bash
# Use specific models
python main.py predict --models gbc_40 lightgbm_40

# ML only
python main.py predict --ml-only

# DL only
python main.py predict --dl-only

# Text analysis only
python main.py predict --text-only

# Custom audio folder
python main.py predict --audio-folder my_audio_files

# Custom output file
python main.py predict --output my_results.json
```

## 📋 Model Management

### List Available Models
```bash
python main.py list-models
```

### Check Model Status
The system will automatically detect available models:
- **ML Models**: Stored in `Saved Models/` folder
- **DL Models**: Stored as `.keras` files
- **Text Analysis**: Uses Gemini API (requires configuration)

## 📊 Output Files

### Training Outputs
- `output{features}.csv` - Standard dataset features
- `custom_output{features}.csv` - Custom dataset features
- `saved_models_ML/` - ML model files and metrics
- `saved_models_DL/` - DL model files and metrics

### Prediction Outputs
- `prediction_results_{timestamp}.json` - Combined results
- `emotion_predictions_{timestamp}.csv` - ML prediction results
- `emotion_results_{timestamp}.csv` - Text analysis results

* The datasets can be found in: 
  - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
  - https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
  - https://www.kaggle.com/datasets/ejlok1/cremad
  - https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-saveec
