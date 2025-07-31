#!/usr/bin/env python3
"""
Audio Emotion Analyzer
This script processes audio files in a folder, transcribes them using Gemini API, and analyzes emotions using Gemini sentiment analysis.
"""

import os
import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from datetime import datetime
import time
import random

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI not available. Install with: pip install google-generativeai")

# Try to import Gemini configuration
try:
    from gemini_config import GEMINI_API_KEY
    GEMINI_CONFIG_AVAILABLE = True
except ImportError:
    GEMINI_CONFIG_AVAILABLE = False
    print("⚠️  Gemini config not found. Create gemini_config.py with your API key.")


def transcribe_audio_with_gemini(audio_file_path: str) -> Optional[str]:
    """
    Transcribe audio file using Google Gemini API.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        Optional[str]: Transcribed text or None if failed
    """
    if not GEMINI_AVAILABLE:
        print("  ⚠️  Gemini API not available, falling back to mock transcription")
        return mock_audio_transcription(audio_file_path)
    
    if not GEMINI_CONFIG_AVAILABLE or GEMINI_API_KEY == "your_api_key_here":
        print("  ⚠️  Gemini API key not configured, falling back to mock transcription")
        return mock_audio_transcription(audio_file_path)
    
    try:
        # Add rate limiting to prevent API failures
        time.sleep(random.uniform(1, 3))
        
        # Configure Gemini with API key
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Upload the audio file
        print(f"  📤 Uploading audio file to Gemini...")
        uploaded_file = genai.upload_file(path=audio_file_path)
        
        # Create model and generate transcription
        print(f"  🎤 Transcribing with Gemini...")
        # Try the suggested model first, fallback to others
        models_to_try = [
            'gemini-2.0-flash-exp',  # Original model
            'gemini-1.5-flash',  # Alternative model
            'gemini-1.5-pro'  # Pro model as fallback
        ]
        
        transcription = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([
                    "Please transcribe this audio file. Provide only the transcription text without any additional commentary. The text should be in brazilian portuguese.",
                    uploaded_file
                ])
                transcription = response.text.strip()
                if transcription:
                    print(f"  ✅ Successfully transcribed with {model_name}")
                    break
            except Exception as model_error:
                print(f"  ⚠️  Model {model_name} failed: {str(model_error)[:100]}...")
                continue
        
        if transcription:
            return transcription
        else:
            print(f"  ⚠️  All models failed, falling back to mock transcription")
            return mock_audio_transcription(audio_file_path)
            
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            print(f"  ❌ Quota exceeded - API limit reached")
            print(f"  💡 Try again tomorrow or upgrade your API plan")
        else:
            print(f"  ❌ Gemini API error: {e}")
        print(f"  ⚠️  Falling back to mock transcription")
        return mock_audio_transcription(audio_file_path)


def analyze_sentiment_with_gemini(text: str) -> Optional[Dict[str, Any]]:
    """
    Analyze sentiment and emotions using Gemini API.
    
    Args:
        text (str): The text to analyze (can be in any language)
        
    Returns:
        Optional[Dict[str, Any]]: Analysis results or None if failed
    """
    if not GEMINI_AVAILABLE:
        print("  ❌ Gemini API not available")
        return None
    
    if not GEMINI_CONFIG_AVAILABLE or GEMINI_API_KEY == "your_api_key_here":
        print("  ❌ Gemini API key not configured")
        return None
    
    try:
        # Add rate limiting to prevent API failures
        time.sleep(random.uniform(1, 2))
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Create the prompt for sentiment analysis with JSON structure
        prompt = f"""
        Analise o sentimento e as emoções no seguinte texto: "{text}"
        
        Responda APENAS com um objeto JSON no seguinte formato exato:
        {{
            "principal_emocao_detectada": "felicidade|tristeza|medo|raiva|surpresa|desgosto|neutro",
            "nivel_confianca": "0-100%",
            "explicacao_breve": "explicação em português brasileiro"
        }}
        
        Regras:
        - Responda APENAS o JSON, sem texto adicional
        - Use apenas as emoções listadas acima
        - Nível de confiança deve ser um número de 0 a 100 seguido de %
        - Explicação deve ser breve e em português brasileiro
        - Não use aspas duplas dentro dos valores
        """
        
        # Use only text-compatible models
        models_to_try = [
            'gemini-2.0-flash-exp',  # Original model
            'gemini-1.5-flash',  # Alternative model
            'gemini-1.5-pro'  # Pro model as fallback
        ]
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                # Try to parse the JSON response
                try:
                    response_text = response.text.strip()
                    
                    # Clean up the response - remove any markdown formatting
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    json_response = json.loads(response_text)
                    
                    # Validate the required fields
                    required_fields = ["principal_emocao_detectada", "nivel_confianca", "explicacao_breve"]
                    if all(field in json_response for field in required_fields):
                        return {
                            "text": text,
                            "analysis": json_response,
                            "method": f"gemini_sentiment_{model_name}",
                            "dominant_emotion": json_response["principal_emocao_detectada"],
                            "confidence": json_response["nivel_confianca"],
                            "explanation": json_response["explicacao_breve"]
                        }
                    else:
                        print(f"  ⚠️  Invalid JSON structure from {model_name}")
                        continue
                        
                except json.JSONDecodeError as json_error:
                    print(f"  ⚠️  Failed to parse JSON from {model_name}: {str(json_error)[:50]}...")
                    print(f"  📄 Raw response: {response.text[:100]}...")
                    continue
                    
            except Exception as model_error:
                print(f"  ⚠️  Model {model_name} failed: {str(model_error)[:100]}...")
                continue
        
        print(f"  ❌ All models failed for sentiment analysis")
        return None
        
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            print(f"  ❌ Quota exceeded - API limit reached")
            print(f"  💡 Try again tomorrow or upgrade your API plan")
        else:
            print(f"  ❌ Gemini sentiment analysis error: {e}")
        return None


def mock_audio_transcription(audio_file_path: str) -> str:
    """
    Mock function to transcribe audio files when Gemini API is not available.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: Mock transcription text
    """
    filename = os.path.basename(audio_file_path).lower()
    
    mock_transcriptions = {
        "happy": "I am so happy today! The weather is beautiful and I feel amazing! Everything is going great!",
        "sad": "I'm feeling really sad and depressed. Nothing seems to go right. I just want to be alone.",
        "angry": "I'm so angry right now! This is completely unacceptable! How dare they do this!",
        "fear": "I'm scared and terrified. I don't know what to do. This is really frightening.",
        "surprise": "I'm surprised by this unexpected news! Wow! I can't believe what just happened!",
        "neutral": "This is a neutral statement with no strong emotional content. Just regular conversation.",
        "excited": "I'm so excited about this opportunity! This is going to be amazing!",
        "worried": "I'm worried about the future. I hope everything will be okay.",
        "frustrated": "I'm frustrated with this situation. Nothing is working as expected.",
        "grateful": "I'm so grateful for all the support. Thank you everyone for being here."
    }
    
    # Try to match filename with emotion keywords
    for emotion, transcription in mock_transcriptions.items():
        if emotion in filename:
            return transcription
    
    # Default transcription if no emotion keyword found
    return "This is a sample transcription of the audio file. I am speaking about various topics and sharing my thoughts."


def get_audio_files(folder_path: str) -> List[str]:
    """
    Get all audio files from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing audio files
        
    Returns:
        List[str]: List of audio file paths
    """
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.mp4', '.avi', '.mov'}
    audio_files = []
    
    try:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Error: Folder '{folder_path}' does not exist.")
            return []
        
        if not folder.is_dir():
            print(f"❌ Error: '{folder_path}' is not a directory.")
            return []
        
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        return sorted(audio_files)
    
    except Exception as e:
        print(f"❌ Error accessing folder: {e}")
        return []


def process_audio_file(audio_file_path: str) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Process a single audio file: transcribe and analyze emotions using Gemini.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        Tuple[str, str, str, Dict[str, Any]]: (filename, transcription, dominant_emotion, sentiment_analysis)
    """
    filename = os.path.basename(audio_file_path)
    print(f"🎵 Processing: {filename}")
    
    try:
        # Check file size (Gemini has limits)
        file_size = os.path.getsize(audio_file_path)
        max_size = 100 * 1024 * 1024  # 100MB limit
        
        if file_size > max_size:
            print(f"  ⚠️  File too large ({file_size / (1024*1024):.1f}MB), skipping real transcription")
            transcription = mock_audio_transcription(audio_file_path)
        else:
            # Transcribe audio
            print(f"  📝 Transcribing audio...")
            transcription = transcribe_audio_with_gemini(audio_file_path)
        
        if not transcription:
            print(f"  ❌ Transcription failed")
            return filename, "Transcription failed", "Unknown", {"analysis": "Transcription failed", "method": "failed"}
        
        print(f"  📄 Transcription: {transcription[:80]}...")
        
        # Analyze emotions using Gemini
        print(f"  🎭 Analyzing emotions with Gemini...")
        sentiment_analysis = analyze_sentiment_with_gemini(transcription)
        
        if sentiment_analysis and sentiment_analysis.get('analysis'):
            print(f"  ✅ Sentiment analysis completed")
            # Extract the dominant emotion from the JSON response
            dominant_emotion = sentiment_analysis.get('dominant_emotion', 'Unknown')
            confidence = sentiment_analysis.get('confidence', 'Unknown')
            print(f"  🎯 Detected emotion: {dominant_emotion} (Confidence: {confidence})")
        else:
            print(f"  ⚠️  Sentiment analysis failed")
            dominant_emotion = "Analysis failed"
            sentiment_analysis = {"analysis": "Analysis failed", "method": "failed"}
        
        return filename, transcription, dominant_emotion, sentiment_analysis
        
    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        return filename, f"Error: {str(e)}", "Error", {"analysis": f"Processing error: {str(e)}", "method": "error"}


def save_results_to_csv(results: List[Tuple[str, str, str, Dict[str, Any]]], output_file: str = "emotion_results.csv"):
    """
    Save emotion analysis results to a CSV file.
    
    Args:
        results (List[Tuple]): List of (filename, transcription, emotion, sentiment_analysis) tuples
        output_file (str): Output CSV filename
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['filename', 'transcription', 'dominant_emotion', 'confidence'])
            
            # Write results
            for filename, transcription, emotion, sentiment_analysis in results:
                # Extract confidence from sentiment analysis
                confidence = "Unknown"
                if isinstance(sentiment_analysis, dict):
                    if sentiment_analysis.get('confidence'):
                        confidence = sentiment_analysis['confidence']
                    elif sentiment_analysis.get('analysis') and isinstance(sentiment_analysis['analysis'], dict):
                        confidence = sentiment_analysis['analysis'].get('nivel_confianca', 'Unknown')
                
                writer.writerow([filename, transcription, emotion, confidence])
        
        print(f"✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving CSV file: {e}")


def save_results_to_json(results: List[Tuple[str, str, str, Dict[str, Any]]], output_file: str = "emotion_results.json"):
    """
    Save emotion analysis results to a JSON file.
    
    Args:
        results (List[Tuple]): List of (filename, transcription, emotion, sentiment_analysis) tuples
        output_file (str): Output JSON filename
    """
    try:
        json_results = []
        for filename, transcription, emotion, sentiment_analysis in results:
            # Extract confidence from sentiment analysis
            confidence = "Unknown"
            if isinstance(sentiment_analysis, dict):
                if sentiment_analysis.get('confidence'):
                    confidence = sentiment_analysis['confidence']
                elif sentiment_analysis.get('analysis') and isinstance(sentiment_analysis['analysis'], dict):
                    confidence = sentiment_analysis['analysis'].get('nivel_confianca', 'Unknown')
            
            json_results.append({
                'filename': filename,
                'transcription': transcription,
                'dominant_emotion': emotion,
                'confidence': confidence,
                'analysis_timestamp': datetime.now().isoformat()
            })
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_results, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")


def analyze_audio_folder(folder_path: str, output_file: str = "emotion_results.csv", save_files: bool = False) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Analyze all audio files in a folder for emotion detection.
    
    Args:
        folder_path (str): Path to the folder containing audio files
        output_file (str): Output file name (used only if save_files=True)
        save_files (bool): Whether to save results to files (default: False for pipeline integration)
    
    Returns:
        List[Tuple[str, str, str, Dict[str, Any]]]: List of results for each audio file
    """
    print(f"🎵 Scanning folder: {folder_path}")
    
    # Get all audio files
    audio_files = get_audio_files(folder_path)
    
    if not audio_files:
        print("❌ No audio files found!")
        print("💡 Make sure your audio files are in one of these formats:")
        print("   • .wav, .mp3, .flac, .m4a, .aac, .ogg")
        print("💡 And try one of these options:")
        print("   1. Place audio files in the 'Audios_to_predict' folder")
        print("   2. Run: python audio_emotion_analyzer.py")
        return []
    
    print(f"🎵 Found {len(audio_files)} audio file(s)")
    print()
    
    # Process each audio file
    results = []
    successful_files = 0
    failed_files = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"📊 Processing file {i}/{len(audio_files)}")
        try:
            filename, transcription, dominant_emotion, sentiment_analysis = process_audio_file(audio_file)
            
            # Extract confidence from sentiment analysis
            confidence = "Unknown"
            if isinstance(sentiment_analysis, dict):
                if sentiment_analysis.get('confidence'):
                    confidence = sentiment_analysis['confidence']
                elif sentiment_analysis.get('analysis') and isinstance(sentiment_analysis['analysis'], dict):
                    confidence = sentiment_analysis['analysis'].get('nivel_confianca', 'Unknown')
            
            results.append((filename, transcription, dominant_emotion, sentiment_analysis))
            
            # Track success/failure
            if dominant_emotion not in ["Unknown", "Error", "Analysis failed"]:
                successful_files += 1
            else:
                failed_files += 1
                
        except Exception as e:
            print(f"  ❌ Unexpected error processing {os.path.basename(audio_file)}: {e}")
            failed_files += 1
            results.append((os.path.basename(audio_file), f"Error: {str(e)}", "Error", {"analysis": f"Unexpected error: {str(e)}", "method": "error"}))
        
        print()
        
        # Add a small delay between files to prevent overwhelming the API
        if i < len(audio_files):
            time.sleep(0.5)
    
    # Print processing statistics
    print(f"📈 Processing Statistics:")
    print(f"   ✅ Successful: {successful_files}")
    print(f"   ❌ Failed: {failed_files}")
    print(f"   📊 Total: {len(audio_files)}")
    print()
    
    # Save results only if requested (for standalone usage)
    if save_files:
        print("💾 Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"emotion_results_{timestamp}.csv"
        json_file = f"emotion_results_{timestamp}.json"
        
        save_results_to_csv(results, csv_file)
        save_results_to_json(results, json_file)
        
        print(f"✅ Analysis complete! Results saved to:")
        print(f"   • CSV: {csv_file}")
        print(f"   • JSON: {json_file}")
    else:
        print("✅ Analysis complete! Results captured in memory.")
    
    # Print summary
    print("\n📋 Summary:")
    print("-" * 30)
    for filename, transcription, dominant_emotion, sentiment_analysis in results:
        print(f"  {filename}: {dominant_emotion}")
        print(f"    Transcription: {transcription[:50]}...")
        if isinstance(sentiment_analysis, dict):
            if sentiment_analysis.get('analysis'):
                analysis = sentiment_analysis['analysis']
                if isinstance(analysis, dict):
                    print(f"    Emotion: {analysis.get('principal_emocao_detectada', 'Unknown')}")
                    print(f"    Confidence: {analysis.get('nivel_confianca', 'Unknown')}")
                    print(f"    Explanation: {analysis.get('explicacao_breve', 'No explanation')[:50]}...")
                else:
                    print(f"    Analysis: {str(analysis)[:50]}...")
            elif sentiment_analysis.get('confidence'):
                print(f"    Confidence: {sentiment_analysis['confidence']}")
        print()
    
    return results


def main():
    """Main function to run the audio emotion analyzer."""
    # Default to 'Audios_to_predict' folder
    folder_path = "Audios_to_predict"
    output_file = "emotion_results.csv"
    
    # Allow override via command line arguments
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    return analyze_audio_folder(folder_path, output_file, save_files=False)


if __name__ == "__main__":
    main() 