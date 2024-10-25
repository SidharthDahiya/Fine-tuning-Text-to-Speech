import torch
from TTS.api import TTS
import os
import json
import pandas as pd
from datetime import datetime
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class HindiTTSTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.setup_directories()
        self.setup_dataset()

        # Initialize model with better error handling
        if self.initialize_model():
            print("Model initialized successfully")
        else:
            print("Failed to initialize any model")

    def setup_directories(self):
        self.dirs = {
            'output': 'hindi_output',
            'eval': 'hindi_evaluation',
            'samples': 'hindi_samples',
            'reference': 'reference_audio',
            'checkpoints': 'model_checkpoints'
        }
        for dir_name in self.dirs.values():
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")

    def setup_dataset(self):
        """Setup Hindi dataset for training"""
        self.dataset = [
            {
                'text': 'नमस्ते दोस्तों',
                'category': 'greeting'
            },
            {
                'text': 'आज का मौसम बहुत अच्छा है',
                'category': 'weather'
            },
            {
                'text': 'मशीन लर्निंग एक रोचक विषय है',
                'category': 'technical'
            },
            {
                'text': 'कंप्यूटर विज्ञान एक महत्वपूर्ण विषय है',
                'category': 'technical'
            },
            {
                'text': 'आर्टिफिशियल इंटेलिजेंस का प्रयोग बढ़ रहा है',
                'category': 'technical'
            }
        ]

    def initialize_model(self):
        try:
            print("\nInitializing TTS model...")

            # Try XTTS-v2 model which has better multilingual support
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("Successfully loaded XTTS-v2 model")

            # Generate initial reference in English
            initial_reference = os.path.join(self.dirs['reference'], 'initial_reference.wav')
            if not os.path.exists(initial_reference):
                print("Generating initial reference...")
                # Use English for initial reference
                self.model.tts_to_file(
                    text="This is a reference audio for voice cloning.",
                    file_path=initial_reference,
                    language="en",
                    speaker_wav=None  # For first generation
                )
                print("Generated initial reference successfully")

            # Use the initial reference to generate Hindi reference
            self.reference_audio = os.path.join(self.dirs['reference'], 'hindi_reference.wav')
            if not os.path.exists(self.reference_audio):
                print("Generating Hindi reference...")
                self.model.tts_to_file(
                    text="नमस्ते, मैं हिंदी बोल सकता हूं",
                    file_path=self.reference_audio,
                    language="hi",
                    speaker_wav=initial_reference
                )
                print("Generated Hindi reference successfully")

            print("Reference audio setup completed")
            return True

        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            try:
                # Fallback to basic multilingual model
                print("Attempting to load basic multilingual model...")
                self.model = TTS("tts_models/multilingual/multi-dataset/bark")
                print("Successfully loaded fallback model")
                return True
            except Exception as e:
                print(f"Error loading fallback model: {str(e)}")
                self.model = None
                return False

    def fine_tune(self, learning_rate=1e-4, batch_size=16, epochs=5):
        if self.model is None:
            print("Model not initialized. Cannot proceed with fine-tuning.")
            return

        if not hasattr(self, 'reference_audio'):
            print("No reference audio available")
            return

        print("\nStarting fine-tuning process for Hindi TTS...")

        training_config = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }

        self.log_training('start', training_config)

        try:
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")

                for idx, item in enumerate(self.dataset):
                    try:
                        output_path = os.path.join(
                            self.dirs['output'],
                            f"epoch_{epoch}_sample_{idx}.wav"
                        )

                        self.model.tts_to_file(
                            text=item['text'],
                            file_path=output_path,
                            language="hi",
                            speaker_wav=self.reference_audio
                        )

                        print(f"✓ Processed sample {idx + 1}/{len(self.dataset)}")

                    except Exception as e:
                        print(f"✗ Error processing sample {idx + 1}: {str(e)}")

                # Save checkpoint
                self.save_checkpoint(epoch)

        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        try:
            checkpoint_path = os.path.join(self.dirs['checkpoints'], f'checkpoint_epoch_{epoch}.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def evaluate(self, test_sentences=None):
        """Evaluate the model"""
        if self.model is None:
            print("Model not initialized. Cannot proceed with evaluation.")
            return

        if not hasattr(self, 'reference_audio'):
            print("No reference audio available")
            return

        if test_sentences is None:
            test_sentences = [item['text'] for item in self.dataset]

        results = []
        print("\nEvaluating model...")

        for idx, text in enumerate(test_sentences):
            try:
                output_path = os.path.join(self.dirs['samples'], f'test_{idx}.wav')

                start_time = datetime.now()
                self.model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=self.reference_audio,
                    language="hi"
                )
                inference_time = (datetime.now() - start_time).total_seconds()

                audio_metrics = self.analyze_audio(output_path)

                results.append({
                    'text': text,
                    'inference_time': inference_time,
                    **audio_metrics
                })
                print(f"✓ Evaluated: {text[:30]}...")

            except Exception as e:
                print(f"✗ Error evaluating: {str(e)}")

        if results:
            self.generate_report(results)
        return results

    def analyze_audio(self, audio_path):
        """Analyze audio file and extract metrics"""
        try:
            audio, sr = librosa.load(audio_path)
            return {
                'duration': float(librosa.get_duration(y=audio, sr=sr)),
                'rms_energy': float(librosa.feature.rms(y=audio)[0].mean()),
                'zero_crossings': int(librosa.zero_crossings(audio).sum())
            }
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return {}

    def log_training(self, stage, data):
        """Log training progress"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'data': data
        }

        log_file = os.path.join(self.dirs['output'], 'training_log.json')

        with open(log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

    def generate_report(self, results):
        """Generate evaluation report"""
        report_path = os.path.join(self.dirs['eval'], 'evaluation_report.html')
        df = pd.DataFrame(results)

        html_content = f"""
        <html>
        <head>
            <title>Hindi TTS Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                .stats {{ background-color: #f9f9f9; padding: 15px; }}
                .hindi {{ font-family: 'Noto Sans Devanagari', sans-serif; }}
            </style>
        </head>
        <body>
            <h1>Hindi TTS Evaluation Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="stats">
                <h2>Summary Statistics</h2>
                <p>Total samples: {len(df)}</p>
                <p>Average inference time: {df['inference_time'].mean():.3f} seconds</p>
                <p>Average duration: {df['duration'].mean():.3f} seconds</p>
            </div>

            <h2>Detailed Results</h2>
            {df.to_html(classes='hindi')}
        </body>
        </html>
        """

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nEvaluation report saved to: {report_path}")


def main():
    print("Initializing Hindi TTS System...")
    trainer = HindiTTSTrainer()

    if trainer.model is not None:
        print("\nStarting fine-tuning...")
        trainer.fine_tune()

        print("\nStarting evaluation...")
        trainer.evaluate()

        print("\nProcess completed!")
    else:
        print("Cannot proceed due to model initialization failure.")


if __name__ == "__main__":
    main()