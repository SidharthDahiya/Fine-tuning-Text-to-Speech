import torch
from TTS.api import TTS
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import librosa
import time


class TTSBenchmark:
    def __init__(self):
        self.dirs = {
            'benchmark': 'benchmark_results',
            'audio': 'benchmark_audio',
            'reports': 'benchmark_reports'
        }
        self.setup_directories()

        # Initialize different TTS models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            'Glow-TTS': TTS("tts_models/en/ljspeech/glow-tts"),
            'Fast-Pitch': TTS("tts_models/en/ljspeech/fast_pitch")
        }

        # Technical terms with correct pronunciations
        self.technical_terms = {
            'API': {'text': 'A.P.I.', 'pronunciation': 'ay-pee-eye'},
            'CUDA': {'text': 'CUDA', 'pronunciation': 'koo-duh'},
            'TTS': {'text': 'T.T.S.', 'pronunciation': 'tee-tee-ess'},
            'REST': {'text': 'REST', 'pronunciation': 'rest'},
            'GPU': {'text': 'G.P.U.', 'pronunciation': 'gee-pee-you'}
        }

    def setup_directories(self):
        for dir_name in self.dirs.values():
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")

    def run_benchmark(self):
        """Run comprehensive benchmark tests"""
        results = []

        # Test sentences incorporating technical terms
        test_sentences = [
            f"The {term} {self.technical_terms[term]['text']} system is running efficiently"
            for term in self.technical_terms.keys()
        ]

        print("\nRunning benchmark tests...")
        for model_name, model in self.models.items():
            print(f"\nTesting model: {model_name}")

            for sentence in test_sentences:
                # Measure inference time
                start_time = time.time()
                output_path = os.path.join(self.dirs['audio'], f'{model_name}_{len(results)}.wav')

                try:
                    # Generate audio
                    model.tts_to_file(text=sentence, file_path=output_path)
                    inference_time = time.time() - start_time

                    # Analyze audio quality
                    audio_metrics = self.analyze_audio(output_path)

                    # Calculate mock MOS score (in real-world, this would come from human evaluators)
                    mock_mos = np.random.uniform(3.5, 4.8)

                    results.append({
                        'model': model_name,
                        'sentence': sentence,
                        'inference_time': inference_time,
                        'mos_score': mock_mos,
                        **audio_metrics
                    })

                    print(f"✓ Processed: {sentence[:50]}...")

                except Exception as e:
                    print(f"✗ Error processing sentence: {str(e)}")

        return results

    def analyze_audio(self, audio_path):
        """Analyze audio file and extract quality metrics"""
        try:
            audio, sr = librosa.load(audio_path)
            return {
                'duration': float(librosa.get_duration(y=audio, sr=sr)),
                'rms_energy': float(librosa.feature.rms(y=audio)[0].mean()),
                'zero_crossings_rate': float(librosa.feature.zero_crossing_rate(audio)[0].mean())
            }
        except Exception as e:
            print(f"Error analyzing {audio_path}: {str(e)}")
            return {}

    def generate_report(self, results):
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.dirs['reports'], f'benchmark_report_{timestamp}.html')

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Calculate aggregate statistics
        model_stats = df.groupby('model').agg({
            'inference_time': ['mean', 'std'],
            'mos_score': ['mean', 'std'],
            'duration': 'mean',
            'rms_energy': 'mean'
        }).round(4)

        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>TTS Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .section {{ margin: 20px 0; }}
                .technical-term {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>TTS Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="section">
                <h2>Technical Terms Dictionary</h2>
                {self._generate_terms_html()}
            </div>

            <div class="section">
                <h2>Model Performance Comparison</h2>
                {model_stats.to_html()}
            </div>

            <div class="section">
                <h2>Detailed Results</h2>
                {df.to_html(float_format=lambda x: '{:.4f}'.format(x) if isinstance(x, float) else x)}
            </div>

            <div class="section">
                <h2>Summary</h2>
                <p>Best performing model (by MOS score): {df.groupby('model')['mos_score'].mean().idxmax()}</p>
                <p>Fastest model (average inference time): {df.groupby('model')['inference_time'].mean().idxmin()}</p>
            </div>
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html_content)

        return report_path

    def _generate_terms_html(self):
        """Generate HTML for technical terms section"""
        terms_html = "<div class='technical-terms'>"
        for term, info in self.technical_terms.items():
            terms_html += f"""
                <div class='technical-term'>
                    <strong>{term}</strong><br>
                    Text representation: {info['text']}<br>
                    Pronunciation guide: {info['pronunciation']}
                </div>
            """
        terms_html += "</div>"
        return terms_html


def main():
    print("Starting TTS benchmark process...")

    benchmark = TTSBenchmark()

    # Run benchmarks
    print("\nRunning benchmarks...")
    results = benchmark.run_benchmark()

    # Generate report
    print("\nGenerating benchmark report...")
    report_path = benchmark.generate_report(results)

    print(f"\nBenchmark complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()