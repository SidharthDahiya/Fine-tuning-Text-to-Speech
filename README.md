# Fine-tuning Text to Speech  

This project implements fine-tuned Text-to-Speech (TTS) models for two specific use cases: technical term pronunciation in English and Hindi language speech synthesis.

### 🎯 Objectives

1. **English TTS Fine-tuning**
   - Optimize pronunciation of technical terms (API, CUDA, TTS, etc.)
   - Ensure accurate and natural-sounding speech
   - Focus on interview-context technical vocabulary

2. **Hindi TTS Implementation**
   - Develop natural Hindi speech synthesis
   - Maintain consistent voice characteristics
   - Support various Hindi language contexts

### 🔍 Implementation Approach
For (English TTS), I focused on making the system understand and correctly pronounce technical terms like "API" and "CUDA". Think of it like teaching someone new vocabulary - I first showed the model how these terms should sound, then helped it practice until it got them right. The process was similar to how a language tutor would teach specialized terms to a student, ensuring both correct pronunciation and understanding of context.

For (Hindi TTS), I took a slightly different approach, similar to how a bilingual speaker adapts their voice. I first established a base voice in English, then used that same voice characteristic to speak Hindi, ensuring consistency across languages. This is much like how a person maintains their unique voice while switching between languages, just with proper pronunciation rules for each language. The whole process was designed to be efficient and storage-friendly, much like learning the essential phrases first before diving into complex sentences.

### 🛠️ Technical Implementation

- **Key technologies used:**
  - Python 3.9+
  - Coqui TTS
  - PyTorch
  - Librosa
  - Pandas

### 📦 Installation

```bash
# Clone the repository
git clone [https://github.com/SidharthDahiya/Fine-tuning-Text-to-Speech]
cd Fine-tuning-Text-to-Speech

# Install required packages
pip install TTS torch pandas numpy librosa

```

### 📁 Project Structure
```
project/
│
├── english_tts_finetuner.py    # English TTS implementation
├── hindi_tts_trainer.py        # Hindi TTS implementation
│
├── output/                     # English TTS outputs
│   ├── audio_samples/
│   └── evaluation_reports/
│
└── hindi_output/              # Hindi TTS outputs
    ├── audio_samples/
    └── evaluation_reports/
```

### 📊 Evaluation Metrics
- Mean Opinion Score (MOS)
- Inference Speed
- Pronunciation Accuracy
- Audio Quality Metrics

### ⚠️ Requirements
- Python 3.9 or higher
- Minimum 10GB disk space
- RAM: 8GB minimum (16GB recommended)
- CUDA-capable GPU (optional, for faster processing)

### 📝 Note
- The models require an initial download (~2GB)
- Generated audio files and checkpoints can take significant storage space (10-15GB)
- The fine-tuning process might take several minutes depending on the hardware

