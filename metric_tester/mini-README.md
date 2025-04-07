
## 1. Overview

The system consists of three main Python files:
- `evalNNproject.py`: The main evaluation script containing the `CaptionEvaluator` class
- `pythspicesim.py`: Implementation of a SPICE-like semantic evaluation metric
- `clip.py`: Implementation of CLIP-based image-text similarity evaluation

## 2. Installation

### Prerequisites
- Python 3.6 or higher
- pip package manager

### Step 1: Clone or download the project files
Ensure you have the following files in your project directory:
- `evalNNproject.py`
- `pythspicesim.py`
- `clip.py`

### Step 2: Install required dependencies
Run the following command to install all required packages:

```bash
pip install nltk rouge-score torch transformers pillow spacy
```

### Step 3: Download required NLTK data
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

### Step 4: Download required spaCy model
```bash
python -m spacy download en_core_web_md
```

## 3. Usage

Run the metric test on exaple captions and images:

```bash
python run_evaluation.py
```





