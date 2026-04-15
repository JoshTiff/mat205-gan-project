## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv env
.\env\Scripts\activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### 3. Get data

Download the following datasets and extract them into a `data/` folder.
- [Abstract Art Gallery](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery/data)
- [Abstract Paintings](https://www.kaggle.com/datasets/danielvalyano/abstract-paintings)
