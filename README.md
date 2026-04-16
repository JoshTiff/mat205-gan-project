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

## Run

Choose a config file from the `configs/` folder, then use it with the scripts below.

### Train a model

```bash
python src/train.py --config configs/<config_name>.yaml
```

Example:

```bash
python src/train.py --config configs/vanilla_gan_64.yaml
```

### Generate images from a trained checkpoint

```bash
python src/generate.py --config configs/<config_name>.yaml --checkpoint <checkpoint_path>
```

Example:

```bash
python src/generate.py --config configs/vanilla_gan_64.yaml --checkpoint outputs/vanilla_gan_64/checkpoints/latest.pt
```

### Visualize training results

```bash
python src/visualize.py --checkpoint <checkpoint_path> --sample-dir <sample_dir>
```

Example:

```bash
python src/visualize.py --checkpoint outputs/vanilla_gan_64/checkpoints/latest.pt --sample-dir outputs/vanilla_gan_64/samples
```