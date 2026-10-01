# 🎨 StyleForge AI: Real-Time Neural Style Transfer with AdaIN

An end-to-end implementation of Arbitrary Style Transfer using **Adaptive Instance Normalization (AdaIN)** powered by **PyTorch** and deployed with an interactive, modern **Flask** web application.

---

## 🌟 Key Features

- **Arbitrary Style Transfer**: Transfer any painting or artistic style onto any photo in real time without retraining for each new style.
- **Adaptive Instance Normalization (AdaIN)**: Aligns the mean and variance of content feature representations with those of the style features in latent space.
- **Adjustable Style Strength ($\alpha$)**: Smoothly interpolate between the original content and full style intensity using an interactive slider ($0.0 \le \alpha \le 1.0$).
- **Customizable Training Pipeline**: Modular training script with configurable learning rate decay, content/style loss weighting, checkpointing, and resume capabilities.
- **Modern Web Interface**: Glassmorphism UI styled with particle neural net animations, instant image previews, and direct download of stylized outputs.

---

## 🏗️ Architecture Overview

```
Content Image ───► [ VGG-19 Encoder ] ───► Content Features (fc)
                                                      │
                                                      ▼
                                           [ AdaIN Layer ] ◄─── Style Features (fs)
                                                      │                     ▲
Style Image ─────► [ VGG-19 Encoder ] ────────┘                     │
                                                      │           [ VGG-19 Encoder ]
                                                      ▼                     │
                                             [ Decoder ] ────────► Stylized Image
```

1. **Encoder**: Fixed, pre-trained normalized VGG-19 (`vgg_normalised.pth`) extracting multi-scale feature maps (`relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`).
2. **AdaIN Layer**: Computes $\text{AdaIN}(x, y) = \sigma(y) \left(\frac{x - \mu(x)}{\sigma(x)}\right) + \mu(y)$.
3. **Decoder**: Inverted convolutional network trained to reconstruct latent representations back into RGB images.

---

## 📁 Repository Structure

```text
NST_PROJECT/
├── app.py                  # Flask web application
├── train.py                # Model training script
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
├── vgg_normalised.pth      # Pretrained normalized VGG-19 weights
├── content_data/           # Content dataset folder (e.g., COCO)
├── style_data/             # Style dataset folder (e.g., WikiArt)
├── experiment/             # Training runs, logs, and checkpoints
│   ├── trial/              # Pretrained / saved checkpoints
│   └── trial2/             # Active experiment checkpoints
├── templates/
│   └── index.html          # Web UI template
└── utils/
    ├── models.py           # VGGEncoder & Decoder architectures
    └── utils.py            # AdaIN function, dataset loaders, and transforms
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ installed. Clone or navigate to the repository directory:

```bash
git clone https://github.com/Ronakjain935/Neural_Style_Transfer.git
cd Neural_Style_Transfer
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Pretrained VGG Weights

Place the normalized VGG-19 model file (`vgg_normalised.pth`) in the project root.

---

## 🏋️ Training the Model

To train the decoder network from scratch or fine-tune an existing model:

### Basic Training

```powershell
python train.py --batch_size 4 --epochs 10 --experiment='trial'
```

### Advanced Training with Custom Datasets & Hyperparameters

```powershell
python train.py `
  --batch_size 4 `
  --epochs 10 `
  --lr 1e-4 `
  --content_weight 1.0 `
  --style_weight 5.0 `
  --content_dir 'C:\path\to\content_images' `
  --style_dir 'C:\path\to\style_images' `
  --experiment 'custom_run'
```

### Resume Training from a Checkpoint

```powershell
python train.py `
  --resume `
  --decoder_path 'experiment/trial/decoder_5.pth' `
  --optimizer_path 'experiment/trial/optimizer_5.pth' `
  --experiment 'custom_run'
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--content_dir` | `str` | `content_data` | Directory containing content images |
| `--style_dir` | `str` | `style_data` | Directory containing style images |
| `--vgg` | `str` | `vgg_normalised.pth` | Path to normalized VGG-19 weights |
| `--experiment` | `str` | `experiment1` | Name of experiment output subfolder |
| `--batch_size` | `int` | `4` | Training batch size |
| `--epochs` | `int` | `1` | Number of training epochs |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--lr_decay` | `float` | `5e-5` | Learning rate decay rate |
| `--content_weight`| `float` | `1.0` | Weight for content reconstruction loss |
| `--style_weight` | `float` | `5.0` | Weight for style reconstruction loss |
| `--save_interval`| `int` | `2` | Epoch interval for saving model checkpoints |

---

## 🌐 Running the Web Application

Launch the interactive Flask UI:

```powershell
python app.py
```

Once started, open your browser and navigate to:
```
http://localhost:5000
```

### Using the App:
1. **Upload Content Image**: Choose the photo you want to stylize.
2. **Upload Style Image**: Choose the painting, sketch, or artwork reference.
3. **Adjust Style Strength**: Drag the alpha slider from `0.0` (original content) to `1.0` (maximum style).
4. **Click "Transfer Style"**: The neural network will process the images and display the stylized result with a one-click download option.

---

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: Pillow (PIL), NumPy
- **Backend**: Flask, Werkzeug, Flask-WTF, WTForms
- **Frontend**: HTML5, Vanilla CSS3, Bootstrap 5, FontAwesome, JavaScript (Canvas animations)

---

## 📜 References

- Xun Huang and Serge Belongie. *"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization."* ICCV 2017. [arXiv:1705.06830](https://arxiv.org/abs/1705.06830)
- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. *"Image Style Transfer Using Convolutional Neural Networks."* CVPR 2016.
