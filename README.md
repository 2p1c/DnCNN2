# Ultrasonic Signal Denoising CAE

A lightweight 1D Convolutional Autoencoder (CAE) for ultrasonic signal denoising in Non-Destructive Testing (NDT) applications.

Based on the paper: *"Ultrasonic signal noise reduction based on convolutional autoencoders for NDT applications"*

## Features

- **Synthetic Data Generation**: Physics-based Gabor pulse signals with configurable SNR (10/20/30 dB)
- **Lightweight Model**: 3-layer encoder/decoder optimized for consumer hardware
- **Visualization**: Pre/post training signal comparisons and training curves
- **Modular Design**: Easy to extend for real data loading

## Signal Parameters

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 6.25 MHz |
| Duration | 160 μs |
| Data Points | 1000 |
| Center Frequency | 250 kHz |

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Run training (50 epochs)
uv run python train.py
```

## Output Files

After training, you'll find:
- `fig_pre_train_samples.png` - Noisy vs Clean samples before training
- `fig_results.png` - Denoising results (Noisy → Clean → Denoised)
- `fig_training_curves.png` - Loss and PSNR curves
- `checkpoints/best_model.pth` - Best model weights

## Project Structure

```
DnCNN2/
├── pyproject.toml      # Project configuration
├── data_utils.py       # Dataset and data generation
├── model.py            # CAE model architecture
├── train.py            # Training pipeline
└── README.md           # This file
```

## Model Architecture

**LightweightCAE**:
- Encoder: Conv1d (1→16→32→64), stride=2, ReLU, BatchNorm, Dropout
- Decoder: ConvTranspose1d (64→32→16→1), stride=2, ReLU, Tanh output
- Input/Output: (Batch, 1, 1000)
- Parameters: ~30K

## License

MIT
