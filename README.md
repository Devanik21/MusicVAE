# Musicvae

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/MusicVAE?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/MusicVAE?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Variational Autoencoder for music generation — learning a continuous latent space of musical sequences for interpolation, sampling, and controlled generation.

---

**Topics:** `audio-ml` · `deep-learning` · `generative-ai` · `hierarchical-lstm` · `latent-space-interpolation` · `magenta` · `music-ai` · `music-generation` · `variational-autoencoder` · `sequential-generation`

## Overview

MusicVAE is an implementation of Variational Autoencoders applied to symbolic music sequences —
a research project exploring the intersection of deep generative modelling and computational creativity.
Inspired by Magenta's MusicVAE paper (Roberts et al., 2018), it trains a hierarchical VAE on MIDI
sequences to learn a smooth, continuous latent space where neighbouring points correspond to
musically similar sequences and linear interpolation produces coherent musical transitions.

The architecture uses a bidirectional LSTM encoder to compress 2- or 4-bar MIDI sequences into a
latent vector of 256 or 512 dimensions, and a hierarchical conductor + decoder LSTM to reconstruct
the sequence from the latent code. The hierarchical decoder structure — where a conductor LSTM
generates bar-level embeddings that a lower-level decoder uses token by token — is critical for
learning coherent long-range musical structure rather than note-by-note prediction.

The latent space has three musically meaningful operations that make MusicVAE practically useful
beyond academic interest: sampling (draw a random point and decode a novel sequence), interpolation
(smoothly blend between two musical ideas by interpolating their latent vectors and decoding at
intermediate points), and attribute manipulation (identify latent directions corresponding to tempo,
density, or rhythmic complexity and move along them for controlled generation).

---

## Motivation

Standard sequence models (vanilla RNNs, Transformers) generate music autoregressively — each note
depends on previous notes but there is no global latent variable representing the 'character' of the
sequence. VAEs provide that global latent variable, enabling the musically powerful operations of
interpolation and attribute manipulation that are impossible with autoregressive models alone.
This project was built to explore whether a relatively small VAE (trained on a single GPU) can
learn a musically meaningful latent space from a few thousand MIDI sequences.

---

## Architecture

```
Input: 2-bar MIDI sequence (piano roll, 32 × 128 binary matrix)
        │
  Bidirectional LSTM Encoder
  → μ, log_σ² (256-dim each)
  → z ~ N(μ, σ²) via reparameterisation trick
        │
  Hierarchical Decoder:
  ├── Conductor LSTM: z → bar embeddings (e₁, e₂)
  └── Token Decoder LSTM: eₜ → note sequence per bar
        │
  Loss: Reconstruction (CE) + KL Divergence
  L = E[log p(x|z)] - β · KL(q(z|x) || p(z))
        │
  Latent Space Operations:
  ├── Sample: z ~ N(0,I) → decode new melody
  ├── Interpolate: z = αz₁ + (1-α)z₂ → blend melodies
  └── Attribute edit: z' = z + λ·direction → modify style
```

---

## Features

### Hierarchical VAE Architecture
Conductor + decoder LSTM hierarchy for learning coherent bar-level structure — addressing the posterior collapse problem that affects standard VAEs on long sequences.

### Piano Roll Representation
128-note × time-step binary piano roll encoding with configurable quantisation (16th note resolution default), supporting monophonic and polyphonic MIDI sequences.

### β-VAE Training
Configurable β parameter for the KL divergence weight, enabling the disentanglement vs reconstruction trade-off to be tuned for the target application (generation vs reconstruction quality).

### Latent Space Interpolation
Smooth musical interpolation between two MIDI sequences by encoding both to latent vectors and decoding at N evenly-spaced intermediate points — producing a musical 'morph'.

### Unconditioned Sampling
Sample random points from the prior N(0,I), decode to MIDI, and play back — demonstrating the diversity and quality of the learned generative model.

### Attribute Direction Discovery
Identify latent space directions correlated with musical attributes (note density, pitch range, rhythmic complexity) using labelled sequence pairs and linear regression in latent space.

### MIDI Playback Interface
In-app MIDI playback via FluidSynth or browser-based MIDI.js, allowing immediate audition of generated sequences without external DAW.

### Training Monitoring Dashboard
Real-time training curves for reconstruction loss, KL divergence, ELBO, and per-timestep accuracy — with sample generation at regular intervals to monitor learning progress.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **PyTorch** | Deep learning framework | BiLSTM encoder, conductor LSTM, decoder LSTM, Adam optimiser |
| **pretty_midi** | MIDI I/O | MIDI loading, piano roll conversion, MIDI synthesis |
| **Magenta (optional)** | Reference implementation | Pre-trained MusicVAE weights for comparison |
| **NumPy** | Array operations | Piano roll manipulation, latent arithmetic |
| **Matplotlib** | Visualisation | Piano roll plots, loss curves, latent space t-SNE |
| **FluidSynth** | MIDI playback | Offline MIDI to audio synthesis for playback |
| **Streamlit** | Generation interface | Latent space explorer, interpolation slider, sample player |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/MusicVAE.git
cd MusicVAE
python -m venv venv && source venv/bin/activate
pip install torch pretty_midi numpy matplotlib streamlit
# Optional: FluidSynth for MIDI playback
# sudo apt-get install fluidsynth  (Linux)
# brew install fluid-synth          (macOS)

# Download/prepare MIDI dataset (Lakh MIDI Dataset recommended)
# python prepare_data.py --data_dir ./midi_data/

# Train the model
python train.py --config configs/2bar_melody.yaml --epochs 100

# Launch generation interface
streamlit run app.py
```

---

## Usage

```bash
# Train MusicVAE
python train.py --epochs 100 --latent_dim 256 --beta 0.5

# Sample new melodies
python sample.py --checkpoint checkpoints/best.pt --n_samples 10

# Interpolate between two MIDI files
python interpolate.py \
  --start melody_a.mid --end melody_b.mid \
  --steps 8 --output interpolations/

# Explore latent space
streamlit run app.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LATENT_DIM` | `256` | VAE latent space dimensionality |
| `BETA` | `0.5` | KL divergence weight β in β-VAE loss |
| `N_BARS` | `2` | Sequence length in bars (2 or 4) |
| `QUANTISATION` | `16` | Time steps per bar (16 = 16th note resolution) |
| `LEARNING_RATE` | `0.001` | Adam optimiser learning rate |
| `BATCH_SIZE` | `64` | Training batch size |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
MusicVAE/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Conditional MusicVAE: condition generation on chord progression or style label
- [ ] Transformer-based encoder for capturing long-range harmonic dependencies
- [ ] Cross-modal latent space: align music latent space with text descriptions for text-to-melody generation
- [ ] Real-time latent space exploration in a browser with WebMIDI API
- [ ] Few-shot style adaptation: fine-tune the decoder on 10–20 examples of a new musical style

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

MusicVAE training on a single GPU (T4/RTX 3060) takes approximately 4–8 hours for 100 epochs on the Lakh MIDI Dataset filtered to monophonic melodies. The model exhibits posterior collapse at high β values — monitor KL divergence during training and reduce β if KL approaches zero. Generated MIDI quality varies significantly with dataset quality and training duration.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
