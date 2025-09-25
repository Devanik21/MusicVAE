import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tempfile
import os
from scipy.io.wavfile import write

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MODEL 1: MusicVAE =====
class MusicVAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, latent_dim=64, num_layers=3):
        super(MusicVAE, self).__init__()
        
        encoder_layers = []
        curr_dim = input_dim
        
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            curr_dim = hidden_dim
            hidden_dim = hidden_dim // 2
        
        self.encoder = nn.Sequential(*encoder_layers)
        final_dim = curr_dim
        self.mu = nn.Linear(final_dim, latent_dim)
        self.logvar = nn.Linear(final_dim, latent_dim)
        
        decoder_layers = []
        curr_dim = latent_dim
        hidden_dim = 128
        
        for i in range(num_layers):
            decoder_layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            curr_dim = hidden_dim
            hidden_dim *= 2
        
        decoder_layers.extend([
            nn.Linear(curr_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, self.temperature)
        return self.decode(z), mu, logvar

class MusicTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=256, nhead=8, num_layers=6, latent_dim=64):
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.input_proj = nn.Linear(1, d_model)  # Project each timestep to d_model
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.latent_proj = nn.Linear(d_model, latent_dim * 2)  # mu and logvar
        self.decoder_proj = nn.Linear(latent_dim, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_proj = nn.Linear(d_model, 1)
        
    def encode(self, x):
        batch_size, seq_len = x.shape
        # Reshape to [batch, seq, 1] then project to [batch, seq, d_model]
        x = x.unsqueeze(-1)  # [batch, seq, 1]
        x = self.input_proj(x)  # [batch, seq, d_model]
        x = x + self.pos_encoding[:, :seq_len, :]  # Add positional encoding
        x = x.transpose(0, 1)  # [seq, batch, d_model] for transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling -> [batch, d_model]
        latent_params = self.latent_proj(x)  # [batch, latent_dim * 2]
        mu, logvar = latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]
        return mu, logvar
        
    def decode(self, z):
        batch_size = z.shape[0]
        # Project latent to d_model and create sequence
        z_proj = self.decoder_proj(z)  # [batch, d_model]
        z_seq = z_proj.unsqueeze(1).repeat(1, self.input_dim, 1)  # [batch, seq, d_model]
        
        # Prepare for transformer decoder
        memory = z_seq.transpose(0, 1)  # [seq, batch, d_model]
        tgt = torch.zeros_like(memory)  # [seq, batch, d_model]
        
        # Decode
        x = self.transformer_decoder(tgt, memory)  # [seq, batch, d_model]
        x = x.transpose(0, 1)  # [batch, seq, d_model]
        x = self.output_proj(x)  # [batch, seq, 1]
        x = x.squeeze(-1)  # [batch, seq]
        
        return torch.sigmoid(x)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

# ===== MODEL 3: Music GAN =====
class MusicGenerator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=128):
        super(MusicGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.net(z)

class MusicDiscriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(MusicDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class MusicGAN(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=128):
        super(MusicGAN, self).__init__()
        self.generator = MusicGenerator(latent_dim, hidden_dim, output_dim)
        self.discriminator = MusicDiscriminator(output_dim, hidden_dim)
        self.latent_dim = latent_dim
        
    def generate(self, batch_size=1):
        z = torch.randn(batch_size, self.latent_dim).to(device)
        return self.generator(z)

# ===== MODEL 4: Music RNN =====
class MusicRNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=3, latent_dim=64):
        super(MusicRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        self.encoder_rnn = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.decoder_rnn = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def encode(self, x):
        x = x.unsqueeze(-1)  # Add feature dimension
        _, (hidden, _) = self.encoder_rnn(x)
        hidden = hidden[-1]  # Take last layer
        mu = self.mu_proj(hidden)
        logvar = self.logvar_proj(hidden)
        return mu, logvar
        
    def decode(self, z):
        z = z.unsqueeze(1).repeat(1, 128, 1)  # Repeat for sequence length
        output, _ = self.decoder_rnn(z)
        return torch.sigmoid(self.output_proj(output).squeeze(-1))
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ===== MODEL 5: Music Diffusion =====
class MusicDiffusion(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, timesteps=1000):
        super(MusicDiffusion, self).__init__()
        self.timesteps = timesteps
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Beta schedule for noise
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def forward_diffusion(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        return (sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise), noise
    
    def forward(self, x):
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,)).to(device)
        
        x_noisy, noise = self.forward_diffusion(x, t)
        
        # Predict noise
        t_normalized = t.float() / self.timesteps
        x_with_time = torch.cat([x_noisy, t_normalized.unsqueeze(-1)], dim=-1)
        predicted_noise = self.noise_predictor(x_with_time)
        
        return predicted_noise, noise
    
    def sample(self, batch_size=1):
        x = torch.randn(batch_size, 128).to(device)
        
        for t in reversed(range(self.timesteps)):
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            t_tensor = torch.full((batch_size,), t).to(device)
            t_normalized = t_tensor.float() / self.timesteps
            x_with_time = torch.cat([x, t_normalized.unsqueeze(-1)], dim=-1)
            
            predicted_noise = self.noise_predictor(x_with_time)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t-1]
            else:
                alpha_cumprod_prev = torch.tensor(1.0)
                
            x = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/torch.sqrt(1-alpha_cumprod)) * predicted_noise)
            if t > 0:
                x += torch.sqrt(beta) * noise
                
        return torch.sigmoid(x)

def create_diverse_music_data(n_samples=2000, sequence_length=128):
    """Create diverse music data with multiple styles"""
    data = []
    styles = ['rhythmic', 'melodic', 'harmonic', 'ambient', 'percussive']
    
    for i in range(n_samples):
        pattern = np.zeros(sequence_length)
        style = styles[i % len(styles)]
        
        if style == 'rhythmic':
            for beat in range(0, sequence_length, 4):
                pattern[beat] = 1.0
                if beat + 2 < sequence_length:
                    pattern[beat + 2] = 0.7
                    
        elif style == 'melodic':
            scale = [0, 2, 4, 5, 7, 9, 11]
            for note_idx in range(0, sequence_length, 8):
                scale_note = scale[np.random.randint(0, len(scale))]
                if note_idx + scale_note < sequence_length:
                    pattern[note_idx + scale_note] = np.random.uniform(0.5, 1.0)
                    
        elif style == 'harmonic':
            chords = [[0, 4, 7], [5, 9, 12], [7, 11, 14], [2, 5, 9]]
            for chord_idx in range(0, sequence_length, 32):
                chord = chords[np.random.randint(0, len(chords))]
                for note in chord:
                    if chord_idx + note < sequence_length:
                        pattern[chord_idx + note] = np.random.uniform(0.6, 0.9)
                        
        elif style == 'ambient':
            for i in range(sequence_length):
                if np.random.random() > 0.8:
                    end_idx = min(i + 8, sequence_length)
                    pattern[i:end_idx] = np.random.uniform(0.3, 0.6)
                    
        elif style == 'percussive':
            for beat in [0, 6, 12, 18, 24, 30]:
                if beat < sequence_length:
                    pattern[beat] = 1.0
            for snare in [8, 16, 24]:
                if snare < sequence_length:
                    pattern[snare] = 0.8
        
        pattern += np.random.normal(0, 0.05, sequence_length)
        pattern = np.clip(pattern, 0, 1)
        data.append(pattern)
    
    return np.array(data, dtype=np.float32)

def train_model(model, model_type, train_loader, epochs=100):
    """Universal training function for all models"""
    if model_type == 'GAN':
        return train_gan(model, train_loader, epochs)
    elif model_type == 'Diffusion':
        return train_diffusion(model, train_loader, epochs)
    else:
        return train_vae_style(model, train_loader, epochs)

def train_vae_style(model, train_loader, epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        beta = min(1.0, epoch / (epochs * 0.5))
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        progress_bar.progress((epoch + 1) / epochs)
        
        if epoch % 20 == 0:
            st.write(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return loss_history

def train_gan(model, train_loader, epochs=100):
    g_optimizer = optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    progress_bar = st.progress(0)
    g_losses, d_losses = [], []
    
    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0, 0
        
        for batch_idx, (real_data,) in enumerate(train_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            real_output = model.discriminator(real_data)
            d_real_loss = F.binary_cross_entropy(real_output, real_labels)
            
            fake_data = model.generate(batch_size)
            fake_output = model.discriminator(fake_data.detach())
            d_fake_loss = F.binary_cross_entropy(fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            fake_output = model.discriminator(fake_data)
            g_loss = F.binary_cross_entropy(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
        
        g_losses.append(g_loss_total / len(train_loader))
        d_losses.append(d_loss_total / len(train_loader))
        progress_bar.progress((epoch + 1) / epochs)
        
        if epoch % 20 == 0:
            st.write(f'Epoch {epoch}, G Loss: {g_losses[-1]:.4f}, D Loss: {d_losses[-1]:.4f}')
    
    return g_losses, d_losses

def train_diffusion(model, train_loader, epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            predicted_noise, actual_noise = model(data)
            loss = F.mse_loss(predicted_noise, actual_noise)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        progress_bar.progress((epoch + 1) / epochs)
        
        if epoch % 20 == 0:
            st.write(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return loss_history

def generate_music(model, model_type, num_samples=1, **kwargs):
    """Universal generation function"""
    model.eval()
    with torch.no_grad():
        if model_type == 'GAN':
            return model.generate(num_samples).cpu().numpy()
        elif model_type == 'Diffusion':
            return model.sample(num_samples).cpu().numpy()
        else:  # VAE-style models
            latent_dim = kwargs.get('latent_dim', 64)
            temperature = kwargs.get('temperature', 1.0)
            
            z = torch.randn(num_samples, latent_dim).to(device) * temperature
            if hasattr(model, 'decode'):
                generated = model.decode(z).cpu().numpy()
            else:
                generated, _, _ = model(torch.zeros(num_samples, 128).to(device))
                generated = generated.cpu().numpy()
            return generated

def music_to_audio(music_data, sample_rate=22050, duration=6):
    """Convert music data to audio"""
    time = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(time)
    
    base_freqs = [110, 220, 440, 880]
    segment_length = len(time) // len(music_data)
    
    for i, intensity in enumerate(music_data):
        if intensity > 0.05:
            freq_range_idx = i % len(base_freqs)
            base_freq = base_freqs[freq_range_idx]
            freq = base_freq * (2 ** ((i % 24) / 12))
            
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(time))
            segment_time = time[start_idx:end_idx]
            
            wave = intensity * (
                np.sin(2 * np.pi * freq * segment_time) +
                0.3 * np.sin(2 * np.pi * freq * 2 * segment_time) +
                0.1 * np.sin(2 * np.pi * freq * 3 * segment_time)
            )
            
            envelope = np.exp(-segment_time * 2)
            wave *= envelope
            audio[start_idx:end_idx] += wave
    
    audio = np.tanh(audio)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.7
    
    return (audio * 32767).astype(np.int16)

def main():
    st.title("🎵 Multi-Model Music Generator")
    st.write("Choose from 5 different AI models to generate music")
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Model Architecture",
        ["MusicVAE", "MusicTransformer", "MusicGAN", "MusicRNN", "MusicDiffusion"]
    )
    
    st.sidebar.header(f"{model_choice} Parameters")
    
    # Model-specific parameters
    if model_choice in ["MusicVAE", "MusicTransformer", "MusicRNN"]:
        latent_dim = st.sidebar.slider("Latent Dimension", 16, 128, 64)
        temperature = st.sidebar.slider("Generation Temperature", 0.1, 2.0, 1.0)
    elif model_choice == "MusicGAN":
        latent_dim = st.sidebar.slider("Noise Dimension", 50, 200, 100)
    elif model_choice == "MusicDiffusion":
        timesteps = st.sidebar.slider("Diffusion Steps", 100, 1000, 500)
    
    hidden_dim = st.sidebar.slider("Hidden Dimension", 128, 512, 256)
    epochs = st.sidebar.slider("Training Epochs", 30, 150, 80)
    
    # Initialize session state
    model_key = f'{model_choice.lower()}_model'
    trained_key = f'{model_choice.lower()}_trained'
    
    if model_key not in st.session_state:
        st.session_state[model_key] = None
    if trained_key not in st.session_state:
        st.session_state[trained_key] = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header(f"Train {model_choice}")
        
        # Model info
        model_descriptions = {
            "MusicVAE": "Variational Autoencoder with hierarchical architecture for learning musical latent representations",
            "MusicTransformer": "Transformer-based model using self-attention for capturing long-range musical dependencies",
            "MusicGAN": "Generative Adversarial Network with generator and discriminator for realistic music synthesis",
            "MusicRNN": "Recurrent Neural Network using LSTM layers for sequential music pattern modeling",
            "MusicDiffusion": "Denoising diffusion model for high-quality music generation through iterative refinement"
        }
        
        st.info(model_descriptions[model_choice])
        
        if st.button(f"Train {model_choice}"):
            with st.spinner(f"Training {model_choice}..."):
                # Create data
                music_data = create_diverse_music_data(n_samples=1500, sequence_length=128)
                dataset = TensorDataset(torch.tensor(music_data))
                train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Initialize model
                if model_choice == "MusicVAE":
                    model = MusicVAE(128, hidden_dim, latent_dim, 3)
                elif model_choice == "MusicTransformer":
                    model = MusicTransformer(128, hidden_dim, 8, 4, latent_dim)
                elif model_choice == "MusicGAN":
                    model = MusicGAN(latent_dim, hidden_dim, 128)
                elif model_choice == "MusicRNN":
                    model = MusicRNN(128, hidden_dim, 3, latent_dim)
                elif model_choice == "MusicDiffusion":
                    model = MusicDiffusion(128, hidden_dim, timesteps if 'timesteps' in locals() else 500)
                
                model.to(device)
                
                # Train model
                model_type = model_choice.replace('Music', '')
                loss_history = train_model(model, model_type, train_loader, epochs)
                
                # Store model
                st.session_state[model_key] = model
                st.session_state[trained_key] = True
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 4))
                if model_choice == "MusicGAN":
                    ax.plot(loss_history[0], label='Generator Loss')
                    ax.plot(loss_history[1], label='Discriminator Loss')
                    ax.legend()
                else:
                    ax.plot(loss_history, label='Training Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{model_choice} Training Progress')
                st.pyplot(fig)
                
                final_loss = loss_history[0][-1] if model_choice == "MusicGAN" else loss_history[-1]
                st.success(f"{model_choice} trained! Final loss: {final_loss:.4f}")
    
    with col2:
        st.header(f"Generate with {model_choice}")
        
        if st.session_state[trained_key] and st.session_state[model_key] is not None:
            prompt = st.text_input("Style Description", f"{model_choice} generated music")
            num_generations = st.slider("Number of Generations", 1, 3, 1)
            
            if st.button("Generate Music"):
                with st.spinner("Generating music..."):
                    model = st.session_state[model_key]
                    model_type = model_choice.replace('Music', '')
                    
                    # Generate
                    kwargs = {}
                    if model_choice in ["MusicVAE", "MusicTransformer", "MusicRNN"]:
                        kwargs = {'latent_dim': latent_dim, 'temperature': temperature}
                    
                    generated_music = generate_music(model, model_type, num_generations, **kwargs)
                    
                    for i, music_sequence in enumerate(generated_music):
                        st.subheader(f"Generation {i+1}")
                        
                        # Visualize
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                        ax1.plot(music_sequence)
                        ax1.set_title(f'{prompt} - Sequence {i+1}')
                        ax1.set_ylabel('Intensity')
                        
                        spectrum = np.abs(np.fft.fft(music_sequence))[:64]
                        ax2.plot(spectrum)
                        ax2.set_title('Frequency Spectrum')
                        ax2.set_xlabel('Frequency Bin')
                        ax2.set_ylabel('Magnitude')
                        
                        st.pyplot(fig)
                        
                        # Convert to audio
                        audio_data = music_to_audio(music_sequence)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            write(tmp_file.name, 22050, audio_data)
                            
                            with open(tmp_file.name, 'rb') as f:
                                audio_bytes = f.read()
                            
                            st.audio(audio_bytes, format='audio/wav')
                            st.download_button(
                                label=f"Download {model_choice} Audio {i+1}",
                                data=audio_bytes,
                                file_name=f"{model_choice.lower()}_music_{i+1}.wav",
                                mime="audio/wav",
                                key=f"download_{model_choice}_{i}"
                            )
                            
                            os.unlink(tmp_file.name)
        else:
            st.info(f"Train the {model_choice} model first!")
    
    # Model comparison section
    st.header("🔍 Model Comparison")
    
    comparison_data = {
        "Model": ["MusicVAE", "MusicTransformer", "MusicGAN", "MusicRNN", "MusicDiffusion"],
        "Architecture": ["Variational Autoencoder", "Transformer", "Adversarial Network", "LSTM Network", "Diffusion Model"],
        "Strengths": [
            "Smooth interpolation, structured latent space",
            "Long-range dependencies, attention mechanism", 
            "Realistic generation, adversarial training",
            "Sequential modeling, temporal patterns",
            "High quality output, stable training"
        ],
        "Best For": [
            "Style transfer, latent exploration",
            "Complex compositions, harmony",
            "Realistic textures, variety",
            "Melody generation, rhythm",
            "High-fidelity synthesis"
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture Details")
        for i, model in enumerate(comparison_data["Model"]):
            with st.expander(f"{model} Details"):
                st.write(f"**Architecture:** {comparison_data['Architecture'][i]}")
                st.write(f"**Strengths:** {comparison_data['Strengths'][i]}")
                st.write(f"**Best For:** {comparison_data['Best For'][i]}")
    
    with col2:
        st.subheader("Training Tips")
        st.write("""
        **MusicVAE**: 
        - Higher latent dims for complex music
        - Lower temperature for coherent output
        
        **MusicTransformer**:
        - More attention heads for harmony
        - Deeper layers for complexity
        
        **MusicGAN**:
        - Balance generator/discriminator learning
        - Higher noise dim for variety
        
        **MusicRNN**:
        - More LSTM layers for long sequences
        - Bidirectional for better context
        
        **MusicDiffusion**:
        - More timesteps for quality
        - Longer training for best results
        """)
    
    st.header("🎼 About This App")
    st.write("""
    This multi-model music generator implements 5 different AI architectures:
    
    1. **MusicVAE**: Learns a continuous latent space for smooth music interpolation
    2. **MusicTransformer**: Uses self-attention to capture musical structure and harmony
    3. **MusicGAN**: Adversarial training for realistic and diverse music generation
    4. **MusicRNN**: Sequential modeling with LSTM for temporal music patterns
    5. **MusicDiffusion**: Iterative denoising for high-quality music synthesis
    
    Each model has unique strengths and is suitable for different musical styles and applications.
    Train multiple models to compare their outputs and find your preferred approach!
    """)

if __name__ == "__main__":
    main()
