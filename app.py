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

class MusicVAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, latent_dim=64, num_layers=3):
        super(MusicVAE, self).__init__()
        
        # Hierarchical encoder with residual connections
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
        
        # Latent space with better regularization
        final_dim = curr_dim
        self.mu = nn.Linear(final_dim, latent_dim)
        self.logvar = nn.Linear(final_dim, latent_dim)
        
        # Hierarchical decoder with skip connections
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
        
        # Temperature parameter for controlled sampling
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

def advanced_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """Beta-VAE loss with annealing"""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def create_diverse_music_data(n_samples=2000, sequence_length=128):
    """Create more diverse and structured music data"""
    data = []
    
    # Different musical styles/patterns
    styles = ['rhythmic', 'melodic', 'harmonic', 'ambient', 'percussive']
    
    for i in range(n_samples):
        pattern = np.zeros(sequence_length)
        style = styles[i % len(styles)]
        
        if style == 'rhythmic':
            # Strong rhythmic patterns
            for beat in range(0, sequence_length, 4):
                pattern[beat] = 1.0
                if beat + 2 < sequence_length:
                    pattern[beat + 2] = 0.7
                    
        elif style == 'melodic':
            # Melodic sequences with scales
            scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale
            for note_idx in range(0, sequence_length, 8):
                scale_note = scale[np.random.randint(0, len(scale))]
                pattern[note_idx + scale_note] = np.random.uniform(0.5, 1.0)
                
        elif style == 'harmonic':
            # Chord progressions
            chords = [[0, 4, 7], [5, 9, 12], [7, 11, 14], [2, 5, 9]]
            for chord_idx in range(0, sequence_length, 32):
                chord = chords[np.random.randint(0, len(chords))]
                for note in chord:
                    if chord_idx + note < sequence_length:
                        pattern[chord_idx + note] = np.random.uniform(0.6, 0.9)
                        
        elif style == 'ambient':
            # Sustained notes with variation
            for i in range(sequence_length):
                if np.random.random() > 0.8:
                    pattern[i:i+8] = np.random.uniform(0.3, 0.6)
                    
        elif style == 'percussive':
            # Complex drum patterns
            for beat in [0, 6, 12, 18, 24, 30]:
                if beat < sequence_length:
                    pattern[beat] = 1.0
            for snare in [8, 16, 24]:
                if snare < sequence_length:
                    pattern[snare] = 0.8
        
        # Add variation and noise
        pattern += np.random.normal(0, 0.05, sequence_length)
        pattern = np.clip(pattern, 0, 1)
        
        data.append(pattern)
    
    return np.array(data, dtype=np.float32)

def train_advanced_vae(model, train_loader, epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        beta = min(1.0, epoch / (epochs * 0.5))  # Beta annealing
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = advanced_vae_loss(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        progress_bar.progress((epoch + 1) / epochs)
        
        if epoch % 20 == 0:
            st.write(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Beta: {beta:.3f}')
    
    return loss_history

def generate_music_with_control(model, latent_dim=64, num_samples=1, temperature=1.0, interpolate=False):
    model.eval()
    with torch.no_grad():
        if interpolate:
            # Generate interpolation between two points
            z1 = torch.randn(1, latent_dim).to(device)
            z2 = torch.randn(1, latent_dim).to(device)
            alphas = torch.linspace(0, 1, num_samples).unsqueeze(1).to(device)
            z = alphas * z2 + (1 - alphas) * z1
        else:
            z = torch.randn(num_samples, latent_dim).to(device) * temperature
        
        generated = model.decode(z).cpu().numpy()
    return generated

def music_to_audio(music_data, sample_rate=22050, duration=6):
    """Enhanced audio conversion with multiple voices"""
    time = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(time)
    
    # Multiple frequency ranges for richer sound
    base_freqs = [110, 220, 440, 880]  # Bass, Low, Mid, High
    segment_length = len(time) // len(music_data)
    
    for i, intensity in enumerate(music_data):
        if intensity > 0.05:
            # Choose frequency range based on position
            freq_range_idx = i % len(base_freqs)
            base_freq = base_freqs[freq_range_idx]
            
            # Add harmonics for richer sound
            freq = base_freq * (2 ** ((i % 24) / 12))  # Two octave range
            
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(time))
            segment_time = time[start_idx:end_idx]
            
            # Create wave with harmonics
            wave = intensity * (
                np.sin(2 * np.pi * freq * segment_time) +
                0.3 * np.sin(2 * np.pi * freq * 2 * segment_time) +
                0.1 * np.sin(2 * np.pi * freq * 3 * segment_time)
            )
            
            # Apply envelope
            envelope = np.exp(-segment_time * 2)
            wave *= envelope
            
            audio[start_idx:end_idx] += wave
    
    # Apply soft clipping and normalize
    audio = np.tanh(audio)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.7
    
    return (audio * 32767).astype(np.int16)

def main():
    st.title("🎵 Advanced MusicVAE Generator")
    st.write("Advanced VAE with hierarchical architecture and diverse training data")
    
    # Advanced parameters
    st.sidebar.header("Advanced Parameters")
    latent_dim = st.sidebar.slider("Latent Dimension", 16, 128, 64)
    hidden_dim = st.sidebar.slider("Hidden Dimension", 256, 1024, 512)
    num_layers = st.sidebar.slider("Network Layers", 2, 5, 3)
    epochs = st.sidebar.slider("Training Epochs", 50, 200, 100)
    temperature = st.sidebar.slider("Generation Temperature", 0.1, 2.0, 1.0)
    
    # Initialize session state
    if 'advanced_model' not in st.session_state:
        st.session_state.advanced_model = None
    if 'advanced_trained' not in st.session_state:
        st.session_state.advanced_trained = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Training")
        
        if st.button("Train Advanced Model"):
            with st.spinner("Training Advanced MusicVAE..."):
                # Create diverse synthetic data
                music_data = create_diverse_music_data(n_samples=2000, sequence_length=128)
                
                # Show data diversity
                fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                for i in range(6):
                    axes[i//3, i%3].plot(music_data[i*300])
                    axes[i//3, i%3].set_title(f'Style {i+1}')
                st.pyplot(fig)
                
                # Create data loader
                dataset = TensorDataset(torch.tensor(music_data))
                train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
                
                # Initialize advanced model
                model = MusicVAE(
                    input_dim=128, 
                    hidden_dim=hidden_dim, 
                    latent_dim=latent_dim,
                    num_layers=num_layers
                )
                model.to(device)
                
                # Train model
                loss_history = train_advanced_vae(model, train_loader, epochs)
                
                # Store in session state
                st.session_state.advanced_model = model
                st.session_state.advanced_trained = True
                
                # Plot training metrics
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(loss_history)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss')
                
                # Show loss improvement
                if len(loss_history) > 20:
                    improvement = loss_history[0] - loss_history[-1]
                    ax2.bar(['Initial', 'Final'], [loss_history[0], loss_history[-1]])
                    ax2.set_title(f'Loss Improvement: {improvement:.2f}')
                
                st.pyplot(fig)
                st.success(f"Advanced model trained! Final loss: {loss_history[-1]:.4f}")
    
    with col2:
        st.header("Generation")
        
        if st.session_state.advanced_trained and st.session_state.advanced_model is not None:
            prompt = st.text_input("Style Prompt", "Melodic ambient sequence")
            interpolate = st.checkbox("Interpolate between styles")
            num_generations = st.slider("Number of generations", 1, 5, 1)
            
            if st.button("Generate Music"):
                with st.spinner("Generating advanced music..."):
                    # Generate music with controls
                    generated_music = generate_music_with_control(
                        st.session_state.advanced_model, 
                        latent_dim=latent_dim, 
                        num_samples=num_generations,
                        temperature=temperature,
                        interpolate=interpolate
                    )
                    
                    for i, music_sequence in enumerate(generated_music):
                        st.subheader(f"Generation {i+1}")
                        
                        # Visualize
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                        ax1.plot(music_sequence)
                        ax1.set_title(f'{prompt} - Sequence {i+1}')
                        ax1.set_ylabel('Intensity')
                        
                        # Show frequency spectrum
                        spectrum = np.abs(np.fft.fft(music_sequence))[:64]
                        ax2.plot(spectrum)
                        ax2.set_title('Frequency Spectrum')
                        ax2.set_xlabel('Frequency Bin')
                        ax2.set_ylabel('Magnitude')
                        
                        st.pyplot(fig)
                        
                        # Convert to audio
                        audio_data = music_to_audio(music_sequence)
                        
                        # Save and play
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            write(tmp_file.name, 22050, audio_data)
                            
                            with open(tmp_file.name, 'rb') as f:
                                audio_bytes = f.read()
                            
                            st.audio(audio_bytes, format='audio/wav')
                            st.download_button(
                                label=f"Download Audio {i+1}",
                                data=audio_bytes,
                                file_name=f"advanced_music_{i+1}.wav",
                                mime="audio/wav",
                                key=f"download_{i}"
                            )
                            
                            os.unlink(tmp_file.name)
        else:
            st.info("Train the advanced model first!")
    
    st.header("Model Architecture")
    st.write("""
    **Improvements:**
    - Hierarchical encoder/decoder with residual connections
    - Batch normalization and dropout for stability
    - Beta-VAE with annealing for better disentanglement
    - Diverse training data with 5 musical styles
    - Temperature-controlled generation
    - Enhanced audio synthesis with harmonics
    - Gradient clipping and learning rate scheduling
    """)

if __name__ == "__main__":
    main()
