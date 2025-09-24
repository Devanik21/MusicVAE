import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import io
import base64
from scipy.io.wavfile import write
import librosa
import tempfile
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MusicVAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=32):
        super(MusicVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Latent space
        self.mu = nn.Linear(hidden_dim//2, latent_dim)
        self.logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def create_synthetic_music_data(n_samples=1000, sequence_length=128):
    """Create synthetic music-like data for training"""
    data = []
    
    for i in range(n_samples):
        # Create patterns that mimic musical structures
        pattern = np.zeros(sequence_length)
        
        # Add some rhythmic patterns
        for beat in range(0, sequence_length, 16):
            if np.random.random() > 0.3:
                pattern[beat] = 1.0
            if np.random.random() > 0.5 and beat + 8 < sequence_length:
                pattern[beat + 8] = 0.8
                
        # Add some melodic content
        for note in range(4, sequence_length, 8):
            if np.random.random() > 0.4:
                pattern[note:note+4] = np.random.random(4) * 0.6
                
        # Add some bass notes
        for bass in range(0, sequence_length, 32):
            if np.random.random() > 0.2:
                pattern[bass] = 1.0
                
        data.append(pattern)
    
    return np.array(data, dtype=np.float32)

def train_vae(model, train_loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        progress_bar.progress((epoch + 1) / epochs)
        
        if epoch % 10 == 0:
            st.write(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return loss_history

def generate_music(model, latent_dim=32, num_samples=1):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated = model.decode(z).cpu().numpy()
    return generated

def music_to_audio(music_data, sample_rate=22050, duration=4):
    """Convert music data to audio waveform"""
    # Simple conversion: map music data to frequencies
    time = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(time)
    
    # Map each element in music_data to a frequency and add to audio
    base_freq = 220  # A3
    for i, intensity in enumerate(music_data):
        if intensity > 0.1:  # Only play notes above threshold
            freq = base_freq * (2 ** (i / 12))  # Chromatic scale
            wave = intensity * np.sin(2 * np.pi * freq * time[:len(time)//len(music_data)])
            audio[i*len(time)//len(music_data):(i+1)*len(time)//len(music_data)] += wave[:len(time)//len(music_data)]
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.5
    
    return (audio * 32767).astype(np.int16)

def main():
    st.title("🎵 MusicVAE Generator")
    st.write("Train a simple VAE on synthetic music data and generate new musical sequences")
    
    # Sidebar for parameters
    st.sidebar.header("Model Parameters")
    latent_dim = st.sidebar.slider("Latent Dimension", 8, 64, 32)
    hidden_dim = st.sidebar.slider("Hidden Dimension", 128, 512, 256)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 30)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Training")
        
        if st.button("Train Model"):
            with st.spinner("Training MusicVAE..."):
                # Create synthetic data
                music_data = create_synthetic_music_data(n_samples=500, sequence_length=128)
                
                # Create data loader
                dataset = TensorDataset(torch.tensor(music_data))
                train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Initialize model
                model = MusicVAE(input_dim=128, hidden_dim=hidden_dim, latent_dim=latent_dim)
                model.to(device)
                
                # Train model
                loss_history = train_vae(model, train_loader, epochs)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.trained = True
                
                # Plot training loss
                fig, ax = plt.subplots()
                ax.plot(loss_history)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss')
                st.pyplot(fig)
                
                st.success("Model trained successfully!")
    
    with col2:
        st.header("Generation")
        
        if st.session_state.trained and st.session_state.model is not None:
            prompt = st.text_input("Prompt (optional - for display only)", "Generated music sequence")
            
            if st.button("Generate Music"):
                with st.spinner("Generating music..."):
                    # Generate music
                    generated_music = generate_music(st.session_state.model, latent_dim=latent_dim, num_samples=1)
                    music_sequence = generated_music[0]
                    
                    # Visualize generated music
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(music_sequence)
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Intensity')
                    ax.set_title(f'Generated Music: {prompt}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Convert to audio
                    audio_data = music_to_audio(music_sequence)
                    
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        write(tmp_file.name, 22050, audio_data)
                        
                        # Read the file and create download link
                        with open(tmp_file.name, 'rb') as f:
                            audio_bytes = f.read()
                        
                        # Audio player
                        st.audio(audio_bytes, format='audio/wav')
                        
                        # Download button
                        st.download_button(
                            label="Download Audio",
                            data=audio_bytes,
                            file_name="generated_music.wav",
                            mime="audio/wav"
                        )
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                    
                    # Show raw data
                    with st.expander("View Raw Music Data"):
                        st.write("Generated sequence (first 20 values):")
                        st.write(music_sequence[:20])
        else:
            st.info("Please train the model first!")
    
    # Info section
    st.header("About")
    st.write("""
    This MusicVAE implementation:
    - Uses synthetic music data for training
    - Learns musical patterns in a 32-dimensional latent space
    - Generates new sequences by sampling from the latent space
    - Converts sequences to audio using simple frequency mapping
    
    The model learns to encode musical sequences into a compact latent representation
    and decode them back to generate new musical content.
    """)

if __name__ == "__main__":
    main()
