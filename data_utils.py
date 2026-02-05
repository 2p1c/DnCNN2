"""
Ultrasonic Signal Dataset Module

Implements synthetic ultrasonic pulse-echo signal generation for CAE training.
Based on "Ultrasonic signal noise reduction based on convolutional autoencoders for NDT applications"

Signal Parameters (from paper):
- Sampling Rate: 6.25 MHz
- Duration: 160 μs
- Total Points: 1000
- Center Frequency: 250 kHz
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Literal, List
from pathlib import Path
from scipy import signal as scipy_signal


class UltrasonicDataset(Dataset):
    """
    Dataset for ultrasonic signal denoising.
    
    Supports two modes:
    - 'synthetic': Generate Gabor pulse signals on-the-fly
    - 'file': Load signals from .npy or .csv files (placeholder for future)
    
    Data Augmentation (for training):
    - Time flip: Reverse signal in time
    - Amplitude scaling: Random scaling of signal amplitude
    - Noise injection: Add small random noise to noisy signal
    - Time shift: Circular shift in time domain
    """
    
    # Physical constants matching paper specifications
    SAMPLING_RATE: float = 6.25e6  # 6.25 MHz
    DURATION: float = 160e-6       # 160 μs
    NUM_POINTS: int = 1000         # Total data points
    CENTER_FREQ: float = 250e3     # 250 kHz center frequency
    
    def __init__(
        self,
        mode: Literal['synthetic', 'file'] = 'synthetic',
        num_samples: int = 1000,
        snr_range: Tuple[float, ...] = (-10.0, -8.0, -6.0),
        num_pulse_bursts_range: Tuple[int, int] = (1, 4),
        cycles_per_burst_range: Tuple[int, int] = (3, 12),
        noise_types: Tuple[str, ...] = ('gaussian', 'pink', 'impulse', 'periodic', 'bandlimited', 'mixed'),
        noise_intensity: float = 1.5,
        data_path: Optional[str] = None,
        seed: Optional[int] = None,
        augment: bool = False,  # Enable data augmentation
        augment_prob: float = 0.3  # Probability of applying each augmentation
    ):
        """
        Initialize the dataset.
        
        Args:
            mode: 'synthetic' for on-the-fly generation, 'file' for loading from disk
            num_samples: Number of samples to generate (synthetic mode)
            snr_range: Tuple of SNR values in dB (negative = noise dominant)
            num_pulse_bursts_range: (min, max) number of pulse bursts per signal
            cycles_per_burst_range: (min, max) number of carrier cycles per burst
            noise_types: Types of noise to randomly select from:
                - 'gaussian': White Gaussian noise
                - 'pink': 1/f noise (low frequency dominant)
                - 'impulse': Random spike noise
                - 'periodic': Power line interference
                - 'bandlimited': Noise in specific frequency band
                - 'mixed': Combination of multiple noise types
            noise_intensity: Multiplier for noise power (>1 for stronger noise)
            data_path: Path to data files (file mode)
            seed: Random seed for reproducibility
            augment: Enable data augmentation (for training set)
            augment_prob: Probability of applying each augmentation (0-1)
        """
        self.mode = mode
        self.num_samples = num_samples
        self.snr_range = snr_range
        self.num_pulse_bursts_range = num_pulse_bursts_range
        self.cycles_per_burst_range = cycles_per_burst_range
        self.noise_types = noise_types
        self.noise_intensity = noise_intensity
        self.data_path = Path(data_path) if data_path else None
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Set random seed
        self.rng = np.random.default_rng(seed)
            
        # Time vector for signal generation
        self.t = np.linspace(0, self.DURATION, self.NUM_POINTS, dtype=np.float32)
        
        # Placeholder for file mode data
        self.clean_signals: List[np.ndarray] = []
        self.noisy_signals: List[np.ndarray] = []
        
        if mode == 'file':
            self._load_file_data()
        
    def _load_file_data(self) -> None:
        """
        Placeholder for loading data from files.
        
        Expected file structure:
        - data_path/clean/*.npy or .csv (clean reference signals)
        - data_path/noisy/*.npy or .csv (noisy input signals)
        
        TODO: Implement file loading for real experimental data.
        """
        if self.data_path is None:
            raise ValueError("data_path must be provided for 'file' mode")
        
        clean_dir = self.data_path / 'clean'
        noisy_dir = self.data_path / 'noisy'
        
        # Check if directories exist
        if not clean_dir.exists() or not noisy_dir.exists():
            print(f"[WARNING] Data directories not found at {self.data_path}")
            print(f"  Expected: {clean_dir} and {noisy_dir}")
            print(f"  Running in empty file mode - implement _load_file_data() for real data.")
            return
        
        # Load .npy files
        for clean_file in sorted(clean_dir.glob('*.npy')):
            self.clean_signals.append(np.load(clean_file).astype(np.float32))
            
        for noisy_file in sorted(noisy_dir.glob('*.npy')):
            self.noisy_signals.append(np.load(noisy_file).astype(np.float32))
        
        # Alternatively, load .csv files
        if len(self.clean_signals) == 0:
            for clean_file in sorted(clean_dir.glob('*.csv')):
                self.clean_signals.append(
                    np.loadtxt(clean_file, delimiter=',').astype(np.float32)
                )
            for noisy_file in sorted(noisy_dir.glob('*.csv')):
                self.noisy_signals.append(
                    np.loadtxt(noisy_file, delimiter=',').astype(np.float32)
                )
        
        if len(self.clean_signals) != len(self.noisy_signals):
            raise ValueError(
                f"Mismatch: {len(self.clean_signals)} clean files vs "
                f"{len(self.noisy_signals)} noisy files"
            )
            
        print(f"[INFO] Loaded {len(self.clean_signals)} signal pairs from {self.data_path}")
        
    def _generate_gabor_pulse(
        self,
        delay: float,
        amplitude: float = 1.0,
        bandwidth: float = 0.5
    ) -> np.ndarray:
        """
        Generate a Gabor pulse (Gaussian-modulated sinusoid).
        
        Formula: s(t) = A * exp(-((t-τ)²)/(2σ²)) * sin(2πf_c(t-τ))
        
        This models the ultrasonic pulse-echo response in NDT applications.
        
        Args:
            delay: Time delay τ in seconds (echo arrival time)
            amplitude: Pulse amplitude A
            bandwidth: Fractional bandwidth (affects Gaussian envelope width)
            
        Returns:
            Gabor pulse signal array of shape (NUM_POINTS,)
        """
        # Gaussian envelope width (inversely related to bandwidth)
        # Wider bandwidth = narrower pulse in time domain
        sigma = bandwidth / (2 * np.pi * self.CENTER_FREQ)
        
        # Gaussian envelope centered at delay
        envelope = np.exp(-((self.t - delay) ** 2) / (2 * sigma ** 2))
        
        # Modulated sinusoidal carrier
        carrier = np.sin(2 * np.pi * self.CENTER_FREQ * (self.t - delay))
        
        return (amplitude * envelope * carrier).astype(np.float32)
    
    def _generate_pulse_burst(
        self,
        center_time: float,
        num_cycles: int,
        amplitude: float = 1.0,
        carrier_freq: Optional[float] = None,
        envelope_type: str = 'gaussian'
    ) -> np.ndarray:
        """
        Generate a pulse burst (carrier wave modulated by envelope).
        
        This better simulates real ultrasonic transducer signals which
        consist of multiple carrier cycles within an envelope.
        
        Args:
            center_time: Center time of the burst in seconds
            num_cycles: Number of carrier cycles in the burst
            amplitude: Peak amplitude
            carrier_freq: Carrier frequency (None for default CENTER_FREQ)
            envelope_type: 'gaussian', 'hanning', 'tukey', or 'exponential'
            
        Returns:
            Pulse burst signal array of shape (NUM_POINTS,)
        """
        if carrier_freq is None:
            carrier_freq = self.CENTER_FREQ
        
        # Calculate burst duration based on number of cycles
        burst_duration = num_cycles / carrier_freq
        
        # Generate carrier wave
        carrier = np.sin(2 * np.pi * carrier_freq * (self.t - center_time))
        
        # Generate envelope based on type
        # Time relative to burst center
        t_rel = self.t - center_time
        half_duration = burst_duration / 2
        
        if envelope_type == 'gaussian':
            # Gaussian envelope
            sigma = burst_duration / 4  # 95% of energy within burst duration
            envelope = np.exp(-(t_rel ** 2) / (2 * sigma ** 2))
            
        elif envelope_type == 'hanning':
            # Hanning window envelope
            envelope = np.zeros_like(self.t)
            mask = np.abs(t_rel) <= half_duration
            envelope[mask] = 0.5 * (1 + np.cos(np.pi * t_rel[mask] / half_duration))
            
        elif envelope_type == 'tukey':
            # Tukey window (tapered cosine) - flat top with tapered edges
            envelope = np.zeros_like(self.t)
            alpha = 0.5  # Taper ratio
            mask = np.abs(t_rel) <= half_duration
            t_norm = t_rel[mask] / half_duration  # Normalized to [-1, 1]
            
            env_values = np.ones_like(t_norm)
            # Taper regions
            taper_mask = np.abs(t_norm) > (1 - alpha)
            env_values[taper_mask] = 0.5 * (1 + np.cos(np.pi * (np.abs(t_norm[taper_mask]) - (1 - alpha)) / alpha))
            envelope[mask] = env_values
            
        elif envelope_type == 'exponential':
            # Exponential decay envelope (asymmetric - realistic for echoes)
            envelope = np.zeros_like(self.t)
            # Rising edge
            mask_rise = (t_rel >= -half_duration) & (t_rel < 0)
            envelope[mask_rise] = np.exp(-(t_rel[mask_rise] + half_duration) / (half_duration / 3))
            # Falling edge (slower decay)
            mask_fall = (t_rel >= 0) & (t_rel <= half_duration * 2)
            envelope[mask_fall] = np.exp(-t_rel[mask_fall] / (half_duration / 1.5))
            
        else:
            # Default: Gaussian
            sigma = burst_duration / 4
            envelope = np.exp(-(t_rel ** 2) / (2 * sigma ** 2))
        
        return (amplitude * envelope * carrier).astype(np.float32)
    
    def _generate_clean_signal(self) -> np.ndarray:
        """
        Generate a clean ultrasonic signal with multiple pulse bursts.
        
        Simulates realistic ultrasonic signals with:
        - Multiple pulse bursts at different times
        - Each burst contains multiple carrier cycles within an envelope
        - Variable carrier frequencies and envelope shapes
        - Amplitude attenuation for later echoes
        
        Returns:
            Clean signal array of shape (NUM_POINTS,)
        """
        signal = np.zeros(self.NUM_POINTS, dtype=np.float32)
        
        # Random number of pulse bursts
        num_bursts = self.rng.integers(
            self.num_pulse_bursts_range[0], 
            self.num_pulse_bursts_range[1] + 1
        )
        
        # Available envelope types for variety
        envelope_types = ['gaussian', 'hanning', 'tukey', 'exponential']
        
        # Generate each pulse burst
        for i in range(num_bursts):
            # Random center time (avoid edges)
            min_time = 0.08 * self.DURATION
            max_time = 0.92 * self.DURATION
            center_time = self.rng.uniform(min_time, max_time)
            
            # Random number of cycles in this burst
            num_cycles = self.rng.integers(
                self.cycles_per_burst_range[0],
                self.cycles_per_burst_range[1] + 1
            )
            
            # Amplitude with attenuation for later bursts
            base_amplitude = self.rng.uniform(0.6, 1.0)
            amplitude = base_amplitude * (0.75 ** i)
            
            # Random carrier frequency variation (±20% of center freq)
            carrier_freq = self.CENTER_FREQ * self.rng.uniform(0.8, 1.2)
            
            # Random envelope type
            envelope_type = self.rng.choice(envelope_types)
            
            # Generate and add the pulse burst
            signal += self._generate_pulse_burst(
                center_time=center_time,
                num_cycles=num_cycles,
                amplitude=amplitude,
                carrier_freq=carrier_freq,
                envelope_type=envelope_type
            )
        
        # Normalize to [-1, 1] range
        max_val = np.abs(signal).max()
        if max_val > 1e-8:
            signal = signal / max_val
            
        return signal
    
    def _add_gaussian_noise(
        self, 
        signal: np.ndarray, 
        snr_db: float
    ) -> np.ndarray:
        """
        Add Gaussian white noise to achieve target SNR.
        
        SNR(dB) = 10 * log10(P_signal / P_noise)
        
        Args:
            signal: Clean signal array
            snr_db: Target Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy signal array
        """
        # Calculate signal power
        signal_power = np.mean(signal ** 2)
        
        # Calculate required noise power for target SNR
        # P_noise = P_signal / 10^(SNR/10)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate Gaussian white noise with calculated variance
        noise_std = np.sqrt(noise_power)
        noise = self.rng.normal(0, noise_std, signal.shape).astype(np.float32)
        
        return signal + noise
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """
        Generate pink noise (1/f noise) - power decreases with frequency.
        
        Pink noise has more energy at low frequencies, simulating
        material grain noise and structural interference in NDT.
        
        Args:
            length: Number of samples
            
        Returns:
            Pink noise array (normalized)
        """
        # Generate white noise in frequency domain
        white = self.rng.standard_normal(length)
        
        # Create 1/f filter
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1e-10  # Avoid division by zero
        
        # 1/f spectrum (pink noise)
        fft_white = np.fft.rfft(white)
        fft_pink = fft_white / np.sqrt(freqs)
        
        # Transform back to time domain
        pink = np.fft.irfft(fft_pink, n=length)
        
        # Normalize
        pink = pink / (np.abs(pink).max() + 1e-10)
        
        return pink.astype(np.float32)
    
    def _generate_impulse_noise(
        self, 
        length: int, 
        density: float = 0.02,
        amplitude_range: Tuple[float, float] = (0.5, 2.0)
    ) -> np.ndarray:
        """
        Generate impulse (spike) noise - random high-amplitude spikes.
        
        Simulates electrical interference, dropouts, and transient disturbances.
        
        Args:
            length: Number of samples
            density: Probability of spike at each sample (0.01-0.05 typical)
            amplitude_range: (min, max) spike amplitude
            
        Returns:
            Impulse noise array
        """
        noise = np.zeros(length, dtype=np.float32)
        
        # Random spike positions
        spike_mask = self.rng.random(length) < density
        num_spikes = spike_mask.sum()
        
        # Random amplitudes and signs
        amplitudes = self.rng.uniform(amplitude_range[0], amplitude_range[1], num_spikes)
        signs = self.rng.choice([-1, 1], num_spikes)
        
        noise[spike_mask] = amplitudes * signs
        
        return noise
    
    def _generate_periodic_noise(
        self, 
        length: int,
        base_freq: Optional[float] = None,
        num_harmonics: int = 3
    ) -> np.ndarray:
        """
        Generate periodic interference noise.
        
        Simulates power line interference (50/60 Hz and harmonics),
        scaled to ultrasonic frequency range for realistic interference.
        
        Args:
            length: Number of samples
            base_freq: Base frequency in Hz (None for random)
            num_harmonics: Number of harmonics to include
            
        Returns:
            Periodic noise array (normalized)
        """
        t = self.t
        
        # Random base frequency if not specified (scaled to signal range)
        if base_freq is None:
            # Frequencies that could interfere with 250 kHz signal
            base_freq = self.rng.uniform(10e3, 100e3)  # 10-100 kHz range
        
        noise = np.zeros(length, dtype=np.float32)
        
        for h in range(1, num_harmonics + 1):
            freq = base_freq * h
            amplitude = 1.0 / h  # Harmonics decay
            phase = self.rng.uniform(0, 2 * np.pi)
            noise += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Normalize
        noise = noise / (np.abs(noise).max() + 1e-10)
        
        return noise.astype(np.float32)
    
    def _generate_bandlimited_noise(
        self, 
        length: int,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate band-limited noise in a specific frequency range.
        
        Simulates interference from specific frequency sources
        like motors, other ultrasonic equipment, or resonances.
        
        Args:
            length: Number of samples
            low_freq: Lower cutoff frequency (None for random)
            high_freq: Upper cutoff frequency (None for random)
            
        Returns:
            Band-limited noise array (normalized)
        """
        # Generate white noise
        white = self.rng.standard_normal(length)
        
        # Random frequency band if not specified
        if low_freq is None:
            low_freq = self.rng.uniform(50e3, 200e3)
        if high_freq is None:
            high_freq = low_freq + self.rng.uniform(50e3, 150e3)
        
        # Ensure valid frequency range
        nyquist = self.SAMPLING_RATE / 2
        low_freq = min(low_freq, nyquist * 0.8)
        high_freq = min(high_freq, nyquist * 0.95)
        
        if low_freq >= high_freq:
            low_freq = high_freq * 0.5
        
        # Design bandpass filter
        try:
            sos = scipy_signal.butter(
                4, 
                [low_freq, high_freq], 
                btype='band', 
                fs=self.SAMPLING_RATE,
                output='sos'
            )
            noise = scipy_signal.sosfilt(sos, white)
        except Exception:
            # Fallback to white noise if filter fails
            noise = white
        
        # Normalize
        noise = noise / (np.abs(noise).max() + 1e-10)
        
        return noise.astype(np.float32)
    
    def _generate_chirp_interference(
        self,
        length: int
    ) -> np.ndarray:
        """
        Generate chirp (frequency sweep) interference.
        
        Simulates swept-frequency interference from other equipment.
        
        Args:
            length: Number of samples
            
        Returns:
            Chirp interference array (normalized)
        """
        t = self.t
        
        # Random chirp parameters
        f0 = self.rng.uniform(100e3, 200e3)  # Start frequency
        f1 = self.rng.uniform(300e3, 500e3)  # End frequency
        
        # Generate chirp
        chirp = scipy_signal.chirp(t, f0, t[-1], f1, method='linear')
        
        # Random amplitude modulation
        mod_freq = self.rng.uniform(1e3, 10e3)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        chirp = chirp * modulation
        
        # Normalize
        chirp = chirp / (np.abs(chirp).max() + 1e-10)
        
        return chirp.astype(np.float32)
    
    def _add_complex_noise(
        self, 
        signal: np.ndarray, 
        snr_db: float,
        noise_type: str
    ) -> np.ndarray:
        """
        Add complex noise of specified type to achieve target SNR.
        
        Args:
            signal: Clean signal array
            snr_db: Target Signal-to-Noise Ratio in dB (negative = noise dominant)
            noise_type: Type of noise ('gaussian', 'pink', 'impulse', 
                       'periodic', 'bandlimited', 'mixed')
            
        Returns:
            Noisy signal array
        """
        length = len(signal)
        signal_power = np.mean(signal ** 2)
        
        # Apply noise intensity multiplier for stronger noise
        # target_noise_power = signal_power / 10^(SNR/10) * intensity
        target_noise_power = signal_power / (10 ** (snr_db / 10)) * self.noise_intensity
        
        if noise_type == 'gaussian':
            noise = self.rng.standard_normal(length).astype(np.float32)
            
        elif noise_type == 'pink':
            noise = self._generate_pink_noise(length)
            # Add some high-frequency content
            hf_noise = self.rng.standard_normal(length).astype(np.float32) * 0.3
            noise = noise + hf_noise
            
        elif noise_type == 'impulse':
            # More aggressive impulse noise
            density = self.rng.uniform(0.03, 0.10)  # Higher density
            impulse = self._generate_impulse_noise(
                length, 
                density=density,
                amplitude_range=(1.0, 4.0)  # Stronger spikes
            )
            gaussian = self.rng.standard_normal(length).astype(np.float32) * 0.5
            noise = impulse + gaussian
            
        elif noise_type == 'periodic':
            # Multiple periodic components
            periodic1 = self._generate_periodic_noise(length, num_harmonics=5)
            periodic2 = self._generate_periodic_noise(length, num_harmonics=3)
            gaussian = self.rng.standard_normal(length).astype(np.float32) * 0.4
            noise = periodic1 * 0.5 + periodic2 * 0.3 + gaussian
            
        elif noise_type == 'bandlimited':
            # Multiple overlapping bands
            band1 = self._generate_bandlimited_noise(length)
            band2 = self._generate_bandlimited_noise(length)
            gaussian = self.rng.standard_normal(length).astype(np.float32) * 0.3
            noise = band1 * 0.5 + band2 * 0.4 + gaussian
            
        elif noise_type == 'chirp':
            chirp = self._generate_chirp_interference(length)
            gaussian = self.rng.standard_normal(length).astype(np.float32) * 0.4
            noise = chirp * 0.8 + gaussian
            
        elif noise_type == 'mixed':
            # Aggressively combine 3-5 noise types
            num_types = self.rng.integers(3, 6)
            available_types = ['gaussian', 'pink', 'impulse', 'periodic', 'bandlimited', 'chirp']
            selected_types = self.rng.choice(available_types, min(num_types, len(available_types)), replace=False)
            
            noise = np.zeros(length, dtype=np.float32)
            weights = self.rng.dirichlet(np.ones(len(selected_types)) * 0.5)  # More uniform weights
            
            for ntype, weight in zip(selected_types, weights):
                if ntype == 'gaussian':
                    component = self.rng.standard_normal(length).astype(np.float32)
                elif ntype == 'pink':
                    component = self._generate_pink_noise(length)
                elif ntype == 'impulse':
                    component = self._generate_impulse_noise(length, density=0.05, amplitude_range=(1.0, 3.0))
                elif ntype == 'periodic':
                    component = self._generate_periodic_noise(length, num_harmonics=4)
                elif ntype == 'bandlimited':
                    component = self._generate_bandlimited_noise(length)
                elif ntype == 'chirp':
                    component = self._generate_chirp_interference(length)
                else:
                    component = self.rng.standard_normal(length).astype(np.float32)
                
                # Normalize component
                component = component / (np.abs(component).max() + 1e-10)
                noise += weight * component
                
            # Add extra Gaussian layer for complexity
            noise += self.rng.standard_normal(length).astype(np.float32) * 0.2
        else:
            # Default to Gaussian
            noise = self.rng.standard_normal(length).astype(np.float32)
        
        # Scale noise to achieve target SNR (with intensity multiplier already applied)
        current_noise_power = np.mean(noise ** 2)
        if current_noise_power > 1e-10:
            scale = np.sqrt(target_noise_power / current_noise_power)
            noise = noise * scale
        
        return (signal + noise).astype(np.float32)
    
    def _apply_augmentation(
        self, 
        noisy: np.ndarray, 
        clean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to signal pairs.
        
        Augmentations (applied with probability augment_prob each):
        1. Time flip: Reverse signal in time domain
        2. Amplitude scaling: Scale by random factor [0.8, 1.2]
        3. Noise injection: Add small Gaussian noise to noisy signal
        4. Time shift: Circular shift by random amount
        
        Args:
            noisy: Noisy signal array
            clean: Clean signal array
            
        Returns:
            Tuple of (augmented_noisy, augmented_clean)
        """
        noisy = noisy.copy()
        clean = clean.copy()
        
        # 1. Time flip (reverse) - apply to both signals together
        if self.rng.random() < self.augment_prob:
            noisy = noisy[::-1].copy()
            clean = clean[::-1].copy()
        
        # 2. Amplitude scaling - same scale for both to preserve relationship
        if self.rng.random() < self.augment_prob:
            scale = self.rng.uniform(0.8, 1.2)
            noisy = noisy * scale
            clean = clean * scale
        
        # 3. Noise injection - only to noisy signal (simulates varying noise levels)
        if self.rng.random() < self.augment_prob:
            noise_std = self.rng.uniform(0.01, 0.05) * np.std(noisy)
            extra_noise = self.rng.standard_normal(len(noisy)).astype(np.float32) * noise_std
            noisy = noisy + extra_noise
        
        # 4. Time shift (circular) - apply to both signals together
        if self.rng.random() < self.augment_prob:
            shift = self.rng.integers(-50, 51)  # Shift up to 50 samples (~8μs)
            noisy = np.roll(noisy, shift)
            clean = np.roll(clean, shift)
        
        return noisy.astype(np.float32), clean.astype(np.float32)
    
    def __len__(self) -> int:
        """Return dataset length."""
        if self.mode == 'file':
            return len(self.clean_signals)
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a (noisy, clean) signal pair.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (noisy_signal, clean_signal), each of shape (1, 1000)
        """
        if self.mode == 'file':
            # File mode: return loaded data
            if len(self.clean_signals) == 0:
                raise RuntimeError("No data loaded in file mode. Check data_path.")
            clean = self.clean_signals[idx]
            noisy = self.noisy_signals[idx]
        else:
            # Synthetic mode: generate on-the-fly
            clean = self._generate_clean_signal()
            
            # Randomly select SNR and noise type
            snr_db = self.rng.choice(self.snr_range)
            noise_type = self.rng.choice(self.noise_types)
            
            # Add complex noise
            noisy = self._add_complex_noise(clean, snr_db, noise_type)
        
        # Apply data augmentation if enabled
        if self.augment:
            noisy, clean = self._apply_augmentation(noisy, clean)
        
        # Convert to tensors with channel dimension: (1, 1000)
        clean_tensor = torch.from_numpy(clean).unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)
        
        return noisy_tensor, clean_tensor


def create_dataloaders(
    num_train: int = 5000,
    num_val: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    mode: Literal['synthetic', 'file'] = 'synthetic',
    data_path: Optional[str] = None,
    augment: bool = False  # Enable augmentation for training set
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        num_train: Number of training samples (only for synthetic mode)
        num_val: Number of validation samples (only for synthetic mode)
        batch_size: Batch size for both loaders
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        mode: 'synthetic' for generated data, 'file' for loading from disk
        data_path: Path to data directory (for file mode)
            Expected structure:
                data_path/train/clean/*.npy
                data_path/train/noisy/*.npy
                data_path/val/clean/*.npy
                data_path/val/noisy/*.npy
        augment: Enable data augmentation for training set
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if mode == 'file':
        if data_path is None:
            raise ValueError("data_path must be provided for file mode")
        
        data_root = Path(data_path)
        train_path = data_root / 'train'
        val_path = data_root / 'val'
        
        if not train_path.exists() or not val_path.exists():
            raise ValueError(f"Expected train/ and val/ directories in {data_path}")
        
        train_dataset = UltrasonicDataset(
            mode='file',
            data_path=str(train_path),
            seed=seed,
            augment=augment,  # Enable augmentation for training
            augment_prob=0.5
        )
        
        val_dataset = UltrasonicDataset(
            mode='file',
            data_path=str(val_path),
            seed=seed + 1000,
            augment=False  # Never augment validation set
        )
        
        aug_str = " (with augmentation)" if augment else ""
        print(f"[INFO] File mode: {len(train_dataset)} training{aug_str}, {len(val_dataset)} validation samples")
    else:
        train_dataset = UltrasonicDataset(
            mode='synthetic',
            num_samples=num_train,
            seed=seed,
            augment=augment,
            augment_prob=0.5
        )
        
        val_dataset = UltrasonicDataset(
            mode='synthetic',
            num_samples=num_val,
            seed=seed + 1000,  # Different seed for validation set
            augment=False  # Never augment validation set
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of the dataset
    print("Testing UltrasonicDataset...")
    
    dataset = UltrasonicDataset(num_samples=10, seed=42)
    noisy, clean = dataset[0]
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Noisy shape: {noisy.shape}, dtype: {noisy.dtype}")
    print(f"Clean shape: {clean.shape}, dtype: {clean.dtype}")
    print(f"Noisy range: [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"Clean range: [{clean.min():.4f}, {clean.max():.4f}]")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        num_train=100, num_val=20, batch_size=16
    )
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    batch_noisy, batch_clean = next(iter(train_loader))
    print(f"Batch noisy shape: {batch_noisy.shape}")
    print(f"Batch clean shape: {batch_clean.shape}")
