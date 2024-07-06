import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

def butter_highpass(cutoff, sr, order=5):
    # Calculate Nyquist frequency and normalize the cutoff frequency
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    # Design a high-pass Butterworth filter
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, sr, order=5):
    # Apply high-pass filter to the data
    b, a = butter_highpass(cutoff, sr, order=order)
    y = lfilter(b, a, data)
    return y

def boost_high_frequencies(audio, sr, boost_factor=1.15, cutoff=2000):
    """
    Boost the high-frequency part of the audio signal.
    :param audio: Input audio signal
    :param sr: Sampling rate
    :param boost_factor: High-frequency boost ratio
    :param cutoff: Cutoff frequency for high-frequency boost (Hz)
    :return: Boosted audio signal
    """
    # Separate high-frequency part
    high_freqs = highpass_filter(audio, cutoff, sr)
    
    # Boost the high-frequency part
    boosted_high_freqs = high_freqs * boost_factor
    
    # Add the boosted high-frequency part back to the original signal
    boosted_audio = audio + boosted_high_freqs
    
    # Clip to avoid clipping distortion
    boosted_audio = np.clip(boosted_audio, -1.0, 1.0)
    
    return boosted_audio

# Load the audio file
audio_path = '1-1.wav'
audio, sr = librosa.load(audio_path, sr=None)

# Boost high frequencies
boost_factor = 1.15  # Boost high frequencies by 15%
cutoff = 2000  # Boost frequencies starting from 2000Hz
boosted_audio = boost_high_frequencies(audio, sr, boost_factor=boost_factor, cutoff=cutoff)

# Save the processed audio
output_path = '1-4.wav'
sf.write(output_path, boosted_audio, sr)
