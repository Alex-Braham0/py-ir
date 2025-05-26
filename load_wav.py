import wave
import numpy as np

BUFFER_SIZE = 512


def load_wav(filename, buffer_split=True):
    print(f"Loading file \"{filename}\"")
    try:
        obj = wave.open(filename, "rb")
        return wav_to_audio(obj, buffer_split)
    except:
        return (False)
    
def wav_to_audio(wav_obj, buffer_split):
    bytes_per_frame = wav_obj.getnchannels() * wav_obj.getparams()[1]
    buffer_size_bytes = BUFFER_SIZE * bytes_per_frame
    frames = wav_obj.readframes(wav_obj.getnframes())
    if buffer_split:
        bytes = [frames[i:i + buffer_size_bytes] for i in range(0, len(frames), buffer_size_bytes)]
    # floating_audio = convert_bytes_to_float32(bytes, wav_obj.getsampwidth(), wav_obj.getnchannels() == 2 and not stereo)
        return (bytes, wav_obj)
    else:
        return (frames, wav_obj)

def wav_sample_depth_to_pyaudio(sample_depth):
    return {1: 16, 2: 8, 3: 4, 4: 2}[sample_depth]

def convert_bytes_to_float32(audio_bytes, bit_depth, channels, stereo=False):

    # bit_depth=wav_sample_depth_to_pyaudio(bit_depth)
 # Map bit depth to dtype and scaling factor
    format_info = {
        16: (np.uint8, 2**7),      # Unsigned 8-bit
        8: (np.int16, 2**15),     # Signed 16-bit
        4: (np.int32, 2**23),     # Signed 24-bit (special handling)
        2: (np.int32, 2**31)      # Signed 32-bit
    }
    
    if bit_depth not in format_info:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    dtype, scale = format_info[bit_depth]
    
    if bit_depth == 4:
        if channels == 2:
            # Reshape into (samples, 6 bytes [L, R, L, R...])
            audio_bytes = np.frombuffer(audio_bytes, dtype=np.uint8).reshape(-1, 6)

            # Extract LEFT channel (first 3 bytes of each 6-byte block)
            audio_bytes = audio_bytes[:, :3]  # Shape: (N, 3)

        elif channels == 1:
            # Mono: reshape to (samples, 3)
            audio_bytes = np.frombuffer(audio_bytes, dtype=np.uint8).reshape(-1, 3)

        else:
            raise ValueError("Unsupported number of channels")
        
        
        # Convert 24-bit to 32-bit float
        # if little_endian:
        int24 = (
            audio_bytes[:, 0].astype(np.int32) + 
            (audio_bytes[:, 1].astype(np.int32) << 8) + 
            (audio_bytes[:, 2].astype(np.int32) << 16)
        )
        # else:
        #     int24 = (
        #         left_samples[:, 2].astype(np.int32) + 
        #         (left_samples[:, 1].astype(np.int32) << 8) + 
        #         (left_samples[:, 0].astype(np.int32) << 16)
        #     )
        
        # Sign-extend 24-bit to 32-bit and normalize to [-1.0, 1.0]
        int32 = np.where(int24 & 0x800000, int24 | (~0xFFFFFF), int24)
        return int32.astype(np.float32) / (2 ** 23)
    else:
        # Load raw bytes into float32
        data = np.frombuffer(audio_bytes, dtype=dtype)
        data = data.astype(np.float32)

        # Handle bit-depth conversion FIRST (before any processing)
        if bit_depth == 8:
            data = (data - 128) / 128.0  # Unsigned 8-bit -> [-1, 1]
        elif bit_depth == 16:
            data = data / 32768.0  # Signed 16-bit -> [-1, 1]

        if channels == 2:
            # Reshape stereo data (L/R channels)
            ir_stereo = data.reshape(-1, 2).T  # Shape: (2, N)

            # Sum to mono (optional, or process channels separately)
            data = np.sum(ir_stereo, axis=0)  # Shape: (N,)

        # Normalize to [-1, 1] (safeguard against clipping)
        data /= np.max(np.abs(data)) + 1e-9  # Avoid division by zero

        bytes = data
    # if channels == 2 and not stereo:
    #     # reshape to (n_samples, 2) and take mean of L/R
    #     bytes = bytes.reshape(-1, 2).mean(axis=1)
    return bytes

