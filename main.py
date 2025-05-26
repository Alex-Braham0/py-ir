import pyaudio
from time import sleep
import wave
import numpy as np
import load_wav
import os
import json
import threading

class Main():
    def __init__(self):
        self.config = self.load_config()
        self.start_streams()
        self.menu()

    def load_config(self):
        config = {}
        # Check if there is a config file
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
                if not self.check_config_valid(config):
                    config = self.create_default_config()
        else:
            # Set settings to default
            config = self.create_default_config()

            # Save settings
            self.save_config(config)
        return config
    
    def check_config_valid(self, check_config):
        valid = True
        if ["input", "output", "default_gain", "default_drywet", "buffer_size"] not in list(check_config.keys()):
            valid = False
        elif ["format", "channels", "rate", "buffer_size"] not in check_config["input"] or ["format", "channels", "rate", "buffer_size"] not in check_config["output"]:
            valid = False
        else:
            if check_config["input"]["format"] not in [pyaudio.paInt8, pyaudio.paInt16, pyaudio.paInt24, pyaudio.paInt32, pyaudio.paFloat32] or check_config["output"]["format"] not in [pyaudio.paInt8, pyaudio.paInt16, pyaudio.paInt24, pyaudio.paInt32, pyaudio.paFloat32]:
                valid = False
            elif check_config["input"]["channels"] not in [1, 2] or check_config["output"]["channels"] not in [1, 2]:
                valid = False
            elif check_config["input"]["rate"] not in [44100, 48000] or check_config["output"]["rate"] not in [44100, 48000]:
                valid = False
            elif check_config["buffer_size"] not in [64, 128, 256, 512, 2048, 4096]:
                valid = False
            elif check_config["ir_filename"] != False or type(check_config["ir_filename"]) != str:
                valid = False
            elif check_config["default_gain"] <=0 or check_config["default_gain"] > 10:
                valid = False
            elif check_config["default_drywet"] <=0 or check_config["default_drywet"] > 1:
                valid = False
        return valid
    
    def create_default_config(self):
        # Create new config
        config = {}

        # Set Input settings
        config["input"] = {}
        config["input"]["format"] = pyaudio.paInt24
        config["input"]["channels"] = 1
        config["input"]["rate"] = 44100

        # Set Output Settings
        config["output"] = {}
        config["output"]["format"] = pyaudio.paFloat32
        config["output"]["channels"] = 1
        config["output"]["rate"] = 44100

        # Set other Settings
        config["buffer_size"] = 512
        config["ir_filename"] = False
        config["default_gain"] = 1
        config["default_drywet"] = 0.5
        return config

    def save_config(self, config):
        # Save to file
        with open("config.json", "w") as f:
            json.dump(config, f)

    def menu(self):
        while True:
            print("")
            choice = input()
            self.close_streams()

    def start_output(self):

        # Start output stream
        output_stream = self.audio.open(
            rate=self.config["output"]["rate"],
            channels=self.config["output"]["channels"],
            format=self.config["output"]["format"],
            output=True,
            frames_per_buffer=self.config["buffer_size"])
        
        print("Starting output audio stream...")
        if output_stream.is_active():
            print("Stream initialised successfully")
            return output_stream
        else:
            print("Stream failed to initialise")
            return False
        

    def start_input(self):
        # Start output stream
        input_stream = self.audio.open(
            rate=self.config["input"]["rate"],
            channels=self.config["input"]["channels"],
            format=self.config["input"]["format"],
            input=True,
            frames_per_buffer=self.config["buffer_size"])
        
        print("Starting input audio stream...")
        if input_stream.is_active():
            print("Stream initialised successfully")
            return input_stream
        else:
            print("Stream failed to initialise")
            return False
    
    def start_streams(self):

        self.gain = self.config["default_gain"]
        self.drywet = self.config["default_drywet"]

        self.audio = pyaudio.PyAudio()  # Might crash here
        self.ir = self.select_ir(self.config["ir_filename"])  # Or here
        output_stream = self.start_output()
        input_stream = self.start_input()
        if not output_stream or not input_stream:
            self.close_streams(output_stream, input_stream)
            return

        self.continue_loop = True

        self.loop = threading.Thread(target=self.perform_loop, args=(output_stream, input_stream,), daemon=True)
        self.loop.start()

    def close_streams(self, output_stream, input_stream):

        if output_stream:
            output_stream.stop_stream()
            output_stream.close()

        if input_stream:
            input_stream.stop_stream()
            input_stream.close()
        self.audio.terminate()

    def perform_loop(self, output_stream, input_stream):

        # Set starting variables
        overlap = np.zeros(len(self.ir) - 1, dtype=np.float32)
        fft_ir = None
        buffer_size = self.config["buffer_size"]
        enable_convolve = True

        while True:
            # Read input stream
            data = input_stream.read(buffer_size)

            #Convert byte array to np.array
            f32_audio = load_wav.convert_bytes_to_float32(data, self.config["input"]["format"], self.config["input"]["channels"])

            #Set input gain
            f32_audio *= self.gain

            #Convolve audio with IR
            if enable_convolve:
                convolved, overlap, fft_ir = self.ols_convolve(f32_audio, self.ir, overlap, fft_ir)

                f32_audio = self.drywet * f32_audio + (1-self.drywet) * (convolved)

            #Audio Decrease Gain
            # if np.max(np.abs(f32_audio)) > 0.9:
            #     gain /= round(np.max(np.abs(f32_audio)), 2) / 0.7
            #     print(f"Auto-reduced gain: {gain}")

            byte = f32_audio.tobytes()
            output_stream.write(byte)

    def ols_convolve(self, chunk, ir, overlap, fft_ir=None):
        """
        Overlap-save convolution using FFT.

        Args:
            chunk (np.ndarray): Input audio chunk of length L.
            ir (np.ndarray): Impulse response of length M.
            overlap (np.ndarray): Previous overlap buffer of length M-1.
            fft_ir (np.ndarray or None): Cached FFT of IR padded to N.

        Returns:
            output (np.ndarray): Output chunk of length L.
            new_overlap (np.ndarray): Updated overlap for next chunk.
            fft_ir (np.ndarray): Cached FFT of IR (for reuse).
        """
        L = len(chunk)
        M = len(ir)

        # Compute FFT size (next power of 2 >= L + M - 1)
        N = 2 ** int(np.ceil(np.log2(L + M - 1)))

        # Prepare IR FFT if needed
        if fft_ir is None or len(fft_ir) != N // 2 + 1:
            padded_ir = np.zeros(N, dtype=np.float32)
            padded_ir[:M] = ir
            fft_ir = np.fft.rfft(padded_ir)

        # Pad overlap if needed (should be length M - 1)
        if len(overlap) < M - 1:
            overlap = np.pad(overlap, (M - 1 - len(overlap), 0))
        else:
            overlap = overlap[-(M - 1):]

        # Build input block: overlap + current chunk
        input_block = np.concatenate((overlap, chunk)).astype(np.float32)

        # FFT of input block
        fft_input = np.fft.rfft(input_block, n=N)

        # Multiply in frequency domain
        fft_product = fft_input * fft_ir

        # Inverse FFT
        conv_result = np.fft.irfft(fft_product, n=N)

        # Save the last M-1 samples as new overlap
        new_overlap = input_block[-(M - 1):]

        # Discard first M-1 samples (overlap), return valid part
        output = conv_result[M - 1:M - 1 + L].astype(np.float32)

        return output, new_overlap, fft_ir

    def select_ir(self, ir_filename):
        if ir_filename and os.path.exists("ir-library/"+ir_filename):
            print("Loading "+ir_filename)
            result = self.open_and_convert_ir(ir_filename)
            if result[0]:
                print("Successfully loaded file")
                ir_np_array = result[1]
            else:
                print("Failed to load file")
                ir_np_array = self.create_delta_ir()

        else:
            print("IR not found")
            ir_np_array = self.create_delta_ir()
        
        return ir_np_array
    
    def create_delta_ir(self):
        print("Creating delta IR")
        length = 1024  # Total length of the IR
        delta_ir = np.zeros(length, dtype=np.float32)  # Create array of zeros
        delta_ir[0] = 1.0  # Set first sample to 1.0
        return delta_ir
    
    def open_and_convert_ir(self, filename):
        data = load_wav.load_wav("ir-library/"+filename, buffer_split=False)
        if data:
        
            ir_bytes, ir_obj = data
            width = ir_obj.getsampwidth()
            channels = ir_obj.getnchannels()

            # Convert to float 32
            ir_array = load_wav.convert_bytes_to_float32(ir_bytes, width, channels)

            # Normalise by energy
            ir_energy = np.sqrt(np.sum(ir_array**2))
            if ir_energy > 0:   
                ir_array /= ir_energy
            return [True, ir_array]
        else:
            return [False]


if __name__ == "__main__":
    Main()