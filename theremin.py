import numpy as np

class Theremin():
    def __init__(self, detector):
        self.samplerate = 44100
        self.blocksize = 0
        self.freq = 440.0
        self.prev_freq = self.freq
        self.amplitude = 1
        self.channels = 2
        self.start_idx = 0
        self.detector = detector

    def sine(self, freq, x):
        return self.amplitude * np.sin(2 * np.pi * freq * x)

    def get_data(self, freq, frames, idx):
        x = (idx + np.arange(frames)) / self.samplerate
        x = x.reshape(-1, 1)
        return self.sine(freq, x)

    def callback(self, outdata, frames, t, status):
        self.update_freq(self.detector.center["pitch"], self.detector.height)
        self.update_amplitude(self.detector.center["dynamics"], self.detector.height)
        idx = (self.prev_freq * self.start_idx) / self.freq
        data = self.get_data(self.freq, frames, idx)
        self.start_idx = idx + frames
        self.prev_freq = self.freq
        outdata[:] = data

    def update_freq(self, center, height):
        self.freq = (height - center[1]) * 2

    def update_amplitude(self, center, height):
        self.amplitude = (height - center[1]) / height