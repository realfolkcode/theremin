import numpy as np
import torch

class Theremin():
    def __init__(self, detector, model=None):
        self.samplerate = 48000
        self.blocksize = 512
        self.freq = 440.0
        self.prev_freq = self.freq
        self.amplitude = 1
        self.channels = 1
        self.start_idx = 0
        self.detector = detector
        self.model = model
        self.register = (200, 1200)

    def sine(self, freq, x):
        return self.amplitude * np.sin(2 * np.pi * freq * x)

    def get_data(self, freq, frames, idx):
        x = (idx + np.arange(frames)) / self.samplerate
        x = x.reshape(-1, 1)
        return self.sine(freq, x)

    def callback(self, outdata, frames, t, status):
        self.update_freq(self.detector.center["pitch"], self.detector.height)
        self.update_amplitude(self.detector.center["dynamics"], self.detector.height)
        if self.model is None:
            idx = (self.prev_freq * self.start_idx) / self.freq
            data = self.get_data(self.freq, frames, idx)
            self.start_idx = idx + frames
            self.prev_freq = self.freq
            outdata[:] = data
        else:
            pitch = self.freq * torch.ones(1, 1, 1)
            loudness = 110 * self.amplitude * torch.ones(1, 1, 1)
            data = self.model(pitch, loudness).detach().numpy()
            outdata[:] = data[0]


    def update_freq(self, center, height):
        low = self.register[0]
        diff = self.register[1] - self.register[0]
        self.freq = diff * (height - center[1]) / height + low

    def update_amplitude(self, center, height):
        self.amplitude = (height - center[1]) / height
