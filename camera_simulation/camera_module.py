import torch.nn as nn
import math
import numpy as np


class CameraModule(nn.Module):
    def __init__(self, iso=2000, shutter_speed=1/60, aperture=9)-> None:
        super().__init__()
        self.iso = iso
        self.shutter_speed = shutter_speed
        self.aperture = aperture
        self.iso_factor = self.iso / 100
        self.qe = 0.7  # Quantum Efficiency
        self.inverse_K = 8.4  # Inverse of the camera's gain factor
        self.dark_noise_sigma = 6.8
        self.saturation_capacity = 32700
        self.abs_sensitivity_threshold = 10

    def set_parameters(self, iso, shutter_speed, aperture):
        self.iso=iso
        self.iso_factor = self.iso / 100
        self.shutter_speed =shutter_speed
        self.aperture =aperture



    def forward(self,x):
        # Input light depending on aperture
        x = (math.pi * x) / 4 * self.aperture ** 2


        # Input light depending on exposure time
        x = x * self.shutter_speed

        # Add Photon Noise (Shot noise)
        x = np.random.poisson(x)

        # Apply quantum efficiency (QE) to convert photons to electrons
        x = x * self.qe

        # Apply Dark Noise
        dark_noise = np.random.normal(0, self.dark_noise_sigma, size=x.shape)
        x = x + dark_noise

        # apply iso
        x = self.iso_factor * x

        # apply system gain
        x = x/self.inverse_K

        # Clip to maximum capacity
        x = np.clip(x, 0, self.saturation_capacity)

        # Quantize to 8 bit
        x = (x / self.saturation_capacity * 255).astype(np.uint8)

        return x