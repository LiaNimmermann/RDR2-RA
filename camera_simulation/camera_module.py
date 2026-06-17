import torch.nn as nn

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

        