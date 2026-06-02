import numpy as np
import cv2

class CameraSimulation:
    def __init__(self, iso=100, shutter_speed=1/60, aperture=2.8):
        self.iso = iso
        self.shutter_speed = shutter_speed
        self.aperture = aperture
        self.iso_factor = self.iso / 100


    def radiance_to_electrons(self, radiance):
        # Convert radiance to electrons based on shutter speed
        sensitivity = 1
        exposure_time = self.shutter_speed
        
        electrons = radiance * sensitivity * exposure_time
        return electrons
    
    def shot_noise(self, electrons):
        # Simulate shot noise using Poisson distribution
        noisy_electrons = np.random.poisson(electrons)
        return noisy_electrons
    
    def apply_iso(self, electrons):
        # Adjust electrons based on ISO
        adjusted_electrons = electrons * self.iso_factor
        return adjusted_electrons
    
    def add_read_noise(self, electrons):
        # Simulate read noise using Gaussian distribution
        read_noise = np.random.normal(0, np.sqrt(self.iso_factor), size=electrons.shape)  # Assuming a read noise of 5 electrons
        noisy_electrons = electrons + read_noise
        return noisy_electrons
    
    def clip_electrons(self, electrons):
        # Clip electrons to the maximum capacity of the sensor
        max_electrons = 10000  # Example maximum capacity
        clipped_electrons = np.clip(electrons, 0, max_electrons)
        return clipped_electrons
    
    def quantize_to_8bit(self, electrons):
        # Quantize electrons to 8-bit values
        max_electrons = 10000  # Example maximum capacity
        quantized_image = (electrons / max_electrons * 255).astype(np.uint8)
        return quantized_image

    def simulate_image(self, image, depth_map):
        # Simulate exposure based on ISO, shutter speed, and aperture
        electrons = self.radiance_to_electrons(image)
        electrons = self.shot_noise(electrons)
        electrons = self.apply_iso(electrons)
        electrons = self.add_read_noise(electrons)
        electrons = self.clip_electrons(electrons)
        electrons = self.quantize_to_8bit(electrons)
        return electrons