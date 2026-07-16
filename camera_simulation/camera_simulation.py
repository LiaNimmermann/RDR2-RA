import math
import numpy as np

aperture_values = [5.0, 9.0, 16.0]
shutter_speed_values = [1/4, 1/60, 1/1000]
shutter_speed_values_str = ["1/4", "1/60", "1/1000"]
iso_values = [250, 2000, 16000]

class CameraSimulation:
    # ISO 250 / 2000 / 16000
    # Shutter Speed (1/4) / (1/60) / (1/1000)
    # Aperture f5 / f9 / f16

    # Camera Parameters from Basler acA1920-155um

    def __init__(self, iso=2000, shutter_speed=1/60, aperture=9, log=True, camera_type="alpha6000", ):
        self.iso = iso
        self.shutter_speed = shutter_speed
        self.aperture = aperture
        self.iso_factor = self.iso / 100
        self.log = log

        if camera_type == "alpha6000":
            self.qe = 1.0#0.7  # Quantum Efficiency
            self.inverse_K = 0.425**-1  # Inverse of the camera's gain factor
            self.dark_noise_sigma = 2.43
            self.saturation_capacity = 9091 
            self.abs_sensitivity_threshold = 10
        else:
            self.qe = 0.7  # Quantum Efficiency
            self.inverse_K = 8.4  # Inverse of the camera's gain factor
            self.dark_noise_sigma = 6.8
            self.saturation_capacity = 32700
            self.abs_sensitivity_threshold = 10

    def set_iso(self, iso):
        self.iso = iso
        self.iso_factor = self.iso / 100

    def set_shutter_speed(self, shutter_speed):
        self.shutter_speed = shutter_speed
    
    def set_aperture(self, aperture):
        self.aperture = aperture

    def set_log(self, log):
        self.log = log

    def set_parameters(self, iso, shutter_speed, aperture):
        self.set_iso(iso)
        self.set_shutter_speed(shutter_speed)
        self.set_aperture(aperture)

    def get_parameters(self):
        return {
            "iso": self.iso,
            "shutter_speed": self.shutter_speed,
            "aperture": self.aperture
        }

    def log_image_stats(self, image):
        if self.log:
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Pixel value range: {image.min()} to {image.max()}")
            print(f"Mean pixel value: {image.mean()}")
            print(f"Standard deviation of pixel values: {image.std()}")

    def input_to_illuminance(self, image):
        min = np.min(image)
        if min<0:
            image -= min
        illuminance = (math.pi * image) / 4 * self.aperture ** 2
        return illuminance

    def illuminance_to_photons_with_shot_noise(self, illuminance):
        # Convert illuminance to photons based on shutter speed
        exposure_time = self.shutter_speed
        # Add Photon noise using Poisson distribution
        photons = np.random.poisson(illuminance * exposure_time)
        return photons
    
    def photons_to_electrons(self, photons):
        # Apply quantum efficiency (QE) to convert photons to electrons
        quantum_efficiency = 0.7  # Example QE value
        electrons = photons * quantum_efficiency
        return electrons
    
    def add_dark_noise(self, x):
        # Simulate dark noise using Gaussian distribution with dark_noise_sigma
        read_noise = np.random.normal(0, np.sqrt(self.dark_noise_sigma), size=x.shape)  # Assuming a read noise of 5 electrons
        noisy_electrons = x + read_noise
        return noisy_electrons
    
    def apply_iso(self, x):
        # Adjust electrons based on ISO
        adjusted_electrons = x * self.iso_factor
        return adjusted_electrons

    def apply_system_gain(self, x):
        return x / self.inverse_K

    def clip_electrons(self, x):
        # Clip electrons to the maximum capacity of the sensor
        max_electrons = 10000  # Example maximum capacity
        clipped_electrons = np.clip(x, 0, max_electrons)
        return clipped_electrons
    
    def quantize_to_8bit(self, x):
        # Quantize electrons to 8-bit values
        max_electrons = 10000  # Example maximum capacity
        quantized_image = (x / max_electrons * 255).astype(np.uint8)
        return quantized_image

    def simulate_image(self, image, depth_map):
        # Simulate exposure based on ISO, shutter speed, and aperture
        #print("*"*15 + " Input Stats " + "*"*15)
        self.log_image_stats(image)

        illuminance = self.input_to_illuminance(image)
        self.log_image_stats(illuminance)
        #print("*"*15 + " Illuminance Stats " + "*"*15)
        #self.log_image_stats(illuminance)
        photons = self.illuminance_to_photons_with_shot_noise(illuminance)
       # print("*"*15 + " Photon Stats " + "*"*15)
        #self.log_image_stats(photons)
        electrons = self.photons_to_electrons(photons)
        #print("*"*15 + " Electron Stats " + "*"*15)
        self.log_image_stats(electrons)
        x = self.add_dark_noise(electrons)
        x = self.apply_system_gain(x)
        x = self.apply_iso(x)
        x = self.clip_electrons(x)
        x = self.quantize_to_8bit(x)
       # print("*"*15 + " Final Stats " + "*"*15)
        self.log_image_stats(x)
        return x