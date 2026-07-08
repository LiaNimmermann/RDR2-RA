import rawpy
from pathlib import Path
import camera_simulation as sim
import cv2

def load_arw_file(path: Path, norm_factor=(1/65535)):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),      # linear
            bright=1.0,
            user_flip=0,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
        )
    normalized = rgb * norm_factor
    return normalized

def load_image_files_to_matrix(base_path: Path, isos = None, fs = None, shs = None):
    if isos == None:
        isos = ["250", "2000", "16000"]
    if fs == None:
        fs = ["5", "9", "16"]
    if shs == None:
        shs = ["1-4", "1-60", "1-1000"]

    images = [[[]]]

    for i, iso in enumerate(isos):
        for j, f in enumerate(fs):
            for k, sh in enumerate(shs):
                path = base_path / f"ISO{iso}_F{f}_sh{sh}.ARW"
                images[i][j][k] = load_arw_file(path)

    return images
    
def simulate_images(hdr_path: Path, depth_path: Path = None, cam: sim.CameraSimulation = None, isos = None, fs = None, shs = None, input_factor = 5000):
    if isos == None:
        isos = sim.iso_values
    if fs == None:
        fs = sim.aperture_values
    if shs == None:
        shs = sim.shutter_speed_values

    input_img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)*input_factor
    input_depth = None #cv2.imread(depth_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)

    images = [[[]]]

    for i, iso in enumerate(isos):
        for j, f in enumerate(fs):
            for k, sh in enumerate(shs):
                cam.set_parameters(iso=iso, shutter_speed=sh, aperture=f)
                images[i][j][k] = cam.simulate_image(input_img, input_depth)

    return images


def compare_images_matrix(sim_imgs, cam_imgs):
    if not sim_imgs.shape[0:2] == cam_imgs.shape[0:2]:
        return None
    results = [[[]]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                metric = 0 # TODO
                results[i][j][k] = metric

    return results
    