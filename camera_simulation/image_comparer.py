import rawpy
from pathlib import Path
import camera_simulation as sim
import cv2
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_arw_file(path: Path, norm_factor=(1/65535)):
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),      # linear
            bright=1.0,
            user_flip=0,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DCB # type: ignore
        )
    normalized = rgb * norm_factor
    return normalized

def load_image_files_to_matrix(base_path: Path, isos = None, fs = None, shs = None, extension = "ARW", hw=(4024, 6024)):
    if isos is None:
        isos = ["250", "2000", "16000"]
    if fs is None:
        fs = ["5", "9", "16"]
    if shs is None:
        shs = ["1-4", "1-60", "1-1000"]


    images = np.zeros((3,3,3, hw[0], hw[1], 3))

    total = len(isos) * len(fs) * len(shs)

    for i, j, k in tqdm(
        product(range(len(isos)), range(len(fs)), range(len(shs))),
        total=total,
        desc="Load Files to Matrix",
    ):
        iso = isos[i]
        f = fs[j]
        sh = shs[k]
        path = base_path / f"ISO{iso}_F{f}_sh{sh}.{extension}"
        if extension == "ARW":
            img = load_arw_file(path)
        else:
            img = plt.imread(path)
        images[i][j][k] = img

    return images
    
def simulate_images(hdr_path: Path, depth_path: Path | None = None, cam: sim.CameraSimulation | None = None, isos = None, fs = None, shs = None, input_factor = 5000):
    if isos is None:
        isos = sim.iso_values
    if fs is None:
        fs = sim.aperture_values
    if shs is None:
        shs = sim.shutter_speed_values

    if cam is None:
        cam = sim.CameraSimulation(log=False)

    input_img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)#*input_factor
    if input_img is None:
        raise ValueError(f"Could not load input image from {hdr_path}")
    
    input_img = input_img * input_factor 
    input_depth = None #cv2.imread(depth_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)

    images = np.zeros((3,3,3, 1440, 2560, 3), dtype=np.float32)


    total = len(isos) * len(fs) * len(shs)

    for i, j, k in tqdm(
        product(range(len(isos)), range(len(fs)), range(len(shs))),
        total=total, 
        desc="Simulate Images",
    ):
        iso = isos[i]
        f = fs[j]
        sh = shs[k]
        cam.set_parameters(iso=iso, shutter_speed=sh, aperture=f)
        images[i][j][k] = cam.simulate_image(input_img, input_depth)

    return images

def get_normalized_histogramm_of_matrix(matrix, val_range = [0.0, 1.0]):
    if matrix.dtype is np.float64:
        matrix = matrix.astype(np.float32)
    results = np.zeros((3,3,3, 256, 1), dtype=np.float32)
    for i, j, k in tqdm(product(range(3), range(3), range(3)), total=27, desc="Get Histogramms"):
        img = matrix[i][j][k]

        shape = img.shape
        val_amount = np.prod(shape)
        sim_hist = cv2.calcHist([img], [0], None, [256], val_range)/val_amount
        results[i][j][k] = sim_hist
    return results

def compare_hist_matrix(sim_hists, cam_hists, method = cv2.HISTCMP_BHATTACHARYYA):
    if not sim_hists.shape[0:2] == cam_hists.shape[0:2]:
        return None
    results = np.zeros((3,3,3, 1), dtype=np.float32)
    for i, j, k in tqdm(product(range(3), range(3), range(3)), total=27, desc="Comparing"):
        sim_hist = sim_hists[i][j][k]
        cam_hist = cam_hists[i][j][k]

        metric = cv2.compareHist(sim_hist, cam_hist, method)
        
        results[i][j][k] = metric

    return results
    

def compare_hist_matrix_from_path(sim_path, cam_path, method=cv2.HISTCMP_BHATTACHARYYA):
    print(cam_path)
    sim_hists = np.load(str(sim_path))
    cam_hists = np.load(str(cam_path))
    if not sim_hists.shape[0:2] == cam_hists.shape[0:2]:
        print("Shapes dont match")
        return None
    return compare_hist_matrix(sim_hists, cam_hists, method=method)