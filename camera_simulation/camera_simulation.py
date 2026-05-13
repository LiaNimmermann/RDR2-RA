"""
camera_simulation.py
====================
Simulate physical camera settings from HDR linear scene data.

Inputs:
  - HDR RGB  : JPEG XR (.jxr) file containing linear scene radiance
  - Depth    : OpenEXR (.exr) file with a single-channel depth map (metres)

Simulated effects (physically motivated):
  - Exposure  : ISO × shutter speed → scene luminance scaling
  - DoF blur  : Aperture (f-number) + focus distance → lens blur via depth map
  - ISO noise : Photon shot noise + read noise (Poisson + Gaussian model)
  - Motion blur: Shutter speed → directional motion blur

Output: 8-bit sRGB PNG

Dependencies:
    pip install openexr numpy scipy scikit-image opencv-python pillow
    sudo apt install libjxr-tools   # for JxrDecApp

Usage example:
    python camera_simulation.py \\
        --hdr     scene.jxr \\
        --depth   depth.exr \\
        --iso     800 \\
        --shutter 1/60 \\
        --aperture 2.8 \\
        --focus   3.5 \\
        --output  result.png
"""

import argparse
import os
import subprocess
import tempfile
import sys

import cv2
import numpy as np
import OpenEXR
import Imath
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise


# ─────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────

def load_jxr_as_linear_float(jxr_path: str) -> np.ndarray:
    """
    Decode a JPEG XR file to a 32-bit float RGB array [H, W, 3] in linear light.
    Uses JxrDecApp (libjxr-tools) to export 128bppRGBFloat into a TIFF,
    then reads it with OpenCV.
    """
    if not os.path.isfile(jxr_path):
        raise FileNotFoundError(f"JXR file not found: {jxr_path}")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["JxrDecApp", "-i", jxr_path, "-o", tmp_path, "-c", "15"],
            capture_output=True, text=True
        )
        if result.returncode not in (0, 151):   # 151 = "success" for this build
            raise RuntimeError(
                f"JxrDecApp failed (code {result.returncode}):\n{result.stderr}"
            )

        img = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR |
                         cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise RuntimeError(f"cv2 could not read decoded TIFF: {tmp_path}")

        # OpenCV loads as BGR; convert to RGB
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_exr_depth(exr_path: str) -> np.ndarray:
    """
    Read a single-channel EXR depth map (metres) → float32 [H, W].
    Tries channels named: Z, depth, R, Y (in that order).
    """
    if not os.path.isfile(exr_path):
        raise FileNotFoundError(f"EXR file not found: {exr_path}")

    exr = OpenEXR.InputFile(exr_path)
    header = exr.header()
    dw = header["dataWindow"]
    width  = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channel_names = list(header["channels"].keys())
    priority = ["Z", "depth", "R", "Y"]
    chosen = next((c for c in priority if c in channel_names), channel_names[0])

    raw = exr.channel(chosen, Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    return depth


def save_png(rgb_uint8: np.ndarray, path: str) -> None:
    out_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out_bgr)
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────
#  Camera model
# ─────────────────────────────────────────────────────────────

def compute_exposure_scale(iso: float, shutter: float,
                            aperture: float) -> float:
    """
    Compute a linear scene scale from camera settings using the Exposure Value
    (EV) model.

      EV = log2(N² / t)   where N = f-number, t = shutter time (seconds)
      Relative gain vs reference camera (ISO 100, 1/125 s, f/8):
        gain = (ISO / ISO_ref) × (t / t_ref) × (N_ref² / N²)

    Returns a multiplicative scale applied to linear scene radiance before
    tone-mapping.
    """
    ISO_REF   = 100.0
    T_REF     = 1.0 / 125.0
    N_REF     = 8.0

    gain = (iso / ISO_REF) * (shutter / T_REF) * (N_REF ** 2 / aperture ** 2)
    return float(gain)


def apply_exposure(hdr: np.ndarray, scale: float) -> np.ndarray:
    """Scale linear HDR radiance by the camera's exposure gain."""
    return hdr * scale


# ─── Depth-of-Field ───────────────────────────────────────────

def compute_coc_diameter(depth: np.ndarray, focus_dist: float,
                          aperture: float, focal_length_mm: float = 50.0,
                          sensor_width_mm: float = 36.0,
                          image_width_px: int = None) -> np.ndarray:
    """
    Circle of Confusion (CoC) diameter in pixels.

    Formula:
        CoC_mm = |f² × (d - dF)| / (N × dF × (d - f))
    where
        f  = focal length (mm → m)
        N  = f-number
        d  = scene depth (m)
        dF = focus distance (m)

    Then convert mm → pixels via sensor size / image width.
    """
    f_m  = focal_length_mm * 1e-3
    d    = np.clip(depth, 1e-3, None)          # avoid zero depth
    dF   = max(focus_dist, f_m * 1.001)        # focus behind front focal plane

    numerator   = f_m ** 2 * np.abs(d - dF)
    denominator = aperture * dF * np.maximum(np.abs(d - f_m), 1e-6)
    coc_mm      = numerator / denominator

    # mm → pixels
    if image_width_px is None:
        image_width_px = depth.shape[1]
    mm_per_px = sensor_width_mm / image_width_px
    coc_px    = coc_mm / mm_per_px

    return coc_px.astype(np.float32)


def apply_dof_blur(image: np.ndarray, depth: np.ndarray,
                   aperture: float, focus_dist: float,
                   focal_length_mm: float = 50.0,
                   max_blur_radius_px: float = 30.0,
                   num_layers: int = 8) -> np.ndarray:
    """
    Layered Gaussian DoF approximation.

    1. Compute per-pixel CoC diameter.
    2. Partition the image into depth layers.
    3. Blur each layer by the median CoC of that layer.
    4. Composite back-to-front (painter's algorithm).

    This avoids bleeding sharp foreground edges into blurred backgrounds.
    """
    H, W = image.shape[:2]
    coc  = compute_coc_diameter(depth, focus_dist, aperture,
                                focal_length_mm=focal_length_mm,
                                image_width_px=W)
    coc  = np.clip(coc / 2.0, 0, max_blur_radius_px)  # radius

    # Sort layers back-to-front (far → near)
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    if len(valid_depth) == 0:
        return image.copy()

    d_min, d_max = valid_depth.min(), valid_depth.max()
    edges = np.linspace(d_min, d_max + 1e-6, num_layers + 1)

    result = np.zeros_like(image)
    weight = np.zeros((H, W), dtype=np.float32)

    # Process from far to near so near objects win
    for i in range(num_layers - 1, -1, -1):
        mask = (depth >= edges[i]) & (depth < edges[i + 1])
        if not mask.any():
            continue

        layer_coc = coc[mask]
        blur_r    = float(np.percentile(layer_coc, 75))  # robust estimate

        # Extract layer pixels; fill non-layer pixels with image values
        layer_img = image.copy()

        if blur_r > 0.5:
            sigma = blur_r / 2.0
            blurred = np.stack([
                gaussian_filter(layer_img[..., c].astype(np.float64), sigma)
                for c in range(3)
            ], axis=-1).astype(np.float32)
        else:
            blurred = layer_img

        # Only contribute pixels belonging to this layer
        layer_mask_3d = np.stack([mask] * 3, axis=-1)
        # Contribution: replace where this layer's weight is higher
        new_weight = mask.astype(np.float32)
        update = new_weight > weight
        result[update] = blurred[update]
        weight = np.maximum(weight, new_weight)

    # Fill any unweighted pixels with the unblurred image
    unfilled = weight < 0.5
    result[unfilled] = image[unfilled]

    return np.clip(result, 0, None)


# ─── ISO Noise ───────────────────────────────────────────────

def apply_iso_noise(image: np.ndarray, iso: float,
                    base_iso: float = 100.0) -> np.ndarray:
    """
    Physically-motivated sensor noise model:

    Total noise = photon shot noise (Poisson) + read noise (Gaussian).

    At high ISO the read noise floor is amplified multiplicatively.
    The image is expected to be in [0, 1] normalised linear light.

    Shot noise variance ∝ signal magnitude (Poisson statistics).
    Read noise std dev ∝ ISO / base_ISO.
    """
    img = np.clip(image, 0.0, 1.0).astype(np.float64)

    # Shot noise: Poisson with intensity proportional to signal
    # skimage random_noise handles Poisson in normalised space
    shot = random_noise(img, mode="poisson", clip=False)

    # Read noise: Gaussian, scales with ISO gain
    read_noise_std = 0.002 * (iso / base_iso)
    read = np.random.normal(0.0, read_noise_std, img.shape)

    # Fixed pattern noise (PRNU / column noise) – subtle banding
    col_noise = np.random.normal(0.0, read_noise_std * 0.3,
                                 (1, img.shape[1], img.shape[2]))
    col_noise = np.broadcast_to(col_noise, img.shape)

    noisy = shot + read + col_noise
    return np.clip(noisy, 0.0, None).astype(np.float32)


# ─── Motion Blur ─────────────────────────────────────────────

def apply_motion_blur(image: np.ndarray, shutter: float,
                      motion_fps: float = 24.0,
                      motion_pixels_per_frame: float = 8.0,
                      angle_deg: float = 0.0) -> np.ndarray:
    """
    Simulate camera/subject motion blur.

    The blur length in pixels scales with shutter time:
        blur_px = motion_pixels_per_frame × (shutter × motion_fps)

    A rotated 1-D box kernel (motion streak) is convolved with the image.
    Angle 0° = horizontal motion.
    """
    blur_px = motion_pixels_per_frame * (shutter * motion_fps)
    blur_px = int(round(blur_px))

    if blur_px < 2:
        return image.copy()

    # Build a 1-D motion kernel rotated to the given angle
    ksize = blur_px if blur_px % 2 == 1 else blur_px + 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize

    # Rotate kernel
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= kernel.sum() + 1e-12

    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return blurred.astype(np.float32)


# ─── Tone Mapping + sRGB conversion ──────────────────────────

def tonemap_reinhard(hdr: np.ndarray, white: float = None) -> np.ndarray:
    """
    Extended Reinhard global tone mapping.
        L_d = L × (1 + L / L_white²) / (1 + L)
    """
    lum = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
    if white is None:
        white = float(np.percentile(lum[lum > 0], 99)) if lum.any() else 1.0
    white = max(white, 1e-6)

    scale = (1.0 + lum / (white ** 2)) / (1.0 + lum + 1e-10)
    tonemapped = hdr * scale[..., np.newaxis]
    return np.clip(tonemapped, 0.0, 1.0).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Apply the sRGB gamma / electro-optical transfer function."""
    lin = np.clip(linear, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308,
                    12.92 * lin,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────

def simulate_camera(
    hdr_path: str,
    depth_path: str,
    iso: float,
    shutter: float,
    aperture: float,
    focus_dist: float,
    focal_length_mm: float = 50.0,
    motion_angle_deg: float = 0.0,
    motion_pixels_per_frame: float = 8.0,
    output_path: str = "output.png",
    tonemap: str = "reinhard",
    verbose: bool = True,
) -> np.ndarray:
    """
    Full camera simulation pipeline.

    Parameters
    ----------
    hdr_path   : Path to the JPEG XR HDR scene file.
    depth_path : Path to the EXR depth map (metres).
    iso        : Sensor ISO (e.g. 100, 400, 1600, 6400).
    shutter    : Shutter time in seconds (e.g. 1/60 → 0.01667).
    aperture   : f-number (e.g. 1.4, 2.8, 8.0, 16.0).
    focus_dist : Focus distance in metres.
    focal_length_mm : Simulated lens focal length in mm (default 50 mm).
    motion_angle_deg: Direction of motion blur in degrees (0 = horizontal).
    motion_pixels_per_frame : Motion speed in px/frame at 24 fps reference.
    output_path: Where to save the final 8-bit sRGB PNG.
    tonemap    : Tone mapping operator — "reinhard" (default).
    verbose    : Print progress.

    Returns
    -------
    uint8 sRGB numpy array [H, W, 3].
    """

    def log(msg):
        if verbose:
            print(f"[camera_sim] {msg}")

    # ── 1. Load inputs ─────────────────────────────────────────
    log(f"Loading HDR: {hdr_path}")
    hdr = load_jxr_as_linear_float(hdr_path)
    log(f"  shape={hdr.shape}, dtype={hdr.dtype}, "
        f"range=[{hdr.min():.4f}, {hdr.max():.4f}]")

    log(f"Loading depth: {depth_path}")
    depth = load_exr_depth(depth_path)
    log(f"  shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}] m")

    # Resize depth to match HDR if needed
    if depth.shape[:2] != hdr.shape[:2]:
        log(f"  Resizing depth {depth.shape[:2]} → {hdr.shape[:2]}")
        depth = cv2.resize(depth, (hdr.shape[1], hdr.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    # ── 2. Exposure scaling ────────────────────────────────────
    exp_scale = compute_exposure_scale(iso, shutter, aperture)
    log(f"Exposure scale: {exp_scale:.4f}  "
        f"(ISO={iso}, t={shutter:.5f}s, f/{aperture})")
    hdr_exposed = apply_exposure(hdr, exp_scale)

    # ── 3. Motion blur (before tone-map, on linear data) ───────
    motion_blur_px = motion_pixels_per_frame * (shutter * 24.0)
    if motion_blur_px >= 2:
        log(f"Motion blur: {motion_blur_px:.1f} px @ {motion_angle_deg}°")
        hdr_exposed = apply_motion_blur(
            hdr_exposed, shutter,
            motion_pixels_per_frame=motion_pixels_per_frame,
            angle_deg=motion_angle_deg
        )
    else:
        log("Motion blur: negligible (< 2 px)")

    # ── 4. Depth-of-field blur ─────────────────────────────────
    log(f"DoF blur: f/{aperture}, focus={focus_dist:.2f}m, "
        f"fl={focal_length_mm}mm")
    hdr_dof = apply_dof_blur(
        hdr_exposed, depth,
        aperture=aperture,
        focus_dist=focus_dist,
        focal_length_mm=focal_length_mm,
    )

    # ── 5. Tone-map to [0, 1] ──────────────────────────────────
    log(f"Tone mapping: {tonemap}")
    if tonemap == "reinhard":
        ldr = tonemap_reinhard(hdr_dof)
    else:
        ldr = np.clip(hdr_dof, 0.0, 1.0).astype(np.float32)

    # ── 6. ISO noise (applied in [0,1] domain) ─────────────────
    log(f"ISO noise: ISO={iso}")
    ldr_noisy = apply_iso_noise(ldr, iso)

    # ── 7. sRGB gamma + quantise ───────────────────────────────
    srgb  = linear_to_srgb(ldr_noisy)
    final = to_uint8(srgb)

    # ── 8. Save ────────────────────────────────────────────────
    save_png(final, output_path)

    return final


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_fraction(s: str) -> float:
    """Accept '1/60', '0.01667', '1/250', etc."""
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate camera settings from HDR JXR + EXR depth data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hdr",      required=True, help="Path to HDR .jxr file")
    p.add_argument("--depth",    required=True, help="Path to depth .exr file")
    p.add_argument("--iso",      type=float,  default=400,    help="ISO value")
    p.add_argument("--shutter",  type=parse_fraction, default="1/60",
                   help="Shutter speed, e.g. '1/60' or '0.0167'")
    p.add_argument("--aperture", type=float,  default=2.8,
                   help="Aperture f-number (lower = more blur)")
    p.add_argument("--focus",    type=float,  default=3.0,
                   help="Focus distance in metres")
    p.add_argument("--focal-length", type=float, default=50.0,
                   help="Lens focal length in mm")
    p.add_argument("--motion-angle", type=float, default=0.0,
                   help="Motion blur direction in degrees (0=horizontal)")
    p.add_argument("--motion-speed", type=float, default=8.0,
                   help="Motion speed in px/frame at 24 fps")
    p.add_argument("--output",   default="output.png", help="Output PNG path")
    p.add_argument("--tonemap",  default="reinhard",
                   choices=["reinhard"], help="Tone mapping operator")
    p.add_argument("--quiet",    action="store_true", help="Suppress log output")
    return p


def main():
    args = build_parser().parse_args()
    simulate_camera(
        hdr_path=args.hdr,
        depth_path=args.depth,
        iso=args.iso,
        shutter=args.shutter,
        aperture=args.aperture,
        focus_dist=args.focus,
        focal_length_mm=args.focal_length,
        motion_angle_deg=args.motion_angle,
        motion_pixels_per_frame=args.motion_speed,
        output_path=args.output,
        tonemap=args.tonemap,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()