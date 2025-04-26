#!/usr/bin/env python3
"""
High-quality raster-to-SVG converter (CPU / GPU aware).
Enhanced with Apple Silicon (M1/M2/M3/M4) support.
Improved compression and color fidelity.

Example:
    python image_to_svg.py input.png output.svgz --colors 24 --gpu
"""
from __future__ import annotations

# Add warning suppression right at the top
import warnings

# Suppress all numerical warnings from numpy/sklearn
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

import argparse
import gzip
import importlib.metadata as meta
import logging
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict, Any

import cv2
import numpy as np
import svgwrite
import webcolors
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans

# Detect platform and architecture
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine().startswith(('arm', 'aarch'))

# Optional GPU support ---------------------------------------------------------
_HAS_GPU = False
_HAS_MPS = False

# Try RAPIDS/CUDA first (for NVIDIA GPUs)
try:
    from cuml.cluster import KMeans as cuKMeans
    import cupy as cp

    _HAS_GPU = True
except ModuleNotFoundError:
    pass

# Try Apple Metal Performance Shaders (for M-series chips)
if IS_APPLE_SILICON and not _HAS_GPU:
    try:
        import torch

        if torch.backends.mps.is_available():
            _HAS_MPS = True
            _HAS_GPU = True
    except (ModuleNotFoundError, AttributeError):
        pass

# ------------------------------------------------------------------------------

REQUIRED_PKGS = {
    'Pillow': '10.4.0',
    'numpy': '1.26.4',
    'opencv-python': '4.8.0',
    'svgwrite': '1.4.3',
    'scikit-learn': '1.3.0',
    'webcolors': '1.13'
}

# Additional packages for Apple Silicon GPU support
APPLE_GPU_PKGS = {
    'torch': '2.1.0',  # For MPS support
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _check_deps() -> None:
    """Abort early if a required dependency is missing / too old."""
    for pkg, ver in REQUIRED_PKGS.items():
        try:
            installed_ver = meta.version(pkg.lower())
            # Changed comparison to handle the opencv-python warning specifically
            if pkg == 'opencv-python' and installed_ver >= '4.11.0':
                # Skip warning for newer opencv-python versions
                continue
            elif installed_ver < ver:
                logging.warning("%s < %s – consider upgrading", pkg, ver)
        except meta.PackageNotFoundError:
            logging.error("Missing dependency: %s", pkg)
            sys.exit(1)

    # Check Apple Silicon specific deps only if needed
    if IS_APPLE_SILICON:
        for pkg, ver in APPLE_GPU_PKGS.items():
            try:
                installed_ver = meta.version(pkg.lower())
                if installed_ver < ver:
                    logging.warning("%s < %s – consider upgrading for optimal Apple Silicon performance",
                                    pkg, ver)
            except meta.PackageNotFoundError:
                logging.warning("Optional dependency for Apple Silicon GPU: %s", pkg)


# -----------------------------------------------------------------------------


def _rgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0-255] to approximate linear-RGB [0-1]."""
    a = rgb / 255.0
    mask = a > 0.04045
    a[mask] = np.power((a[mask] + 0.055) / 1.055, 2.4)
    a[~mask] /= 12.92
    return a


def _linear_to_rgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0-1] back to sRGB [0-255]."""
    a = np.copy(linear)
    mask = a > 0.0031308
    a[mask] = 1.055 * np.power(a[mask], 1.0 / 2.4) - 0.055
    a[~mask] *= 12.92
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def _contours_to_path(
        contour: np.ndarray,
        hex_colour: str,
        opacity: float,
        simplify_eps: float,
        precision: int
) -> str:
    """Return a single SVG <path> element as string."""
    # Use Douglas-Peucker algorithm for path simplification
    approx = cv2.approxPolyDP(contour, simplify_eps, True)

    # Format with specified precision to reduce file size
    format_str = f"{{:.{precision}f}},{{:.{precision}f}}"
    pts = " ".join(format_str.format(x, y) for [[x, y]] in approx)

    # Build path attributes
    attribs = [
        f'd="M {pts} Z"',
        f'fill="{hex_colour}"'
    ]

    # Only add opacity if needed to reduce file size
    if opacity < 1.0:
        # Use fewer decimal places for opacity
        attribs.append(f'fill-opacity="{opacity:.2f}"')

    # Don't include stroke="none" as it's implied by default
    return "<path " + " ".join(attribs) + "/>"


def _process_colour_layer(
        args: Tuple[Tuple[int, int, int, int], np.ndarray, float, int, bool]
) -> List[str]:
    """Worker: build all <path> strings for a given RGBA colour."""
    colour, quant_img, simplify_eps, precision, remove_small_areas = args

    # Create mask for the current color
    mask = np.all(quant_img == colour, axis=2)
    if not np.any(mask):
        return []

    # Convert mask to bitmap and dilate
    bitmap = (mask.astype(np.uint8) * 255)
    bitmap = cv2.dilate(bitmap, np.ones((3, 3), np.uint8), iterations=1)

    # Find contours
    contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Convert color to hex
    hex_colour = webcolors.rgb_to_hex(colour[:3])
    opacity = colour[3] / 255.0 if colour[3] < 255 else 1.0

    # Filter out very small contours if requested
    if remove_small_areas:
        min_area = 5.0  # Minimum area in pixels
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Generate path for each contour
    return [
        _contours_to_path(c, hex_colour, opacity, simplify_eps, precision)
        for c in contours if len(c) > 1
    ]


class ImageToSVG:
    """Raster → SVG converter."""

    def __init__(
            self,
            src: Path,
            dst: Path,
            *,
            max_colors: int = 16,
            edge_sensitivity: float = 1.0,
            workers: int | None = None,
            use_gpu: bool = False,
            simplify: float = 1.5,
            precision: int = 1,
            enhance_colors: bool = False,
            remove_small_areas: bool = True,
            optimize: bool = True
    ):
        self.src = src
        self.dst = dst
        self.max_colors = int(np.clip(max_colors, 2, 256))
        self.edge_sigma = float(np.clip(edge_sensitivity, .5, 2.0))
        self.workers = workers or max(1, cpu_count() - 1)  # Leave one core free
        self.simplify = float(np.clip(simplify, 0.5, 5.0))
        self.precision = precision
        self.enhance_colors = enhance_colors
        self.remove_small_areas = remove_small_areas
        self.optimize = optimize

        # Determine appropriate GPU mode
        self.use_gpu = False
        self.use_cuda = False
        self.use_mps = False

        if use_gpu and _HAS_GPU:
            self.use_gpu = True
            if _HAS_MPS and IS_APPLE_SILICON:
                self.use_mps = True
                logging.info("Using Apple Metal Performance Shaders (MPS) for GPU acceleration")
            else:
                self.use_cuda = True
                logging.info("Using CUDA/RAPIDS for GPU acceleration")

        self.image: Optional[Image.Image] = None
        self.quant_img: Optional[np.ndarray] = None
        self.w = self.h = 0
        self.start_time = time.time()

    # ---------------------------------------------------------------------

    def _load(self) -> None:
        """Load and optionally enhance the input image."""
        # Load the image and convert to RGBA
        self.image = Image.open(self.src).convert("RGBA")

        # Enhance colors if requested
        if self.enhance_colors:
            enhancer = ImageEnhance.Color(self.image)
            self.image = enhancer.enhance(1.2)  # Increase color saturation by 20%

            # Also enhance contrast slightly
            enhancer = ImageEnhance.Contrast(self.image)
            self.image = enhancer.enhance(1.1)

        self.w, self.h = self.image.size
        logging.info("Loaded %s  (%dx%d)", self.src.name, self.w, self.h)

    def _quantize(self) -> None:
        """Perform color quantization with improved color fidelity."""
        arr = np.array(self.image)
        pixels = arr.reshape(-1, 4)
        rgba, alpha = pixels[:, :3], pixels[:, 3:]

        # Convert to linear RGB color space for more perceptually accurate clustering
        features = _rgb_to_linear(rgba.astype(np.float32))

        # Handle NaN and Inf values that could cause numerical issues
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        # Scale to [0,1] to prevent numerical instability
        feature_min = features.min(axis=0, keepdims=True)
        feature_max = features.max(axis=0, keepdims=True)
        feature_range = feature_max - feature_min
        # Avoid division by zero
        feature_range[feature_range == 0] = 1.0
        features = (features - feature_min) / feature_range

        # Add alpha channel with weight to influence clustering
        # This helps preserve transparency boundaries
        if np.any(alpha < 255):
            # Normalize alpha to [0,1] and add as a feature with weight
            alpha_normalized = alpha.astype(np.float32) / 255.0
            # Weight alpha less than color to prioritize color fidelity
            weighted_features = np.hstack([features, 0.5 * alpha_normalized])
        else:
            weighted_features = features

        if not self.use_gpu:
            # CPU branch - always works
            km = KMeans(n_clusters=self.max_colors,
                        random_state=42,
                        n_init='auto',
                        algorithm='elkan',  # More numerically stable
                        tol=1e-4)  # Increased tolerance
            labels = km.fit_predict(weighted_features)

            # If we added alpha as a feature, extract just the RGB part of centers
            if np.any(alpha < 255):
                centers = km.cluster_centers_[:, :3]
            else:
                centers = km.cluster_centers_

        elif self.use_cuda:
            # CUDA/RAPIDS branch
            km = cuKMeans(n_clusters=self.max_colors,
                          random_state=42,
                          init="k-means||")
            labels = km.fit_predict(cp.asarray(weighted_features)).get()

            if np.any(alpha < 255):
                centers = km.cluster_centers_.get()[:, :3]
            else:
                centers = km.cluster_centers_.get()

        elif self.use_mps:
            # Apple Metal branch using PyTorch MPS
            import torch

            # More robust k-means implementation for Apple Silicon
            device = torch.device("mps")
            tensor_features = torch.tensor(weighted_features, dtype=torch.float32, device=device)

            # We'll use k-means++ style initialization on MPS
            centroids = self._mps_kmeans_pp_init(tensor_features, self.max_colors, device)

            # Move initialized centroids back to CPU for scikit-learn
            initial_centers = centroids.cpu().numpy()

            # Run k-means on CPU with our carefully initialized centers
            km = KMeans(n_clusters=self.max_colors,
                        random_state=42,
                        n_init=1,
                        init=initial_centers,
                        algorithm='elkan',
                        tol=1e-4)

            # Move back to CPU for actual fitting
            labels = km.fit_predict(weighted_features)

            if np.any(alpha < 255):
                centers = km.cluster_centers_[:, :3]
            else:
                centers = km.cluster_centers_

            logging.info("Used robust MPS+CPU hybrid approach for k-means")

        # Convert back from normalized to original scale
        centers = centers * (feature_max - feature_min) + feature_min

        # Convert centers back to RGB colorspace using proper gamma correction
        centers_rgb = _linear_to_rgb(centers)

        # Create the quantized image
        # Match each pixel to its cluster center
        quant_colors = centers_rgb[labels]
        quant = np.hstack([quant_colors, alpha])
        self.quant_img = quant.reshape(arr.shape)

        logging.info("K-means complete (k=%d, gpu=%s, platform=%s%s)",
                     self.max_colors,
                     self.use_gpu,
                     "macOS " if IS_MACOS else "",
                     "Apple Silicon" if IS_APPLE_SILICON else "")

    def _mps_kmeans_pp_init(self, features, k, device):
        """K-means++ initialization on MPS device."""
        # First centroid is random
        rand_indices = torch.randperm(len(features), device=device)
        centroids = features[rand_indices[0]].unsqueeze(0)

        # For each remaining centroid, select based on distance
        for _ in range(1, k):
            # Calculate distance to closest centroid
            dists = torch.cdist(features, centroids)
            min_dists, _ = torch.min(dists, dim=1)

            # Square distances for k-means++ weighting
            weights = min_dists.square()

            # Handle any numerical issues
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

            # If all weights are zero, choose randomly
            if torch.sum(weights) == 0:
                next_idx = torch.randint(0, len(features), (1,), device=device)[0]
            else:
                # Sample proportional to squared distance
                probs = weights / torch.sum(weights)
                next_idx = torch.multinomial(probs, 1)[0]

            # Add the new centroid
            new_centroid = features[next_idx].unsqueeze(0)
            centroids = torch.cat([centroids, new_centroid], dim=0)

        return centroids

    # ------------------------------------------------------------------

    def _build_svg(self, paths: Iterable[str]) -> str:
        """Build the SVG document string."""
        # More compact SVG header without width/height (uses viewBox instead)
        header = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}">'
        body = "\n".join(paths)
        return f"{header}\n{body}\n</svg>"

    # ------------------------------------------------------------------

    def convert(self) -> None:
        """Convert the image to SVG."""
        self._load()
        self._quantize()

        unique_colours = np.unique(self.quant_img.reshape(-1, 4), axis=0)
        logging.info("Rendering %d layers on %d workers",
                     len(unique_colours), self.workers)

        # Process color layers in parallel with additional parameters
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            tasks = [
                (tuple(col.tolist()), self.quant_img, self.simplify, self.precision, self.remove_small_areas)
                for col in unique_colours
            ]

            futs = [ex.submit(_process_colour_layer, task) for task in tasks]

            all_paths: List[str] = []
            for f in as_completed(futs):
                all_paths.extend(p for p in f.result())

        svg_text = self._build_svg(all_paths)
        self._write_output(svg_text)

        elapsed = time.time() - self.start_time
        size_kb = self.dst.stat().st_size / 1024
        logging.info("Done → %s (%.1f kB) in %.2f seconds",
                     self.dst.name, size_kb, elapsed)

        # Report compression ratio if original file exists
        if self.src.exists():
            original_size = self.src.stat().st_size / 1024
            if original_size > 0:
                ratio = original_size / size_kb
                logging.info("Compression ratio: %.1fx (from %.1f kB)", ratio, original_size)

    # ------------------------------------------------------------------
    def _write_output(self, svg: str) -> None:
        """
        Save SVG (plain or .svgz) and optionally run scour for minification.
        Enhanced with more aggressive optimization options.
        """
        # 1  write gzip-compressed if user asked for *.svgz
        if self.dst.suffix.lower() == ".svgz":
            with gzip.open(self.dst, "wt", encoding="utf-8", compresslevel=9) as fh:
                fh.write(svg)
            return

        # 2  plain-text fallback in case scour is absent / errors out
        def _plain_write(text: str) -> None:
            self.dst.write_text(text, encoding="utf-8")

        # 3  Use scour if available and optimization is enabled
        if self.optimize:
            try:
                from scour import scour
                from inspect import signature

                # Enhanced optimization options
                if "newConfig" in signature(scour.sanitizeOptions).parameters:
                    opts = scour.sanitizeOptions(
                        options=None,
                        remove_descriptive_elements=True,
                        strip_comments=True,
                        strip_ids=True,
                        shorten_ids=True,
                        simple_colors=True,
                        strip_xml_space=True,
                        remove_metadata=True,
                        remove_unreferenced_defs=True,
                        enable_viewboxing=True,
                        strip_xml_prolog=True,
                    )
                else:  # ≤ 0.38.x
                    opts = scour.sanitizeOptions(options=None)
                    opts.remove_descriptive_elements = True
                    opts.strip_comments = True
                    opts.strip_ids = True
                    opts.shorten_ids = True
                    opts.simple_colors = True
                    opts.strip_xml_space = True
                    opts.remove_metadata = True
                    opts.remove_unreferenced_defs = True
                    opts.enable_viewboxing = True
                    opts.strip_xml_prolog = True

                rendered = scour.scourString(svg, opts)

                # Normalize to str
                if hasattr(rendered, "getvalue"):  # 1.x-dev API
                    min_svg = rendered.getvalue()
                elif isinstance(rendered, tuple):  # very old API -> (svg, errs)
                    min_svg = rendered[0]
                else:  # 0.38.x -> plain str
                    min_svg = rendered

                _plain_write(min_svg)
                logging.info("Applied SVG optimization with scour")

            except ModuleNotFoundError:
                logging.warning("scour not installed - SVG not optimized")
                _plain_write(svg)

            except Exception as exc:
                logging.warning("scour failed (%s); writing un-minified SVG", exc)
                _plain_write(svg)
        else:
            _plain_write(svg)


# -----------------------------------------------------------------------------


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raster image to compact SVG with M-series Mac support."
    )
    p.add_argument("input", type=Path, help="Input image")
    p.add_argument("output", type=Path, help="Output .svg or .svgz file")
    p.add_argument("-c", "--colors", type=int, default=16,
                   help="Maximum colours (2-256)")
    p.add_argument("-e", "--edge-sensitivity", type=float, default=1.0,
                   help="Canny edge sensitivity (0.5-2.0)")
    p.add_argument("-w", "--workers", type=int,
                   help="Parallel workers (default = CPU count - 1)")
    p.add_argument("-s", "--simplify", type=float, default=1.5,
                   help="Path simplification factor (0.5-5.0, higher = simpler paths)")
    p.add_argument("-p", "--precision", type=int, default=1,
                   help="Decimal precision for path coordinates (0-3)")
    p.add_argument("--enhance-colors", action="store_true",
                   help="Enhance color saturation in output")
    p.add_argument("--keep-small-areas", action="store_true",
                   help="Keep very small areas in the output (default is to remove them)")
    p.add_argument("--no-optimize", action="store_true",
                   help="Skip SVG optimization (faster but larger file)")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU acceleration (RAPIDS for NVIDIA, MPS for Apple Silicon)")

    # Add Apple Silicon specific options
    if IS_APPLE_SILICON:
        p.add_argument("--force-cpu", action="store_true",
                       help="Force CPU computation even on Apple Silicon")

    return p.parse_args()


def main() -> None:
    if sys.version_info < (3, 12):
        logging.error("Python ≥ 3.12 required")
        sys.exit(1)
    _check_deps()

    # Detect platform
    if IS_MACOS:
        if IS_APPLE_SILICON:
            logging.info("Detected Apple Silicon Mac (M-series)")
            if _HAS_MPS:
                logging.info("MPS acceleration available for Apple Silicon")
            else:
                logging.info("MPS acceleration not available. Install PyTorch ≥ 2.1.0 for GPU support")
        else:
            logging.info("Detected Intel Mac")

    ns = _cli()
    if not ns.input.exists():
        logging.error("Input file not found: %s", ns.input)
        sys.exit(1)

    # Handle Apple Silicon specific options
    use_gpu = ns.gpu
    if IS_APPLE_SILICON and hasattr(ns, 'force_cpu') and ns.force_cpu:
        use_gpu = False
        logging.info("Forcing CPU computation on Apple Silicon as requested")

    converter = ImageToSVG(
        src=ns.input,
        dst=ns.output,
        max_colors=ns.colors,
        edge_sensitivity=ns.edge_sensitivity,
        workers=ns.workers,
        use_gpu=use_gpu,
        simplify=ns.simplify,
        precision=ns.precision,
        enhance_colors=ns.enhance_colors,
        remove_small_areas=not ns.keep_small_areas,
        optimize=not ns.no_optimize
    )
    try:
        converter.convert()
    except Exception as exc:
        logging.exception("Conversion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()