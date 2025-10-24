"""
File Loading and Mask Processing Function
=======================================

This function loads nd2 files and their corresponding probability masks from the Data folder structure.

Directory Structure:
-----------------
base_path/
└── Data/
    ├── [nd2 files]
    └── Processed/
        ├── C3/
        │   └── [phase masks]
        ├── C1/
        │   └── [c1 masks]
        └── C2/
            └── [c2 masks]

File Naming Convention:
--------------------
- ND2 files: example.nd2
- Phase masks: example_C3_Probabilities.tif
- C2 masks: example_C1_Probabilities.tif
- F3 masks: example_C2_Probabilities.tif

Returns:
-------
The function returns a tuple containing:
1. nd2_list: List of loaded nd2 image data
2. mask_f1: List of C1 masks
3. mask_f2: List of C2 masks
4. labeled_mask_phase: List of processed phase masks
5. position_list: List of position information
6. marker_list: List of marker colors
7. physical_size: Tuple of physical dimensions
8. delta_t: Tuple of time information

To unpack the results:
(nd2_list, mask_f1, mask_f2, labeled_mask_phase,
 position_list, marker_list, physical_size, delta_t) = results
"""

from aicsimageio import AICSImage
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join

import cv2
from skimage import (
    morphology, 
    measure, 
    exposure
)
import scipy.ndimage as ndi
from scipy import stats
import math
from typing import Union, List, Tuple, Optional
from scipy.ndimage import label
from scipy.spatial import cKDTree
import pandas as pd

def load_tif(file_path):
    """Load a tif image."""
    import tifffile as tiff
    try:
        image = tiff.imread(file_path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def stretch_contrast(image, method='mean2x', target_max=None, percentile=None, input_format='TCZYX'):
    """
    Enhance contrast for multi-dimensional images using exposure.rescale_intensity.
    The contrast is stretched between min and target_max for each YX plane.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Method to calculate target_max if not provided:
            - 'mean2x': uses 2 * mean (default)
            - 'max': uses maximum value of each plane
            - 'percentile': uses specified percentile value
        target_max (float, optional): Fixed maximum intensity to stretch to. Overrides method if provided
        percentile (float, optional): Percentile value (0-100) to use if method='percentile'
        input_format (str): Image dimension order (e.g., 'TCZYX', 'TCYX', 'CYX', 'YX')
    
    Returns:
        numpy.ndarray: Contrast-enhanced image with the same shape and dtype
    """
    # Validate input format
    valid_dims = set('TCZYX')
    if not set(input_format).issubset(valid_dims):
        raise ValueError(f"Invalid input format. Must use characters from {valid_dims}.")
    
    # Validate method
    valid_methods = {'mean2x', 'max', 'percentile'}
    if method not in valid_methods:
        raise ValueError(f"Invalid method. Must be one of {valid_methods}")
    
    # Validate percentile if method is 'percentile'
    if method == 'percentile':
        if percentile is None:
            raise ValueError("Percentile value must be provided when using 'percentile' method")
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
    
    # Ensure input dimensions match the format
    if len(input_format) != image.ndim:
        raise ValueError(f"Input image dimensions ({image.ndim}) do not match the format ({len(input_format)}).")
    
    # Determine axes for YX and other dimensions
    yx_axes = [input_format.index(dim) for dim in 'YX']
    other_axes = [i for i in range(len(input_format)) if i not in yx_axes]
    
    # Transpose image to move YX to the last two dimensions
    transpose_order = other_axes + yx_axes
    transposed_image = np.transpose(image, transpose_order)
    
    # Reshape to isolate YX planes
    reshaped_image = transposed_image.reshape(-1, transposed_image.shape[-2], transposed_image.shape[-1])
    
    # Process each 2D plane
    enhanced_planes = []
    for plane in reshaped_image:
        if target_max is not None:
            # Use fixed target_max if provided
            plane_target_max = target_max
        else:
            # Calculate target_max based on method
            if method == 'mean2x':
                plane_target_max = plane.mean() * 2
            elif method == 'max':
                plane_target_max = plane.max()
            elif method == 'percentile':
                plane_target_max = np.percentile(plane, percentile)
                
        # Stretch contrast
        enhanced_plane = exposure.rescale_intensity(plane, in_range=(plane.min(), plane_target_max))
        enhanced_planes.append(enhanced_plane)
    
    enhanced_planes = np.array(enhanced_planes)
    
    # Restore the original shape
    restored_shape = transposed_image.shape[:-2] + enhanced_planes.shape[-2:]
    enhanced_image = enhanced_planes.reshape(restored_shape)
    
    # Reverse transpose to return to the original dimension order
    reverse_order = np.argsort(transpose_order)
    final_image = np.transpose(enhanced_image, reverse_order)
    
    return final_image
    
# Function to compute mean intensity per cell
def calculate_mean_intensity(labeled_mask, intensity_image):
    cell_intensity_means = []

    # Loop through each unique label in the labeled mask
    for cell_label in np.unique(labeled_mask):
        if cell_label == 0:  # Skip the background
            continue

        # Create a mask for the current cell
        cell_mask = (labeled_mask == cell_label)

        # Calculate the mean intensity for the current cell
        mean_intensity = np.mean(intensity_image[cell_mask])

        # Store the result as (label, mean_intensity)
        cell_intensity_means.append((cell_label, mean_intensity))

    return cell_intensity_means

def count_maxima_per_label(maxima_coords, labeled_mask, maximum_distance=0):
    '''
    Count the number of maxima points found in each labeled region of a mask,
    assigning each maximum to the closest region within the specified distance.

    Inputs:
    maxima_coords - numpy array of shape (N, 2) containing (row, col) coordinates of maxima
    labeled_mask - 2D numpy array where each unique positive integer represents a different region
    maximum_distance - int, maximum distance in pixels from a region to consider a maximum point
                      (default: 0, only count points inside the region)

    Output:
    Returns numpy array of shape (M, 2) where each row contains [label_number, maxima_count]
    sorted by label number. Only labels that exist in the mask are included.
    '''
    # Get unique labels from the mask (excluding background label 0)
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background

    if len(maxima_coords) == 0:
        # Return zero counts for all labels if no maxima
        return np.array([[label, 0] for label in unique_labels])

    # Filter out any NaN or infinite values from maxima_coords
    valid_maxima = maxima_coords[np.all(np.isfinite(maxima_coords), axis=1)]
    
    if len(valid_maxima) == 0:
        # Return zero counts if no valid maxima remain
        return np.array([[label, 0] for label in unique_labels])

    # Initialize results dictionary
    label_counts = {label: 0 for label in unique_labels}
    
    # Get coordinates for all labels at once
    label_coords = {
        label: np.argwhere(labeled_mask == label)
        for label in unique_labels
    }
    
    # Create KDTrees for each label's coordinates
    trees = {
        label: cKDTree(coords)
        for label, coords in label_coords.items()
    }
    
    # Process each maximum
    for max_coord in valid_maxima:
        min_dist = np.inf
        closest_label = None
        
        # Find the closest label to this maximum
        for label, tree in trees.items():
            try:
                dist, _ = tree.query(max_coord, k=1)
                if dist < min_dist and (maximum_distance == 0 or dist <= maximum_distance):
                    min_dist = dist
                    closest_label = label
            except ValueError:
                continue  # Skip if there's an issue with this coordinate
        
        # Increment count for the closest label if one was found
        if closest_label is not None:
            label_counts[closest_label] += 1
    
    # Convert results to array format
    results = [[label, label_counts[label]] for label in unique_labels]
    
    return np.array(results)

# Companion MIP function
def mip(nd_array, input_format='TCZYX'):
    '''
    Maximum Intensity Projection along the z-Axis with flexible input format
    
    Parameters:
    -----------
    nd_array : numpy.ndarray
        Input array to project
    input_format : str, optional
        Order of dimensions (default 'TCZYX')
        Supports formats: TCZYX, CZYX, ZYX, etc.
    
    Returns:
    --------
    numpy.ndarray
        Projected array with z-dimension set to 1
    '''
    # Determine z-axis based on input format
    z_axis = input_format.index('Z')
    
    # Perform max projection along z-axis
    projected = np.max(nd_array, axis=z_axis)
    
    # If input format doesn't already match TCZYX, we might need to add z dimension
    if len(projected.shape) < len(nd_array.shape):
        # Insert z-dimension with size 1 at the original z position
        projected = np.expand_dims(projected, axis=z_axis)
    
    return projected

def normalize(img_array: np.ndarray, 
              target_min: float = 0, 
              target_max: float = 1, 
              dtype: Optional[int] = None) -> np.ndarray:
    """
    Normalize image array to a specified range.

    Parameters
    ----------
    img_array : numpy.ndarray
        Input image array to be normalized
    target_min : float, optional
        Minimum value of the output range (default: 0)
    target_max : float, optional
        Maximum value of the output range (default: 1)
    dtype : int, optional
        OpenCV data type for normalization (e.g., cv2.CV_16U)
        If None, uses standard numpy normalization

    Returns
    -------
    numpy.ndarray
        Normalized image array
    
    Notes
    -----
    Supports different normalization methods:
    - CV_8U: 8-bit unsigned integers (0..255)
    - CV_16U: 16-bit unsigned integers (0..65535)
    - Numpy-style float normalization (0..1)
    """
    if dtype is not None:
        return cv2.normalize(img_array, None, target_min, target_max, 
                              cv2.NORM_MINMAX, dtype=dtype)
    
    # Standard numpy normalization
    img_min = np.amin(img_array)
    img_max = np.amax(img_array)
    
    # Prevent division by zero
    if img_max == img_min:
        return np.zeros_like(img_array, dtype=float)
    
    return (img_array - img_min) / (img_max - img_min) * (target_max - target_min) + target_min

def hex_to_rgb(color: Union[object, str]) -> Tuple[float, float, float]:
    """
    Convert color to RGB values between 0 and 1.

    Parameters
    ----------
    color : Union[Color, str]
        Color object or hex string to convert
        Supports OME Color objects, hex strings, and 'white'

    Returns
    -------
    Tuple[float, float, float]
        RGB values between 0 and 1

    Raises
    ------
    ValueError
        If color is not a supported type
    """
    if hasattr(color, 'as_rgb_tuple'):  # OME Color object
        return tuple(x / 255.0 for x in color.as_rgb_tuple())
    
    if isinstance(color, str):
        if color.lower() == 'white':
            return (1.0, 1.0, 1.0)
        
        color = color.lstrip('#')
        return tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    raise ValueError("Color must be an OME Color object or hex string")

def colorize_grayscale(image: np.ndarray, 
                        color: Union[Tuple[float,float,float], str]) -> np.ndarray:
    """
    Colorize a grayscale image with a specific color.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image array
    color : Union[Tuple[float,float,float], str]
        Color specification (RGB tuple or hex string)

    Returns
    -------
    numpy.ndarray
        RGB image with applied color
    
    Notes
    -----
    - Image is first normalized to [0, 1] range
    - Color is multiplied across RGB channels
    """
    # Normalize the image to [0, 1] if needed
    normalized_image = normalize(image)
    
    # Convert color to RGB if it's a hex string
    if isinstance(color, str):
        color = hex_to_rgb(color)
    
    # Create RGB image by multiplying each channel with the color
    rgb_image = np.zeros((*image.shape, 3), dtype=float)
    rgb_image[..., 0] = normalized_image * color[0]  # Red channel
    rgb_image[..., 1] = normalized_image * color[1]  # Green channel
    rgb_image[..., 2] = normalized_image * color[2]  # Blue channel
    
    return rgb_image

def layer_fluorescent_images(colored_data: List[np.ndarray], 
                              weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Layer multiple fluorescent channels with corresponding colors and weights.

    Parameters
    ----------
    colored_data : List[numpy.ndarray]
        List of RGB images to layer
    weights : Optional[List[float]], optional
        Weights for each channel. Defaults to equal weights if None

    Returns
    -------
    numpy.ndarray
        Combined RGB image

    Raises
    ------
    ValueError
        If number of weights does not match number of images
    """
    # Validate inputs
    if not colored_data:
        raise ValueError("No images provided for layering")

    # Default to equal weights if none provided
    if weights is None:
        weights = [1.0] * len(colored_data)
    elif len(weights) != len(colored_data):
        raise ValueError("Number of weights must match number of images")

    # Validate image shapes are consistent
    h, w, channels = colored_data[0].shape
    combined_image = np.zeros((h, w, channels), dtype=float)

    # Combine images with weights
    for weight, img in zip(weights, colored_data):
        combined_image += weight * img

    # Normalize the combined image
    return normalize(combined_image)

def nd2_extract_marker_color(nd2_file):
    """
    Extract marker and color information from ND2 file.

    Parameters
    ----------
    nd2_file : AICSImage
        ND2 file object containing image metadata

    Returns
    -------
    List[Tuple[str, Tuple[float,float,float]]]
        List of (marker_name, rgb_color) tuples
    """
    # Access channel information from the first image
    channels = nd2_file.metadata.images[0].pixels.channels
    marker_color_list = []

    for idx, channel in enumerate(channels):
        # Extract channel-specific metadata
        marker = channel.name or f"Channel {idx}"
        color = hex_to_rgb(channel.color) if channel.color else (1.0, 1.0, 1.0)
        marker_color_list.append((marker, color))

    return marker_color_list

def nd2_extract_position_info(nd2_file: Union[str, AICSImage]) -> Tuple[Tuple[float, ...], Tuple[float, float], str]:
    """
    Extract Z, Y, and X position information from an ND2 file.
    
    Parameters
    ----------
    nd2_file : str or AICSImage
        File path to the ND2 file or an already loaded AICSImage object.
        
    Returns
    -------
    Tuple[Tuple[float, ...], Tuple[float, float], str]
        A tuple containing:
        - A tuple of absolute Z positions (base_z + relative_positions)
        - A tuple with a single Y and X position
        - The unit of measurement
    """
    # Load the ND2 file if a file path is provided
    img = AICSImage(nd2_file) if isinstance(nd2_file, str) else nd2_file
    
    # Get Y, X positions (these are typically stage positions)
    metadata = img.metadata
    planes = metadata.images[0].pixels.planes
    
    y_pos = planes[0].position_y if planes[0].position_y is not None else 0.0
    x_pos = planes[0].position_x if planes[0].position_x is not None else 0.0

    # Get physical pixel sizes which includes Z step size
    pixel_sizes = img.physical_pixel_sizes
    
    # Calculate Z positions using the number of Z slices and Z step size
    num_z = img.dims.Z
    z_step = pixel_sizes.Z
    
    # Calculate relative Z positions
    if z_step and num_z > 0:
        z_start = -(num_z - 1) * z_step / 2
        relative_z_positions = tuple(z_start + i * z_step for i in range(num_z))
        
        # Get base Z position (focus position)
        base_z = planes[0].position_z if planes[0].position_z is not None else 0.0
        
        # Calculate absolute Z positions
        z_positions = tuple(base_z + rel_z for rel_z in relative_z_positions)
    else:
        z_positions = tuple(range(num_z))
    
    # Get the unit (usually micrometers)
    unit = 'µm'  # Most microscopy data uses micrometers
    if planes[0].position_z_unit:
        unit = planes[0].position_z_unit.value
    
    return z_positions, (y_pos, x_pos), unit
    
def display_images_maxima(images, maxima=None, max_per_row=5):
    """
    Displays images in a grid with a maximum of `max_per_row` images per row.
    
    Parameters:
    - images (list or array): List or array of images (e.g., NumPy arrays or PIL images).
    - max_per_row (int): Maximum number of images displayed horizontally in a row. Default is 5.
    """
    # Validate input
    if not isinstance(images, (list, tuple)):
        raise ValueError("Input images should be a list or array.")
    
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return

    if maxima== None:
        print('No Maxima given')
    
    # Calculate the number of rows needed
    rows = math.ceil(num_images / max_per_row)
    
    # Create a grid layout
    fig, axes = plt.subplots(rows, max_per_row, figsize=(max_per_row * 3, rows * 3))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])

            if maxima!=None:
                ax.scatter(maxima[i][:, 1], maxima[i][:, 0], c='red', s=10, label="Maxima", alpha=.3)
                ax.legend()
            ax.axis('off')  # Hide axes
            
        else:
            ax.axis('off')  # Hide unused subplots
    
    plt.tight_layout()
    plt.show()

def remove_border_touching(labeled_mask, border_distance=0):
    """
    Remove labeled regions that are within specified distance of the image border.
    
    Parameters:
        labeled_mask: 2D numpy array where each region has a unique integer label
        border_distance: int, minimum distance from border (pixels)
                        regions closer than this will be removed
    
    Returns:
        2D numpy array with near-border regions removed (keeps original labels)
    """
    # Create border region mask
    h, w = labeled_mask.shape
    border_region = np.ones_like(labeled_mask, dtype=bool)
    if border_distance > 0:
        border_region[border_distance:h-border_distance, 
                     border_distance:w-border_distance] = False
    
    # Find unique labels in the border region
    border_labels = set(labeled_mask[border_region])
    
    # Remove 0 from labels if present (assuming 0 is background)
    border_labels.discard(0)
    
    # Create output mask
    output_mask = labeled_mask.copy()
    
    # Remove regions that are too close to border
    for label in border_labels:
        output_mask[labeled_mask == label] = 0
    
    return output_mask

def filter_labeled_mask_by_size(
    labeled_mask: np.ndarray, 
    min_size: int = 0, 
    max_size: int = None
    ) -> np.ndarray:
    """
    Filter a labeled mask to keep only regions within specified size bounds.
    
    Parameters:
    -----------
    labeled_mask : np.ndarray
        Input labeled mask where each region has a unique integer label
    min_size : int
        Minimum area (in pixels) for a region to be kept
    max_size : int or None
        Maximum area (in pixels) for a region to be kept. If None, no upper limit is applied
        
    Returns:
    --------
    np.ndarray
        Filtered labeled mask with regions outside size bounds removed
    """
    # Get properties of all regions
    props = measure.regionprops(labeled_mask)
    
    # Create a map of labels to keep
    valid_labels = {
        prop.label for prop in props
        if prop.area >= min_size and (max_size is None or prop.area <= max_size)
    }
    
    # Create new mask with only valid regions
    filtered_mask = np.zeros_like(labeled_mask)
    for label in valid_labels:
        filtered_mask[labeled_mask == label] = label
        
    return filtered_mask

def find_intensity_centers(image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """
    Find centers of mass in a fluorescent intensity image based on binary mask.
    
    Parameters:
    -----------
    image : np.ndarray
        Fluorescent intensity image
    binary_mask : np.ndarray
        Binary mask where True/1 indicates regions of interest
        
    Returns:
    --------
    np.ndarray
        Array of (row, col) coordinates for centers of mass
    """
    # Convert binary mask to labeled mask
    labeled_mask, _ = measure.label(binary_mask, return_num=True, connectivity=2)
    
    # Get properties for all regions
    props = measure.regionprops(labeled_mask, intensity_image=image)
    
    # Extract weighted centroids (center of mass based on intensity)
    centers = []
    for prop in props:
        if prop.weighted_centroid is not None:
            centers.append(prop.weighted_centroid)
            
    if not centers:
        return np.empty((0, 2), dtype=float)
        
    return np.array(centers)

def load_matched_files(base_path, 
                      cell_mask_path=None,
                      foci1_path=None, 
                      foci2_path=None,
                      seg_prob=0.5, 
                      max_prob=0.5, 
                      use_on_windows=False):
    """
    Load and match nd2 files with their corresponding probability masks.
    
    Parameters:
    -----------
    base_path : str
        Path to the base directory containing the nd2 files
    cell_mask_path : str, optional
        Path to the cell mask probability files (default: base_path/Processed/C3/)
    foci1_path : str, optional
        Path to the first foci mask probability files (default: base_path/Processed/C1/)
    foci2_path : str, optional
        Path to the second foci mask probability files (default: base_path/Processed/C2/)
    seg_prob : float, optional (default=0.5)
        Probability threshold for cell mask segmentation
    max_prob : float, optional (default=0.5)
        Probability threshold for foci masks
    use_on_windows : bool, optional (default=False)
        If True, uses Windows-style paths ('\\'), otherwise uses Unix-style ('/')
    """
    try:
        # Set path separator based on OS
        sep = '\\' if use_on_windows else '/'
        if not base_path.endswith(sep):
            data_path = base_path + sep
            
        # Define paths - use custom paths if provided, otherwise use defaults
        processed_path = join(data_path, f'Processed{sep}')
        
        # Use provided paths or default to original structure
        phase_path = cell_mask_path if cell_mask_path else join(processed_path, f'C3{sep}')
        f1_path = foci1_path if foci1_path else join(processed_path, f'C1{sep}')
        f2_path = foci2_path if foci2_path else join(processed_path, f'C2{sep}')
        
        print(f"Using paths:")
        print(f"Data path: {data_path}")
        print(f"Cell mask path: {phase_path}")
        print(f"Foci1 path: {f1_path}")
        print(f"Foci2 path: {f2_path}")
        
        # Validate paths
        for path in [data_path, phase_path, f1_path, f2_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory not found: {path}")
        
        # Initialize lists
        position_list = []
        marker_list = []
        nd2_list = []
        mask_f1 = []
        mask_f2 = []
        labeled_mask_phase = []
        physical_size = None
        delta_t = None
        
        # Get all nd2 files
        nd2_files = sorted([f for f in listdir(data_path) if f.endswith('.nd2')])
        if not nd2_files:
            raise FileNotFoundError(f"No .nd2 files found in {data_path}")
            
        # Create dictionaries for mask files - handle different naming patterns including .tif/.tiff
        # and multiple suffix/endings like '_Probabilities', '_mask', '_masks'
        def mask_base_variants(f, code):
            suffixes = [
                f'_{code}_Probabilities',          # e.g., _C3_Probabilities
                '_Probabilities',
                f'_{code}_mask',
                '_mask',
                f'_{code}_masks',
                '_masks',
            ]
            extensions = ['.tif', '.tiff']
            for suff in suffixes:
                for ext in extensions:
                    if f.endswith(suff + ext):
                        return f[:-len(suff + ext)]
            return None

        phase_masks = {}
        for f in listdir(phase_path):
            base = mask_base_variants(f, 'C3')
            if base is not None:
                phase_masks[base] = f

        f1_masks = {}
        for f in listdir(f1_path):
            base = mask_base_variants(f, 'C1')
            if base is not None:
                f1_masks[base] = f

        f2_masks = {}
        for f in listdir(f2_path):
            base = mask_base_variants(f, 'C2')
            if base is not None:
                f2_masks[base] = f
        
        processed_count = 0
        skipped_count = 0
        
        # Process files in order
        for nd2_file in nd2_files:
            base = nd2_file.rsplit('.', 1)[0]  # Remove extension
            
            # Check if all required masks exist
            has_all_masks = (
                base in phase_masks and
                base in f1_masks and
                base in f2_masks
            )
            
            if not has_all_masks:
                missing_masks = []
                if base not in phase_masks:
                    missing_masks.append('cell mask')
                if base not in f1_masks:
                    missing_masks.append('foci1')
                if base not in f2_masks:
                    missing_masks.append('foci2')
                print(f"Skipping {nd2_file} - Missing masks: {', '.join(missing_masks)}")
                skipped_count += 1
                continue
            
            # Load nd2 file
            try:
                img = AICSImage(join(data_path, nd2_file))
                
                # Validate image data
                if img.data is None or img.data.size == 0:
                    raise ValueError("Empty or invalid image data")
                
                position_list.append(nd2_extract_position_info(img))
                marker_list.append(nd2_extract_marker_color(img))
                nd2_list.append(img.data)
                
                if physical_size is None:
                    physical_size = (img.metadata.images[0].pixels.physical_size_z,
                                   img.metadata.images[0].pixels.physical_size_y,
                                   img.metadata.images[0].pixels.physical_size_x)
                if delta_t is None:
                    delta_t = (img.metadata.images[0].pixels.planes[0].delta_t,
                             img.metadata.images[0].pixels.planes[0].delta_t_unit.value)
                
                del img
            except Exception as e:
                print(f'Error loading {nd2_file}: {str(e)}')
                skipped_count += 1
                continue
                
            # Load corresponding masks
            try:
                # Load and validate all masks
                phase_mask = np.array(load_tif(join(phase_path, phase_masks[base])))
                f1_mask = np.array(load_tif(join(f1_path, f1_masks[base])))
                f2_mask = np.array(load_tif(join(f2_path, f2_masks[base])))

                # ilastik segmentation settings (assuming first channel is background, second channel is foci)
                segmentation_channel = 1
                
                # Validate mask dimensions - handle both 3D (x,y,c) and 2D (x,y) masks
                if len(phase_mask.shape) == 3 and phase_mask.shape[2] >= 2:
                    # Multi-channel mask - use second channel (index 1) which typically contains probabilities 
                    cell_prob_mask = phase_mask[:,:,segmentation_channel]
                else:
                    # Single channel mask - use as is
                    cell_prob_mask = phase_mask
                    
                # Check if phase_mask is already a labeled mask (2D with only 0 and 1 as unique values)
                if len(np.unique(cell_prob_mask)) == 2 and set(np.unique(cell_prob_mask)) == {0, 1}:
                    labeled_mask = cell_prob_mask
                else:
                    # Process cell mask
                    binary_mask = morphology.opening(cell_prob_mask > seg_prob)
                    labeled_mask, cnt = label(binary_mask)

                filtered_mask = filter_labeled_mask_by_size(labeled_mask, min_size=10)
                filtered_mask = remove_border_touching(filtered_mask, border_distance=3)
                labeled_mask_phase.append(filtered_mask)
                
                # Process foci masks - handle both multi-channel and single channel masks
                if len(f1_mask.shape) == 3 and f1_mask.shape[2] >= 2:
                    mask_f1.append(f1_mask[:,:,segmentation_channel] > max_prob)
                else:
                    mask_f1.append(f1_mask > max_prob)
                    
                # Getting the second channel of the foci masks
                if len(f2_mask.shape) == 3 and f2_mask.shape[2] >= 2:
                    mask_f2.append(f2_mask[:,:,segmentation_channel] > max_prob)
                else:
                    mask_f2.append(f2_mask > max_prob)
                
                processed_count += 1
                    
            except Exception as e:
                print(f'Error loading masks for {base}: {str(e)}')
                # Remove the corresponding nd2 data since we couldn't load all masks
                position_list.pop()
                marker_list.pop()
                nd2_list.pop()
                skipped_count += 1
                continue
        
        if processed_count == 0:
            raise ValueError("No files were successfully processed")
            
        print(f"\nProcessing Summary:")
        print(f"Successfully processed: {processed_count} files")
        print(f"Skipped: {skipped_count} files")
        
        return (nd2_list, mask_f1, mask_f2, labeled_mask_phase, 
                position_list, marker_list, physical_size, delta_t)
                
    except Exception as e:
        raise Exception(f"Fatal error in load_matched_files: {str(e)}")

def filter_mask_by_size(mask: np.ndarray, min_size: int = None, max_size: int = None, return_labeled: bool = False):
    """
    Filter objects in a mask based on their size.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        2D binary or labeled mask with objects to filter
    min_size : int, optional
        Minimum object size to keep (number of pixels)
    max_size : int, optional
        Maximum object size to keep (number of pixels)
    return_labeled : bool, optional
        If True, returns labeled mask along with filtered binary mask
    
    Returns:
    --------
    numpy.ndarray or tuple
        Filtered binary mask, or (filtered_mask, labeled_mask) if return_labeled is True
    """
    import skimage.measure as measure

    if mask.max() == 0:
        raise ValueError('No objects found in mask')
    elif mask.max() == 1:
        # Label connected components in the binary mask
        labeled_mask, _ = measure.label(mask, connectivity=2, return_num=True)
    else:
        labeled_mask = mask
    
    # Create a copy of the original mask to modify
    filtered_mask = mask.copy()
    
    # Compute object sizes
    object_sizes = np.bincount(labeled_mask.ravel())[1:]
    
    # Filter out objects based on size criteria
    for label, size in enumerate(object_sizes, start=1):
        # Check if object should be removed
        if (min_size is not None and size < min_size) or \
           (max_size is not None and size > max_size):
            filtered_mask[labeled_mask == label] = 0
    
    # Return based on return_labeled flag
    return (filtered_mask, labeled_mask) if return_labeled else filtered_mask 

def compare_maxima_between_foci_lists(array_list1: list, array_list2: list,
                                    foci1_name: str = "Foci 1", 
                                    foci2_name: str = "Foci 2",
                                    labels: list = None,
                                    individual_idx: int = None) -> list:
    """
    Compare maxima counts between two lists of foci arrays across cells.
    
    Parameters:
    -----------
    array_list1 : list
        List of arrays, each containing [cell_label, maxima_count] pairs for first foci
    array_list2 : list
        List of arrays, each containing [cell_label, maxima_count] pairs for second foci
    foci1_name : str, optional
        Name of first foci type for labeling
    foci2_name : str, optional
        Name of second foci type for labeling
    labels : list, optional
        Labels for each dataset (default: Dataset 1, Dataset 2, etc.)
    individual_idx : int, optional
        Index of single dataset to visualize. If None, shows all datasets.
        
    Returns:
    --------
    list
        List of dictionaries containing statistics for each comparison
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if len(array_list1) != len(array_list2):
        raise ValueError("Both lists must have the same length")
    
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(array_list1))]
    
    if len(labels) != len(array_list1):
        raise ValueError("Number of labels must match number of datasets")
        
    if individual_idx is not None:
        if not 0 <= individual_idx < len(array_list1):
            raise ValueError(f"individual_idx must be between 0 and {len(array_list1)-1}")
        # Convert to lists with single items for consistent processing
        array_list1 = [array_list1[individual_idx]]
        array_list2 = [array_list2[individual_idx]]
        labels = [labels[individual_idx]]
    
    # Calculate statistics for each dataset
    all_stats = []
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(array_list1)))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Lists to store all foci counts for combined correlation
    all_foci1 = []
    all_foci2 = []
    
    for idx, (array1, array2, label, color) in enumerate(zip(array_list1, array_list2, labels, colors)):
        # Verify arrays have matching cell labels
        if len(array1) != len(array2):
            raise ValueError(f"Arrays in dataset {label} must have the same length")
        if not np.array_equal(array1[:, 0], array2[:, 0]):
            raise ValueError(f"Cell labels must match between arrays in dataset {label}")
        
        # Add foci counts to combined lists
        all_foci1.extend(array1[:, 1])
        all_foci2.extend(array2[:, 1])
        
        # Calculate differences and statistics
        differences = array1[:, 1] - array2[:, 1]
        no_foci_both = np.sum((array1[:, 1] == 0) & (array2[:, 1] == 0))
        correlation = np.corrcoef(array1[:, 1], array2[:, 1])[0, 1]
        unique_diffs, diff_counts = np.unique(differences, return_counts=True)
        
        stats = {
            'label': label,
            'total_cells_array1': len(array1),
            'total_cells_array2': len(array2),
            'differences': dict(zip(unique_diffs, diff_counts)),
            'no_foci_both': no_foci_both,
            'correlation': correlation,
            'more_foci1': np.sum(differences > 0),
            'more_foci2': np.sum(differences < 0),
            'equal_counts': np.sum(differences == 0)
        }
        all_stats.append(stats)
        
        # Bar plot of differences
        ax1.bar(unique_diffs + idx*0.2, diff_counts, 0.2, 
                color=color, alpha=0.7, label=label)
        
        # Scatter plot
        ax2.scatter(array1[:, 1], array2[:, 1], 
                   color=color, alpha=0.5, label=label)
    
    # Calculate combined Pearson correlation
    from scipy import stats
    pearson_r, p_value = stats.pearsonr(all_foci1, all_foci2)
    
    # Add combined correlation line
    z = np.polyfit(all_foci1, all_foci2, 1)
    p = np.poly1d(z)
    x_range = np.array([0, max(all_foci1)])
    ax2.plot(x_range, p(x_range), 'k--', alpha=0.7, label='Correlation')
    
    # Calculate total statistics
    total_differences = np.array(all_foci1) - np.array(all_foci2)
    total_unique_diffs, total_diff_counts = np.unique(total_differences, return_counts=True)
    
    total_stats = {
        'label': 'Total',
        'total_cells_array1': sum(stats['total_cells_array1'] for stats in all_stats),
        'total_cells_array2': sum(stats['total_cells_array2'] for stats in all_stats),
        'differences': dict(zip(total_unique_diffs, total_diff_counts)),
        'no_foci_both': sum(stats['no_foci_both'] for stats in all_stats),
        'correlation': pearson_r,
        'pearson_r': pearson_r,
        'p_value': p_value,
        'more_foci1': np.sum(total_differences > 0),
        'more_foci2': np.sum(total_differences < 0),
        'equal_counts': np.sum(total_differences == 0)
    }
    
    all_stats.append(total_stats)
    
    # Add correlation statistics text
    stats_text = f'Correlation:\nr = {pearson_r:.3f}\np = {p_value:.2e}'
    ax2.text(0.95, 0.95, stats_text,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             horizontalalignment='right')

    # Customize bar plot
    ax1.set_xlabel(f'Difference in maxima ({foci1_name} - {foci2_name})')
    ax1.set_ylabel('Number of cells')
    ax1.set_title('Distribution of Maxima Count Differences')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Customize scatter plot
    ax2.set_xlabel(f'{foci1_name} count')
    ax2.set_ylabel(f'{foci2_name} count')
    ax2.set_title('Correlation of Maxima Counts')
    
    # Add identity line (y=x)
    max_val = max(max(np.max(arr[:, 1]) for arr in array_list1),
                  max(np.max(arr[:, 1]) for arr in array_list2))
    x_line = np.array([0, max_val])
    ax2.plot(x_line, x_line, 'r--', alpha=0.5, label='y=x line')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return all_stats

def analyze_colocalization_lists(coords1_list: list, coords2_list: list,
                               max_distance: float = None, 
                               one_to_one: bool = True,
                               labels: list = None,
                               individual_idx: int = None,
                               in_nm: bool = False,
                               pixel_size: float = 0.107,
                               bin_cnt: int = None,
                               cell_masks: list = None,
                               colocalisation_threshold: float = None,
                               cell_edge_tolerance: float = 0.0,
                               filter_boundaries: bool = False,
                               save_plot: bool = False,               
                               plot_filename: str = "colocalization_plot.png",
                               title: str = None):
    """
    Analyze colocalization for multiple sets of coordinates with option for individual analysis.
    Only considers colocalization within the same cell boundaries, but also allows maxima
    within a given distance (cell_edge_tolerance) to be considered as valid.
    Now supports colocalisation_threshold to determine valid/invalid colocalizations.

    Parameters:
    -----------
    coords1_list : list
        List of numpy arrays, each of shape (N, 2) containing (x, y) coordinates
    coords2_list : list
        List of numpy arrays, each of shape (M, 2) containing (x, y) coordinates
    max_distance : float, optional
        Maximum distance to consider for pairing (in nm if in_nm=True, pixels otherwise)
    one_to_one : bool, optional
        If True, ensure each point matches only once using Hungarian algorithm
    labels : list, optional
        Labels for each dataset (default: Dataset 1, Dataset 2, etc.)
    individual_idx : int, optional
        Index of single dataset to visualize. If None, shows all datasets.
    in_nm : bool, optional
        If True, display distances in nanometers instead of pixels
    bin_cnt : int, optional
        Number of bins for the histogram. If None, uses pixel-wise bins.
        If specified, divides max_distance into bin_cnt equal bins.
    cell_masks : list, optional
        List of cell segmentation masks. Each mask should be a 2D numpy array where
        each cell has a unique integer label (0 = background). If None, no cell
        boundary restrictions are applied.
    colocalisation_threshold : float, optional
        Threshold (in nm if in_nm=True, else in pixels) to determine valid/invalid colocalizations
    cell_edge_tolerance : float, optional
        Distance (in nm if in_nm=True, else in pixels) from the cell boundary within which
        maxima are still considered as belonging to the cell (default: 0.0)
    filter_boundaries : bool, optional
        If True, only consider pairs within the same cell boundaries
    save_plot : bool, optional
        If True, save the plot to a file
    plot_filename : str, optional
        Filename for the plot. If None, no plot is saved.
    title : str, optional
        Title for the plot. If None, a default title is used.

    Returns:
    --------
    dict or list
        Statistics about the colocalization analysis, including valid/invalid colocalizations
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from scipy.ndimage import distance_transform_edt

    # Convert max_distance from nm to pixels if needed
    max_distance_px = int(np.ceil(max_distance / pixel_size)) if (max_distance is not None and in_nm) else max_distance

    # Convert colocalisation_threshold from nm to pixels if needed
    colocalisation_threshold_px = None
    if colocalisation_threshold is not None:
        colocalisation_threshold_px = colocalisation_threshold / pixel_size if in_nm else colocalisation_threshold

    # Convert cell_edge_tolerance from nm to pixels if needed
    cell_edge_tolerance_px = cell_edge_tolerance / pixel_size if in_nm else cell_edge_tolerance

    if len(coords1_list) != len(coords2_list):
        raise ValueError("coords1_list and coords2_list must have the same length")

    # Validate cell_masks if provided
    if cell_masks is not None:
        if len(cell_masks) != len(coords1_list):
            raise ValueError("cell_masks must have the same length as coords1_list and coords2_list")
        for mask in cell_masks:
            if not isinstance(mask, np.ndarray) or mask.ndim != 2:
                raise ValueError("Each cell mask must be a 2D numpy array")

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(coords1_list))]

    if len(labels) != len(coords1_list):
        raise ValueError("Number of labels must match number of datasets")

    if individual_idx is not None:
        if not 0 <= individual_idx < len(coords1_list):
            raise ValueError(f"individual_idx must be between 0 and {len(coords1_list)-1}")
        coords1_list = [coords1_list[individual_idx]]
        coords2_list = [coords2_list[individual_idx]]
        labels = [labels[individual_idx]]
        if cell_masks is not None:
            cell_masks = [cell_masks[individual_idx]]

    # First pass to determine max_distance if not provided
    if max_distance_px is None:
        max_dist_observed = 0
        for coords1, coords2 in zip(coords1_list, coords2_list):
            if len(coords1) == 0 or len(coords2) == 0:
                continue
            distances = cdist(coords1, coords2)
            max_dist_observed = max(max_dist_observed, np.ceil(distances.min(axis=1).max()))
        max_distance_px = int(max_dist_observed)

    # Initialize arrays for aggregating results
    total_distance_counts = np.zeros(max_distance_px + 1, dtype=int)
    all_stats = []

    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(coords1_list)))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Modify binning based on bin_cnt
    if bin_cnt is not None:
        if in_nm:
            bin_edges = np.linspace(0, max_distance, bin_cnt + 1)
        else:
            bin_edges = np.linspace(0, max_distance_px, bin_cnt + 1)
        total_distance_counts = np.zeros(bin_cnt, dtype=int)
    else:
        bin_edges = np.arange(max_distance_px + 2) * (pixel_size if in_nm else 1)
        total_distance_counts = np.zeros(max_distance_px + 1, dtype=int)

    # Process each pair of coordinate sets
    for idx, (coords1, coords2, label, color) in enumerate(zip(coords1_list, coords2_list, labels, colors)):

        if len(coords1) == 0 or len(coords2) == 0:
            continue

        # Get cell mask for this dataset if available
        cell_mask = cell_masks[idx] if cell_masks is not None else None

        # Filter coordinates by cell boundaries if cell mask is provided
        if cell_mask is not None and filter_boundaries:
            coords1_filtered, coords1_cell_ids = filter_coords_by_cells_with_tolerance(
                coords1, cell_mask, cell_edge_tolerance_px)
            coords2_filtered, coords2_cell_ids = filter_coords_by_cells_with_tolerance(
                coords2, cell_mask, cell_edge_tolerance_px)

            if len(coords1_filtered) == 0 or len(coords2_filtered) == 0:
                continue
        else:
            coords1_filtered = coords1
            coords2_filtered = coords2
            coords1_cell_ids = None
            coords2_cell_ids = None
        
        
        # Calculate pairwise distances
        distances = cdist(coords1_filtered, coords2_filtered)

        # Apply cell boundary restrictions if cell mask is provided
        if cell_mask is not None and filter_boundaries:
            # Create a mask where only pairs within the same cell are valid
            cell_compatibility_mask = np.zeros_like(distances, dtype=bool)
            for i, cell_id1 in enumerate(coords1_cell_ids):
                for j, cell_id2 in enumerate(coords2_cell_ids):
                    if (cell_id1 == cell_id2) and (cell_id1 > 0):  # Same cell and not background
                        cell_compatibility_mask[i, j] = True

            # Apply cell compatibility mask to distances
            distances = np.where(cell_compatibility_mask, distances, np.inf)

        
        # Apply distance threshold before pairing
        if max_distance_px is not None:
            distances_masked = np.where(distances <= max_distance_px, distances, np.inf)
        else:
            distances_masked = distances

        if one_to_one:
            # Only consider pairs within max_distance and same cell
            valid_pairs = np.any(distances_masked != np.inf, axis=1)
            if np.any(valid_pairs):
                # Create a copy of distances_masked with finite values for valid pairs
                assignment_distances = distances_masked.copy()
                # Replace inf with a large finite number for the assignment algorithm
                assignment_distances[assignment_distances == np.inf] = np.finfo(np.float64).max

                try:
                    row_ind, col_ind = linear_sum_assignment(assignment_distances)
                    # Filter out infinite distance pairs
                    valid_assignments = distances_masked[row_ind, col_ind] != np.inf
                    row_ind = row_ind[valid_assignments]
                    col_ind = col_ind[valid_assignments]
                    min_distances = distances[row_ind, col_ind]
                    pair_mapping = list(zip(row_ind, col_ind))
                except ValueError:
                    # If assignment fails, treat as no valid pairs
                    min_distances = np.array([])
                    pair_mapping = []
            else:
                min_distances = np.array([])
                pair_mapping = []
        else:
            # Find minimum distances within threshold and same cell
            min_distances = []
            pair_mapping = []
            for i in range(len(coords1_filtered)):
                if np.any(distances_masked[i] != np.inf):
                    min_idx = np.argmin(distances_masked[i])
                    min_distances.append(distances[i, min_idx])
                    pair_mapping.append((i, min_idx))
            min_distances = np.array(min_distances)

        if len(min_distances) == 0:
            continue

        
        # Calculate distance counts based on bin_cnt
        if bin_cnt is not None:
            distance_counts, _ = np.histogram(min_distances * (pixel_size if in_nm else 1), 
                                           bins=bin_edges)
        else:
            distance_counts = np.zeros(max_distance_px + 1, dtype=int)
            for dist in min_distances:
                dist_int = int(np.round(dist))
                if dist_int <= max_distance_px:
                    distance_counts[dist_int] += 1

        # Apply colocalisation_threshold to split valid/invalid
        if colocalisation_threshold_px is not None:
            valid_mask = min_distances <= colocalisation_threshold_px
            valid_pairs = [pair_mapping[i] for i, v in enumerate(valid_mask) if v]
            valid_distances = min_distances[valid_mask] * (pixel_size if in_nm else 1)
            invalid_pairs = [pair_mapping[i] for i, v in enumerate(valid_mask) if not v]
            invalid_distances = min_distances[~valid_mask] * (pixel_size if in_nm else 1)
        else:
            valid_pairs = pair_mapping
            valid_distances = min_distances * (pixel_size if in_nm else 1)
            invalid_pairs = []
            invalid_distances = np.array([])

        # Add to total counts
        total_distance_counts += distance_counts

        # Calculate statistics
        stats = {
            'label': label,
            'total_pairs': len(pair_mapping),
            'mean_distance': np.mean(min_distances) * (pixel_size if in_nm else 1),
            'median_distance': np.median(min_distances) * (pixel_size if in_nm else 1),
            'max_distance': np.max(min_distances) * (pixel_size if in_nm else 1),
            'distance_counts': distance_counts,
            'bin_edges': bin_edges,
            'valid_pairs': valid_pairs,
            'valid_distances': valid_distances,
            'invalid_pairs': invalid_pairs,
            'invalid_distances': invalid_distances,
            'colocalisation_threshold': colocalisation_threshold,
            'distance_unit': 'nm' if in_nm else 'pixels',
        }
        all_stats.append(stats)

        # If showing individual dataset, plot its distribution
        if individual_idx is not None:
            if bin_cnt is not None:
                width = (bin_edges[1] - bin_edges[0]) * 0.8
                ax.bar(bin_edges[:-1], distance_counts, width=width,
                      color=color, alpha=0.7, label=label)
            else:
                if in_nm:
                    bin_edges_px = np.arange(len(distance_counts) + 1) * pixel_size
                    ax.bar(bin_edges_px[:-1], distance_counts, width=pixel_size*0.8,
                          color=color, alpha=0.7, label=label)
                else:
                    ax.bar(range(len(distance_counts)), distance_counts,
                          color=color, alpha=0.7, label=label)
            # Add vertical line for colocalisation_threshold
            if colocalisation_threshold is not None:
                ax.axvline(colocalisation_threshold, color='red', linestyle='--', label='Colocalisation threshold')

    # If showing all datasets combined, plot total distribution
    if individual_idx is None:
        if bin_cnt is not None:
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            ax.bar(bin_edges[:-1], total_distance_counts, 
                  width=width, alpha=0.7, label='Combined datasets')
        else:
            if in_nm:
                bin_edges_px = np.arange(len(total_distance_counts) + 1) * pixel_size
                ax.bar(bin_edges_px[:-1], total_distance_counts, 
                      width=pixel_size*0.8, alpha=0.7, label='Combined datasets')
            else:
                ax.bar(range(len(total_distance_counts)), total_distance_counts,
                      alpha=0.7, label='Combined datasets')

        # Calculate combined statistics
        total_pairs = np.sum(total_distance_counts)
        if bin_cnt is not None:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_distance = np.average(bin_centers, weights=total_distance_counts)
            # Approximate median for binned data
            cumsum = np.cumsum(total_distance_counts)
            median_idx = np.searchsorted(cumsum, total_pairs / 2)
            median_distance = bin_centers[median_idx]
        else:
            distances_array = np.repeat(range(len(total_distance_counts)), total_distance_counts)
            if len(distances_array) > 0:
                mean_distance = np.mean(distances_array) * (pixel_size if in_nm else 1)
                median_distance = np.median(distances_array) * (pixel_size if in_nm else 1)

        if total_pairs > 0:
            stats_text = f'Total pairs: {total_pairs}\n'
            stats_text += f'Mean distance: {mean_distance:.2f} {"nm" if in_nm else "px"}\n'
            stats_text += f'Median distance: {median_distance:.2f} {"nm" if in_nm else "px"}'
            if colocalisation_threshold is not None:
                stats_text += f'\nColocalisation threshold: {colocalisation_threshold:.2f} {"nm" if in_nm else "px"}'

            ax.text(0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # Add vertical line for colocalisation_threshold
        if colocalisation_threshold is not None:
            ax.axvline(colocalisation_threshold, color='red', linestyle='--', label='Colocalisation threshold')

    ax.set_xlabel('Distance (nm)' if in_nm else 'Distance (pixels)')
    ax.set_ylabel('Number of pairs')
    if title is None:
        title = 'Distance Distribution (Within Cells)' if cell_masks is not None else 'Distance Distribution'
    else:
        title = title
    if individual_idx is not None:
        title += f' - {labels[0]}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if individual_idx is not None:
        ax.legend()
    elif colocalisation_threshold is not None:
        ax.legend()

    plt.tight_layout()

    if save_plot:
            from pathlib import Path
            plot_path = Path.cwd() / plot_filename  # saves next to the notebook
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved to: {plot_path}")
    
    plt.show()

    return all_stats if individual_idx is None else all_stats[0]

def filter_coords_by_cells_with_tolerance(coords: np.ndarray, cell_mask: np.ndarray, tolerance_px: float):
    """
    Filter coordinates to only include those within cell boundaries, or within a given
    distance (tolerance_px) to the nearest cell boundary. Returns the filtered coordinates
    and their corresponding cell IDs (0 if not in any cell).

    Parameters:
    -----------
    coords : np.ndarray
        Array of shape (N, 2) containing (x, y) coordinates
    cell_mask : np.ndarray
        2D array where each cell has a unique integer label (0 = background)
    tolerance_px : float
        Distance in pixels from the cell boundary within which maxima are still considered valid

    Returns:
    --------
    tuple
        (filtered_coords, cell_ids) where filtered_coords contains only coordinates
        within cells or within tolerance of a cell, and cell_ids contains the corresponding
        cell ID for each coordinate (0 if not in any cell)
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    if len(coords) == 0:
        return np.array([]), np.array([])

    filtered_coords = []
    cell_ids = []

    # For each cell, create a mask and a distance map to the cell boundary
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]  # Exclude background

    # Precompute distance maps for each cell
    cell_distance_maps = {}
    for cell_id in cell_labels:
        cell_binary = (cell_mask == cell_id).astype(np.uint8)
        # Distance inside the cell (to background)
        dist_inside = distance_transform_edt(cell_binary)
        # Distance outside the cell (to cell)
        dist_outside = distance_transform_edt(1 - cell_binary)
        # For each pixel, if inside cell, dist = 0; if outside, dist = dist_outside
        cell_distance_maps[cell_id] = (cell_binary, dist_inside, dist_outside)

    for coord in coords:
        x, y = int(coord[1]), int(coord[0])
        # Check bounds
        if 0 <= y < cell_mask.shape[0] and 0 <= x < cell_mask.shape[1]:
            cell_id = cell_mask[y, x]
            if cell_id > 0:
                # Inside a cell
                filtered_coords.append(coord)
                cell_ids.append(cell_id)
            else:
                # Not inside a cell, check if within tolerance of any cell
                found = False
                for cid in cell_labels:
                    cell_binary, dist_inside, dist_outside = cell_distance_maps[cid]
                    # Only consider points outside the cell
                    if cell_binary[y, x] == 0:
                        # Distance to this cell
                        dist = dist_outside[y, x]
                        if dist <= tolerance_px:
                            filtered_coords.append(coord)
                            cell_ids.append(cid)
                            found = True
                            break
                # If not within tolerance of any cell, skip
    return np.array(filtered_coords), np.array(cell_ids)
