# FliI ATPase Analysis

This repository contains tools for analyzing FliI ATPase protein localization and colocalization in bacterial cells using microscopy data processed with ilastik.

## Overview

The analysis pipeline processes ND2 microscopy files containing:
- **FliI** (dsRed channel) - ATPase protein
- **FliG** (GFP channel) - Motor switch protein  
- **Phase** (Phase contrast) - Cell segmentation

The workflow includes:
1. Loading ND2 files and corresponding ilastik probability masks
2. Cell segmentation and foci detection
3. Colocalization analysis between FliI and FliG proteins
4. Statistical analysis and visualization

## Files

- `ilastik - Cell segmentation and Maxima detection_organized.ipynb` - Main analysis notebook
- `Flil_notebook_utils.py` - Utility functions for image processing and analysis
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Setup

### Prerequisites

- Python 3.7 or higher
- ilastik (for generating probability masks)

### Installation

1. Clone or download this repository
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Structure

Organize your data in the following structure:
```
base_path/
└── Data/
    ├── [nd2 files]
    └── Processed/
        ├── C3/
        │   └── [phase masks - *_C3_Probabilities.tif]
        ├── C1/
        │   └── [foci1 masks - *_C1_Probabilities.tif]
        └── C2/
            └── [foci2 masks - *_C2_Probabilities.tif]
```

## Usage

### Running the Analysis

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook "ilastik - Cell segmentation and Maxima detection_organized.ipynb"
   ```

2. Update the data path in the notebook:
   ```python
   base_dir = '/path/to/your/data'
   ```

3. Adjust analysis parameters as needed:
   - `cell_seg_prob` - Cell segmentation probability threshold
   - `foci_seg_prob` - Foci detection probability threshold
   - `min_foci1_area`, `max_foci1_area` - FliI foci size filters
   - `min_foci2_area`, `max_foci2_area` - FliG foci size filters

### Key Functions

#### `load_matched_files()`
Loads ND2 files and corresponding probability masks:
```python
(nd2_list, mask_C1, mask_C3, labeled_mask_phase,
 position_list, marker_list, physical_size, delta_t) = load_matched_files(
    base_path=base_dir,
    cell_mask_path=os.path.join(base_dir, 'Processed', 'C3'),
    foci1_path=os.path.join(base_dir, 'Processed', 'C1'),
    foci2_path=os.path.join(base_dir, 'Processed', 'C2'),
    seg_prob=0.8,
    max_prob=0.8
)
```

#### `analyze_colocalization_lists()`
Performs colocalization analysis between two sets of coordinates:
```python
stats = analyze_colocalization_lists(
    coords1_list=il_maxima_coords_c1,
    coords2_list=il_maxima_coords_c2,
    cell_masks=labeled_mask_phase,
    max_distance=30,
    colocalisation_threshold=2,
    in_nm=True,
    pixel_size=0.107
)
```

#### `compare_maxima_between_foci_lists()`
Compares maxima counts between different foci types:
```python
stats = compare_maxima_between_foci_lists(
    il_results_c1, 
    il_results_c2,
    foci1_name="FliI",
    foci2_name="FliG"
)
```

## Analysis Outputs

The notebook generates:

1. **Cell segmentation visualization** - RGB composite images with detected cells
2. **Foci detection results** - Counts of FliI and FliG foci per cell
3. **Colocalization analysis** - Distance distributions and colocalization statistics
4. **Statistical summaries** - Tables with group-wise comparisons
5. **Plots** - Histograms, scatter plots, and violin plots

### Output Files (when saving is enabled)

- `statistics_summary.xlsx` - Summary statistics table
- `colocalization_summary.xlsx` - Colocalization analysis results
- `colocalization_plot_*.png` - Individual colocalization plots

## Parameters

### Image Processing
- **seg_prob** (default: 0.5) - Cell segmentation probability threshold
- **max_prob** (default: 0.5) - Foci detection probability threshold
- **pixel_size** (default: 0.107) - Physical pixel size in micrometers

### Foci Filtering
- **min_foci1_area** (default: 6) - Minimum FliI foci area in pixels
- **max_foci1_area** (default: 20) - Maximum FliI foci area in pixels
- **min_foci2_area** (default: 1) - Minimum FliG foci area in pixels
- **max_foci2_area** (default: 50) - Maximum FliG foci area in pixels

### Colocalization Analysis
- **max_distance** (default: 30) - Maximum distance for colocalization (pixels)
- **colocalisation_threshold** (default: 2) - Distance threshold for valid colocalization
- **cell_edge_tolerance** (default: 0.0) - Tolerance for foci near cell boundaries

## Dependencies

- **numpy** - Numerical computations
- **scipy** - Scientific computing (spatial operations, statistics)
- **pandas** - Data manipulation
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical visualizations
- **opencv-python** - Image processing
- **scikit-image** - Image analysis and morphology
- **aicsimageio** - ND2 file reading
- **tifffile** - TIFF file handling

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure your data follows the expected directory structure
2. **Memory issues**: Process smaller batches of images or reduce image resolution
3. **Import errors**: Verify all dependencies are installed correctly
4. **Empty results**: Check probability thresholds and file naming conventions

### File Naming Convention

Ensure your files follow this naming pattern:
- ND2 files: `example.nd2`
- Phase masks: `example_C3_Probabilities.tif`
- FliI masks: `example_C1_Probabilities.tif`
- FliG masks: `example_C2_Probabilities.tif`

## Citation

If you use this analysis pipeline in your research, please cite the relevant publications for:
- ilastik (Berg et al., 2019)
- aicsimageio (Moore et al., 2021)
- scikit-image (van der Walt et al., 2014)

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or issues, please contact the project maintainers.
