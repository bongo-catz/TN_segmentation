import SimpleITK as sitk
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import sys
import logging as log
import click
import os
import csv
from helper_class.region_selector import RegionSelector
from helper_class.region_viewer import RegionViewer
import subprocess
import antspynet
from antspynet.utilities import brain_extraction as antspynet_brain_extraction
import ants
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# TO DO:
# 1) Do brain extraction (remove mask of skull)

def load_image(file_path):
    """Load a NIfTI image (.nii.gz) and return a SimpleITK image object."""
    try:
        image = sitk.ReadImage(file_path)
        image = sitk.Cast(image, sitk.sitkFloat32)
        size = image.GetSize()
        spacing = image.GetSpacing()
        log.info(f"Loaded image {file_path}")
        log.info(f"Size: {size}, Spacing: {spacing}")
        log.info(f"Origin: {image.GetOrigin()}")
        log.info(f"Direction: {image.GetDirection()}")
        log.info(f"Spacing: {image.GetSpacing()}")
        return image
    except Exception as e:
        log.info(f"Error loading image {file_path}: {str(e)}")
        sys.exit(1)

def resample_to_reference(moving_image, reference_image):
    """Resample moving image to match reference image resolution."""
    log.info("Resampling image to reference space...")
    log.info(f"Original size: {moving_image.GetSize()}")
    log.info(f"Original spacing: {moving_image.GetSpacing()}")
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    
    resampled = resampler.Execute(moving_image)
    log.info(f"Resampled size: {resampled.GetSize()}")
    log.info(f"Resampled spacing: {resampled.GetSpacing()}")
    return resampled

def normalize_image(image):
    """Normalize image intensities to mean 0 and variance 1."""
    try:
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        log.info(f"Before normalization - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        
        normalized = sitk.Normalize(image)
        normalized = sitk.Cast(normalized, sitk.sitkFloat32)  # Explicit cast to float32
        
        stats.Execute(normalized)
        log.info(f"After normalization - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        return normalized
    except Exception as e:
        log.info(f"Error in normalization: {str(e)}")
        sys.exit(1)

def validate_region(region, name=""):
    """Validate that a region is not empty or invalid."""
    try:
        size = region.GetSize()
        if any(s <= 0 for s in size):
            log.info(f"Invalid {name} region size: {size}")
            return False
        
        stats = sitk.StatisticsImageFilter()
        stats.Execute(region)
        log.info(f"{name} region stats:")
        log.info(f"Size: {size}")
        log.info(f"Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        log.info(f"Mean: {stats.GetMean()}")
        return True
    except Exception as e:
        log.info(f"Error validating {name} region: {str(e)}")
        return False

def center_transform_initialize(fixed_image, moving_image, transform_type='euler'):
    """Initialize transform by aligning image centers.
    
    transform_type options:
    - 'euler': rotation + translation (6 DOF)
    - 'similarity': rotation + translation + isotropic scaling (7 DOF)
    - 'versor': rotation (quaternion) + translation (6 DOF)
    - 'affine': rotation + translation + scaling + shearing (12 DOF)
    """
    if transform_type == 'euler':
        transform = sitk.Euler3DTransform()
    elif transform_type == 'similarity':
        transform = sitk.Similarity3DTransform()
    elif transform_type == 'versor':
        transform = sitk.VersorRigid3DTransform()
    elif transform_type == 'affine':
        transform = sitk.AffineTransform(3)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
        
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image,
        transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    log.info(f"Using {transform_type} transform with {len(initial_transform.GetParameters())} parameters")
    return initial_transform

def create_mask_from_bounds(image, bounds):
    """Create a binary mask from the selected bounds."""
    # Create a zero-filled array of the same size as the image
    mask_array = np.zeros(sitk.GetArrayFromImage(image).shape, dtype=np.uint8)
    
    # Fill the selected region with ones
    mask_array[
        bounds['z'][0]:bounds['z'][1],
        bounds['x'][0]:bounds['x'][1],
        bounds['y'][0]:bounds['y'][1]
    ] = 1
    
    # Convert back to SimpleITK image
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    
    # Verify mask creation
    mask_stats = sitk.StatisticsImageFilter()
    mask_stats.Execute(mask)
    log.info("Mask statistics:")
    log.info(f"Size: {mask.GetSize()}")
    log.info(f"Sum of mask (number of selected voxels): {mask_stats.GetSum()}")
    log.info(f"Selected region dimensions: {bounds['z'][1]-bounds['z'][0]} x {bounds['x'][1]-bounds['x'][0]} x {bounds['y'][1]-bounds['y'][0]}")
    
    return mask

def calculate_grid_spacing(image, grid_size):
    """Convert grid size to physical spacing based on image dimensions."""
    image_size = image.GetSize()
    physical_dimensions = [sz * sp for sz, sp in zip(image_size, image.GetSpacing())]
    return [dim / (sz - 1) for dim, sz in zip(physical_dimensions, [grid_size] * 3)]

def calculate_grid_size(image, grid_spacing):
    """Convert physical grid spacing to grid size."""
    image_size = image.GetSize()
    physical_dimensions = [sz * sp for sz, sp in zip(image_size, image.GetSpacing())]
    return [int(dim / spacing) + 1 for dim, spacing in zip(physical_dimensions, [grid_spacing] * 3)]

def perform_initial_registration(fixed_image, moving_image, transform_type='euler', grid_spacing=50):
    """
    Perform initial registration on full images.
    
    transform_type options:
    - 'euler': rotation + translation (6 DOF)
    - 'similarity': rotation + translation + isotropic scaling (7 DOF)
    - 'versor': rotation (quaternion) + translation (6 DOF)
    - 'affine': rotation + translation + scaling + shearing (12 DOF)
    - 'bspline': non-rigid B-spline transformation
    
    grid_spacing: Physical spacing between B-spline control points in mm
                 Smaller spacing = more control points = more local deformation
                 Larger spacing = fewer control points = more global deformation
    """
    try:
        log.info("\nPerforming initial registration on full images...")
        log.info(f"Using transform type: {transform_type}")
        
        registration = sitk.ImageRegistrationMethod()
        
        if transform_type == 'bspline':
            # Calculate grid size from desired physical spacing
            transform_mesh_size = calculate_grid_size(fixed_image, grid_spacing)
            initial_transform = sitk.BSplineTransformInitializer(fixed_image, transform_mesh_size)
            
            actual_spacing = calculate_grid_spacing(fixed_image, transform_mesh_size[0])
            log.info(f"B-spline transform settings:")
            log.info(f"Requested grid spacing: {grid_spacing}mm")
            log.info(f"Actual grid spacing: {actual_spacing[0]:.1f}mm")
            log.info(f"Grid size: {transform_mesh_size[0]}x{transform_mesh_size[1]}x{transform_mesh_size[2]} control points")
            log.info(f"Image size: {fixed_image.GetSize()}")
            log.info(f"Image spacing: {fixed_image.GetSpacing()}")
            
            # Set optimizer for B-spline
            registration.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-4,
                numberOfIterations=50,
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=200,
                costFunctionConvergenceFactor=1e6)
        else:
            # Initialize center-aligned transform
            if transform_type == 'euler':
                transform = sitk.Euler3DTransform()
            elif transform_type == 'similarity':
                transform = sitk.Similarity3DTransform()
            elif transform_type == 'versor':
                transform = sitk.VersorRigid3DTransform()
            elif transform_type == 'affine':
                transform = sitk.AffineTransform(3)
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
                
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, 
                moving_image,
                transform,
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
            
            log.info(f"Using {transform_type} transform with {len(initial_transform.GetParameters())} parameters")
            
            # Set optimizer for linear transforms
            registration.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=100,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10)
            registration.SetOptimizerScalesFromPhysicalShift()
        
        registration.SetInitialTransform(initial_transform)
        
        # Set up similarity metric
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.01)
        
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Add observers to track progress
        def command_iteration(method):
            log.info(f"Iteration {method.GetOptimizerIteration()}: "
                  f"Metric = {method.GetMetricValue():.6f}")
            
        registration.AddCommand(sitk.sitkIterationEvent, 
                              lambda: command_iteration(registration))
        
        # Perform registration
        final_transform = registration.Execute(fixed_image, moving_image)
        
        if transform_type != 'bspline':
            log.info("Final transform parameters: %s", final_transform.GetParameters())
        
        # Apply transform
        result = sitk.Resample(moving_image, fixed_image, final_transform)
        
        return result, final_transform
        
    except Exception as e:
        log.info(f"Error during initial registration: {str(e)}")
        return None, None

def get_incremented_logfile_name(base_dir, prefix="alignment", extension=".log"):
    """Find the next available log file name in the format alignment_01.log, alignment_02.log, etc."""
    idx = 1
    while True:
        filename = f"{prefix}_{idx:02d}{extension}"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            return full_path
        idx += 1
        
def perform_nonrigid_registration(fixed_image, moving_image, mask, grid_spacing=30):
    """Perform non-rigid registration on masked region."""
    try:
        log.info("\nPerforming non-rigid registration on selected region...")
        
        # Calculate grid size from desired physical spacing
        transform_mesh_size = calculate_grid_size(fixed_image, grid_spacing)
        log.info(f"Using B-spline transform with grid spacing: {grid_spacing}mm")
        log.info(f"Grid size: {transform_mesh_size[0]}x{transform_mesh_size[1]}x{transform_mesh_size[2]} control points")
        
        # Initialize B-spline transform
        bspline = sitk.BSplineTransformInitializer(fixed_image, transform_mesh_size)
        
        registration = sitk.ImageRegistrationMethod()
        registration.SetInitialTransform(bspline)
        registration.SetMetricFixedMask(mask)
        
        # Set up similarity metric
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.4)
        
        # Set multi-resolution
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Set linear interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer for non-rigid registration
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=75,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7)
        
        # Add observers to track progress
        def command_iteration(method):
            log.info(f"Iteration {method.GetOptimizerIteration()}: "
                  f"Metric = {method.GetMetricValue():.6f}")
            
        registration.AddCommand(sitk.sitkIterationEvent, 
                              lambda: command_iteration(registration))
        
        # Perform non-rigid registration
        final_transform = registration.Execute(fixed_image, moving_image)
        
        # Apply final transform
        transformed_image = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32)
        
        # Convert mask to same type as image for masking operation
        # mask_converted = sitk.Cast(mask, transformed_image.GetPixelID())
        
        # Apply mask to get only the region of interest
        # transformed_region = sitk.Mask(transformed_image, mask_converted)

        return transformed_image, final_transform
        
    except Exception as e:
        log.info(f"Error during non-rigid registration: {str(e)}")
        return None, None

def perform_nonrigid_registration_v2(fixed_image, moving_image, mask, grid_spacing=30):
    """Optimized non-rigid registration for MRI-MRA vascular alignment."""
    try:
        log.info("\nStarting optimized MRI-MRA non-rigid registration...")
        
        # ===== 1. PRE-PROCESSING ENHANCEMENTS =====
        # Vessel-specific preprocessing for MRA
        moving_processed = vessel_enhancement_filter(moving_image)
        
        # Anatomical preservation for MRI
        fixed_processed = anatomical_preservation_filter(fixed_image)
        
        # ===== 2. MASK OPTIMIZATION =====
        # Dilate mask for boundary tolerance (3mm dilation)
        dilated_mask = sitk.BinaryDilate(mask, [3]*mask.GetDimension())
        registration.SetMetricFixedMask(dilated_mask)
        
        # ===== 3. ADVANCED TRANSFORM CONFIGURATION =====
        transform_mesh_size = calculate_grid_size(fixed_processed, grid_spacing)
        bspline = sitk.BSplineTransformInitializer(fixed_processed, transform_mesh_size)
        
        # ===== 4. IMPROVED REGISTRATION SETTINGS =====
        registration = sitk.ImageRegistrationMethod()
        registration.SetInitialTransform(bspline, inPlace=False)
        
        # Multi-resolution strategy
        registration.SetShrinkFactorsPerLevel([8, 4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([4.0, 2.0, 1.0, 0.0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # ===== 5. VASCULAR-OPTIMIZED METRIC =====
        registration.SetMetricAsJointHistogramMutualInformation(
            numberOfHistogramBins=64,
            varianceForJointPDFSmoothing=1.5
        )
        registration.SetMetricSamplingStrategy(registration.ADAPTIVE)
        registration.SetMetricSamplingPercentage(0.25)

        # ===== 6. REGULARIZATION =====
        registration.SetMetricUseRegularization(True)
        registration.SetMetricBendingEnergyWeight(0.8)  # Controls surface smoothness
        registration.SetMetricLinearElasticityWeight(0.2)  # Preserves topology

        # ===== 7. OPTIMIZER IMPROVEMENTS =====
        registration.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=1.5,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=20
        )

        # ===== 8. ENHANCED MONITORING =====
        def log_iteration(method):
            if method.GetOptimizerIteration() % 10 == 0:
                log.info(
                    f"Iter {method.GetOptimizerIteration():03d}: "
                    f"Metric={method.GetMetricValue():.4f} "
                    f"LearningRate={method.GetOptimizerLearningRate():.3f}"
                )
        registration.AddCommand(sitk.sitkIterationEvent, log_iteration)

        # ===== 9. EXECUTE REGISTRATION =====
        final_transform = registration.Execute(fixed_processed, moving_processed)
        
        # ===== 10. POST-REGISTRATION ANALYSIS =====
        transformed_image = sitk.Resample(moving_image, fixed_image, final_transform)
        analyze_alignment_quality(fixed_image, transformed_image, dilated_mask)

        return transformed_image, final_transform

    except Exception as e:
        log.error(f"Registration failed: {str(e)}")
        return None, None

# New helper functions -------------------------------------------------

def vessel_enhancement_filter(image):
    """Enhance vascular structures in MRA images."""
    # Multi-scale vesselness filtering
    vessel_filter = sitk.HessianToObjectnessMeasureImageFilter()
    vessel_filter.SetObjectDimension(1)
    vessel_filter.SetBrightObject(True)
    
    enhanced = sitk.Image(image.GetSize(), sitk.sitkFloat32)
    enhanced.CopyInformation(image)
    
    for sigma in [0.5, 1.0, 1.5]:
        smoothed = sitk.RecursiveGaussian(image, sigma=sigma)
        vessels = vessel_filter.Execute(smoothed)
        enhanced = sitk.Add(enhanced, vessels)
    
    # Combine with original image
    return sitk.Cast(image, sitk.sitkFloat32) + enhanced

def anatomical_preservation_filter(image):
    """Preprocess MRI to preserve anatomical structures."""
    # Edge-preserving smoothing
    smoothed = sitk.CurvatureFlow(
        image1=image,
        timeStep=0.125,
        numberOfIterations=5
    )
    
    # Adaptive intensity windowing
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    return sitk.IntensityWindowing(
        smoothed,
        windowMinimum=stats.GetMean() - stats.GetSigma(),
        windowMaximum=stats.GetMean() + 3*stats.GetSigma(),
        outputMinimum=0.0,
        outputMaximum=1.0
    )

def analyze_alignment_quality(fixed, moved, mask):
    """Quantitative alignment assessment."""
    # Calculate Normalized Cross-Correlation
    ncc = sitk.NormalizedCorrelationImageFilter()
    ncc_value = ncc.Execute(fixed, moved, mask)
    log.info(f"Alignment NCC: {ncc_value:.3f}")

    # Calculate deformation field magnitude
    deformation_field = sitk.TransformToDisplacementField(
        sitk.Transform(),
        sitk.sitkVectorFloat64,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection()
    )
    
    mag_filter = sitk.VectorMagnitudeImageFilter()
    deformation_magnitude = mag_filter.Execute(deformation_field)
    
    stats = sitk.StatisticsImageFilter()
    stats.Execute(deformation_magnitude)
    log.info(f"Deformation Stats - Max: {stats.GetMaximum():.2f}mm, Mean: {stats.GetMean():.2f}mm")
    
def save_image(image, file_path):
    """Save a SimpleITK image to a NIfTI file."""
    try:
        if image is None:
            raise ValueError("Cannot save None image")
        
        if image.GetPixelID() not in [sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32]:
            image = sitk.Cast(image, sitk.sitkFloat32)
            
        # Validate image before saving
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        log.info(f"Saving image stats:")
        log.info(f"Size: {image.GetSize()}")
        log.info(f"Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        log.info(f"Mean: {stats.GetMean()}")
        
        # Ensure the image is in a proper format for saving
        if stats.GetMaximum() <= 1.0 and stats.GetMinimum() >= -1.0:
            # If normalized, scale to 0-4095 range for better visualization
            log.info("Rescaling normalized values to 0-4095 range...")
            rescaler = sitk.RescaleIntensityImageFilter()
            rescaler.SetOutputMinimum(0)
            rescaler.SetOutputMaximum(4095)
            image = rescaler.Execute(image)
            
            # Cast to Float 32-bit
            image = sitk.Cast(image, sitk.sitkFloat32)
        
        sitk.WriteImage(image, file_path)
        log.info(f"Successfully saved image to {file_path}")
    except Exception as e:
        log.info(f"Error saving image: {str(e)}")
        log.info("Attempting to save with basic settings...")
        try:
            # Try basic save without any preprocessing
            sitk.WriteImage(image, file_path)
            log.info(f"Successfully saved image to {file_path}")
        except Exception as e2:
            log.info(f"Final error saving image: {str(e2)}")
            sys.exit(1)

def n4bias_field_preprocess(image, source_image_path, image_type):
    """Preprocess image with N4Bias Field Correction"""
    source_dir = os.path.dirname(source_image_path)
    output_dir = os.path.join(source_dir, "preprocess_image")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"n4bias_preprocess_{image_type}.nii.gz")
    
    # If file already exists, load and return it
    if os.path.exists(output_path):
        log.info(f"Preprocessed image already exists at {output_path}, loading...")
        return sitk.ReadImage(output_path)
    # Otherwise, perform N4 bias field correction
    log.info(f"Performing N4Bias Field Correction on {image_type}...")
    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_filter.SetMaximumNumberOfIterations([10, 10, 10])
    bias_field_filter.SetConvergenceThreshold(1e-6)
    preprocess_image = bias_field_filter.Execute(image)
    
    # Save the preprocessed image
    save_image(preprocess_image, output_path)
    return preprocess_image

def read_centroid_locations(centroid_locations_file_path, patient_mrn, mri_image):
    """Read and convert centroid coordinates to match SimpleITK/ITK-SNAP's coordinate system"""
    # Get image dimensions
    mri_array = sitk.GetArrayFromImage(mri_image)  # Shape: (Z, Y, X)
    size_z, size_y, size_x = mri_array.shape
    log.info(f"MRI Image dimensions - Z (axial): {size_z}, Y (coronal): {size_y}, X (sagittal): {size_x}")
    
    centroids = []
    with open(centroid_locations_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        
        for row in reader:
            if str(row[0]).zfill(8) == str(patient_mrn).zfill(8):
                for i in range(2):  # Process up to 2 centroids per row
                    try:
                        start_idx = 1 + i*3
                        # Read raw coordinates from CSV
                        sagittal = row[start_idx].strip()    # X coordinate
                        coronal = row[start_idx+1].strip()   # Y coordinate 
                        axial = row[start_idx+2].strip()     # Z coordinate
                        
                        # Skip invalid entries
                        if any(val in ['', '#N/A'] for val in [sagittal, coronal, axial]):
                            continue

                        # Convert coordinates with proper flipping
                        x = int(float(sagittal))             # X: Sagittal (no flip)
                        y = int(float(coronal))              # Y: Coronal (no flip)
                        z = size_z - int(float(axial)) - 1   # Z: Axial (flip)
                        
                        # Validate bounds
                        if (0 <= z < size_z and 
                            0 <= y < size_y and 
                            0 <= x < size_x):
                            centroids.append((z, y, x))  # Store as (Z,Y,X) for numpy array
                            log.info(f"Converted centroid {i+1}: CSV ({sagittal},{coronal},{axial}) → SITK ({z},{y},{x})")
                        else:
                            log.warning(f"Out-of-bounds centroid {i+1}: CSV ({sagittal},{coronal},{axial}) → SITK ({z},{y},{x})")

                    except (ValueError, IndexError, TypeError) as e:
                        log.warning(f"Error processing centroid {i+1}: {str(e)}")
                        continue
                break
    
    log.info(f"Found {len(centroids)} valid centroids for MRN {patient_mrn}")
    return centroids

def ants_brain_extraction(input_path, output_path, image_modality="t1"):
    """
    Perform brain extraction using ANTsPyNet's method.
    
    Parameters:
    - input_path: path to input MRI (.nii.gz)
    - output_path: path to save brain-extracted result (.nii.gz)
    - image_modality: mr imaging modality (t1, t2, etc.)
    
    Returns:
    - skull-stripped SimpleITK image
    """
    try:
        log.info(f"Running ANTsPyNet brain extraction on {input_path}...")
        
        # Load image using ANTs
        image_ants = ants.image_read(input_path)
        
        # Perform brain extraction using antspynet
        probability_mask = antspynet_brain_extraction(image_ants, modality=image_modality)
        
        # Threshold probability mask
        brain_mask = ants.threshold_image(probability_mask, 0.5, 1.0, 1, 0)
        
        # Apply mask to original image
        brain = ants.mask_image(image_ants, brain_mask)
        
        # Save result
        ants.image_write(brain, output_path)
        log.info(f"Brain-extracted image saved to {output_path}")
        
        # Read back with SimpleITK for further processing
        return sitk.ReadImage(output_path)
    except Exception as e:
        log.error(f"ANTs brain extraction failed: {str(e)}")
        log.info("Falling back to original image")
        return sitk.ReadImage(input_path)
    
@click.command()
@click.option('--mri', required=True, type=click.Path(exists=True), help='Path to the reference MRI file (.nii.gz)')
@click.option('--mra', required=True, type=click.Path(exists=True), help='Path to the moving MRA file (.nii.gz)')
@click.option('--output', required=True, type=click.Path(), help='Path to save the aligned MRA file')
@click.option('--patient-mrn', required=True, type=str)
@click.option('--initial-transform', default='euler',
              type=click.Choice(['euler', 'similarity', 'versor', 'affine', 'bspline']),
              help='Type of transform to use for initial registration')
@click.option('--initial-spacing', default=50.0, type=float,
              help='B-spline control point spacing in mm for initial registration')
@click.option('--final-spacing', default=30.0, type=float,
              help='B-spline control point spacing in mm for final registration')
@click.option("--centroid-locations", required=True, type=click.Path(exists=True), help="Path to csv file containing locations of centroids")
@click.option('--brain-extraction', is_flag=True, default=False,
              help='Enable AntsPy brain extraction on MRI (default: disabled)')
@click.option('--logfile', default=None, type=click.Path(), help='Optional path to log output to a file')
def main(mri, mra, output, patient_mrn, initial_transform, initial_spacing, final_spacing, centroid_locations, brain_extraction, logfile):
    # Configure logging
    if logfile:
        log_path = logfile
    else:
        # Default: create incremented log in same directory as MRI
        default_log_dir = os.path.dirname(mri)
        log_path = get_incremented_logfile_name(default_log_dir)

    log.basicConfig(filename=log_path, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=log.INFO)
    log.info(f"Logging to file: {log_path}")

    # Load images
    log.info("Loading images...")
    mri_image = load_image(mri)
    mra_image = load_image(mra)
    
    # Ensure both images are in same coordinate reference space
    mri_image = sitk.DICOMOrient(mri_image, "LPS")
    mra_image = sitk.DICOMOrient(mra_image, "LPS")
    
    # Read in centroids of Trigeminal Nerve
    centroids_list = read_centroid_locations(centroid_locations, patient_mrn, mri_image)
    if not centroids_list:
        log.warning(f"No centroids found for patient MRN {patient_mrn}")
    else:
        log.info(f"Loaded {len(centroids_list)} centroid(s) for patient MRN {patient_mrn}")
        for idx, centroid in enumerate(centroids_list):
            log.info(f"Centroid {idx + 1}: {centroid}")
    
    # Resample MRA to match MRI resolution
    mra_resampled = resample_to_reference(mra_image, mri_image)
    
    # Run N4BiasField Correction (MRI → skull-included)
    preprocess_mri_path = os.path.join(os.path.dirname(mri), "preprocess_image", "n4bias_preprocess_mri.nii.gz")
    preprocess_mri_image = n4bias_field_preprocess(mri_image, mri, "mri")

    # Run brain extraction on preprocessed MRI
    bet_output_dir = os.path.join(os.path.dirname(mri), "preprocess_image")
    
    # Perform brain extraction only if requested
    if brain_extraction:
        log.info("Brain extraction enabled. Running ANTsPy Brain Extraction...")
        ants_bet_path = os.path.join(bet_output_dir, "ants_bet.nii.gz")
        brain_mri_image = ants_brain_extraction(preprocess_mri_path, ants_bet_path, image_modality="t2")
    else:
        log.info("Brain extraction disabled. Using N4-corrected skull-included MRI.")
        brain_mri_image = preprocess_mri_image

    # Preprocess MRA
    preprocess_mra_image = n4bias_field_preprocess(mra_resampled, mra, "mra")

    # Normalize images
    log.info("Normalizing images...")
    mri_norm = normalize_image(brain_mri_image)
    mra_norm = normalize_image(preprocess_mra_image)
    
    # Perform initial registration on full images
    initial_result, initial_transform = perform_initial_registration(
        mri_norm, mra_norm, 
        transform_type=initial_transform,
        grid_spacing=initial_spacing
    )
    
    if initial_result is None:
        log.info("Initial registration failed. Exiting...")
        return
    
    # Show initial registration result and let user VIEW region
    log.info("\nInitial registration complete. Please view region of final registration and trigeminal centroids...")
    viewer = RegionViewer(mri_norm, initial_result, centroids_list)
    viewer.root.mainloop()

    if viewer.confirmed:
        # Get auto-generated mask from viewer
        log.info("\nCreating registration mask from centroid bounding box...")
        mask = viewer.get_mask()
    else:
        log.info("\nOperation cancelled by user")
        return
    
    # Validate mask
    if not validate_region(mask, "centroid-based"):
        log.info("\nInvalid mask generated from centroids. Exiting...")
        return
    
    # Perform non-rigid registration on selected region
    log.info("\nPerforming final non-rigid registration on selected region...")
    aligned_image, nonrigid_transform = perform_nonrigid_registration_v2(
        mri_norm, initial_result, mask, grid_spacing=final_spacing)
    
    # Going to save the entire aligned image, not just region
    if aligned_image is not None:
        # Save result
        log.info(f"\nSaving aligned image to {output}")
        save_image(aligned_image, output)
        log.info("\nRegistration complete!")
    else:
        log.info("\nFinal registration failed. Please check input images and centroids.")
    
if __name__ == "__main__":
    main()
