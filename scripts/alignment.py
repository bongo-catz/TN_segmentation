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
from helper_class.region_selector import RegionSelector

def load_image(file_path):
    """Load a NIfTI image (.nii.gz) and return a SimpleITK image object."""
    try:
        image = sitk.ReadImage(file_path)
        size = image.GetSize()
        spacing = image.GetSpacing()
        log.info(f"Loaded image {file_path}")
        log.info(f"Size: {size}, Spacing: {spacing}")
        log.info(f"Origin: {image.GetOrigin()}")
        log.info(f"Direction: {image.GetDirection()}")
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
                numberOfIterations=100,
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
            log.info(f"Final transform parameters: {final_transform.GetParameters()}")
        
        # Apply transform
        result = sitk.Resample(moving_image, fixed_image, final_transform)
        
        return result, final_transform
        
    except Exception as e:
        log.info(f"Error during initial registration: {str(e)}")
        return None, None

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
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.01)
        
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer for non-rigid registration
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-4,
            numberOfIterations=50,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=500,
            costFunctionConvergenceFactor=1e6)
        
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
            moving_image.GetPixelID())
        
        # Extract only the region of interest from the transformed image
        transformed_region = sitk.Mask(transformed_image, mask)
        
        return transformed_region, final_transform
        
    except Exception as e:
        log.info(f"Error during non-rigid registration: {str(e)}")
        return None, None

def save_image(image, file_path):
    """Save a SimpleITK image to a NIfTI file."""
    try:
        if image is None:
            raise ValueError("Cannot save None image")
            
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
            
            # Cast to 16-bit integer
            image = sitk.Cast(image, sitk.sitkUInt16)
        
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

@click.command()
@click.option('--mri', required=True, type=click.Path(exists=True), help='Path to the reference MRI file (.nii.gz)')
@click.option('--mra', required=True, type=click.Path(exists=True), help='Path to the moving MRA file (.nii.gz)')
@click.option('--output', required=True, type=click.Path(), help='Path to save the aligned MRA file')
@click.option('--initial-transform', default='euler',
              type=click.Choice(['euler', 'similarity', 'versor', 'affine', 'bspline']),
              help='Type of transform to use for initial registration')
@click.option('--initial-spacing', default=50.0, type=float,
              help='B-spline control point spacing in mm for initial registration')
@click.option('--final-spacing', default=30.0, type=float,
              help='B-spline control point spacing in mm for final registration')
@click.option('--logfile', default=None, type=click.Path(), help='Optional path to log output to a file')
def main(mri, mra, output, initial_transform, initial_spacing, final_spacing, logfile):
    # Configure logging
    if logfile:
        log.basicConfig(filename=logfile, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=log.INFO)
    else:
        log.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=log.INFO)

    
    # Load images
    log.info("Loading images...")
    mri_image = load_image(mri)
    mra_image = load_image(mra)
    
    # Resample MRA to match MRI resolution
    mra_resampled = resample_to_reference(mra_image, mri_image)
    
    # Normalize images
    log.info("Normalizing images...")
    mri_norm = normalize_image(mri_image)
    mra_norm = normalize_image(mra_resampled)
    
    # Perform initial registration on full images
    initial_result, initial_transform = perform_initial_registration(
        mri_norm, mra_norm, 
        transform_type=initial_transform,
        grid_spacing=initial_spacing
    )
    
    if initial_result is None:
        log.info("Initial registration failed. Exiting...")
        return
    
    # Show initial registration result and let user select region
    log.info("\nInitial registration complete. Please select region for final registration...")
    selector = RegionSelector(mri_norm, initial_result)
    selector.root.mainloop()
    
    bounds = selector.get_selection()
    if bounds is None:
        log.info("No region selected. Exiting...")
        return
    
    log.info("\nSelected bounds:", bounds)
    
    # Create mask from selected region
    log.info("\nCreating registration mask...")
    mask = create_mask_from_bounds(mri_norm, bounds)
    
    # Perform non-rigid registration on selected region
    log.info("\nPerforming final non-rigid registration on selected region...")
    aligned_region, nonrigid_transform = perform_nonrigid_registration(
        mri_norm, initial_result, mask, grid_spacing=final_spacing)
    
    if aligned_region is not None:
        # Save result
        log.info(f"\nSaving aligned image to {output}")
        save_image(aligned_region, output)
        log.info("Done!")
    else:
        log.info("Final registration failed. Please try again with a different region selection.")

if __name__ == "__main__":
    main()
