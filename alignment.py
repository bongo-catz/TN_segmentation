import SimpleITK as sitk
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import sys

def load_image(file_path):
    """Load a NIfTI image (.nii.gz) and return a SimpleITK image object."""
    try:
        image = sitk.ReadImage(file_path)
        size = image.GetSize()
        spacing = image.GetSpacing()
        print(f"Loaded image {file_path}")
        print(f"Size: {size}, Spacing: {spacing}")
        print(f"Origin: {image.GetOrigin()}")
        print(f"Direction: {image.GetDirection()}")
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        sys.exit(1)

def resample_to_reference(moving_image, reference_image):
    """Resample moving image to match reference image resolution."""
    print("Resampling image to reference space...")
    print(f"Original size: {moving_image.GetSize()}")
    print(f"Original spacing: {moving_image.GetSpacing()}")
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    
    resampled = resampler.Execute(moving_image)
    print(f"Resampled size: {resampled.GetSize()}")
    print(f"Resampled spacing: {resampled.GetSpacing()}")
    return resampled

def normalize_image(image):
    """Normalize image intensities to mean 0 and variance 1."""
    try:
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        print(f"Before normalization - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        
        normalized = sitk.Normalize(image)
        
        stats.Execute(normalized)
        print(f"After normalization - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        return normalized
    except Exception as e:
        print(f"Error in normalization: {str(e)}")
        sys.exit(1)

def validate_region(region, name=""):
    """Validate that a region is not empty or invalid."""
    try:
        size = region.GetSize()
        if any(s <= 0 for s in size):
            print(f"Invalid {name} region size: {size}")
            return False
        
        stats = sitk.StatisticsImageFilter()
        stats.Execute(region)
        print(f"{name} region stats:")
        print(f"Size: {size}")
        print(f"Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        print(f"Mean: {stats.GetMean()}")
        return True
    except Exception as e:
        print(f"Error validating {name} region: {str(e)}")
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
    
    print(f"Using {transform_type} transform with {len(initial_transform.GetParameters())} parameters")
    return initial_transform

class RegionSelector:
    def __init__(self, mri_image, mra_image):
        self.mri_image = mri_image
        self.mra_image = mra_image
        self.selected_bounds = None
        
        # Get image arrays and convert to numpy for display
        self.mri_array = sitk.GetArrayFromImage(mri_image)
        self.mra_array = sitk.GetArrayFromImage(mra_image)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Region Selection - Axial View")
        
        # Create frame for controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Image selection
        self.view_var = tk.StringVar(value="MRI")
        ttk.Radiobutton(control_frame, text="MRI", variable=self.view_var, 
                       value="MRI", command=self.update_slice).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="MRA", variable=self.view_var, 
                       value="MRA", command=self.update_slice).pack(side=tk.LEFT)
        
        # Slice slider
        self.slice_var = tk.IntVar(value=self.mri_array.shape[0]//2)
        slice_label = ttk.Label(control_frame, text="Axial Slice:")
        slice_label.pack(side=tk.LEFT, padx=5)
        
        self.slice_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=self.mri_array.shape[0]-1,
            orient=tk.HORIZONTAL,
            variable=self.slice_var,
            command=self.update_slice
        )
        self.slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_slice = None
        
        # Preview button
        preview_button = ttk.Button(control_frame, text="Preview Selection", 
                                  command=self.preview_selection)
        preview_button.pack(side=tk.RIGHT, padx=5)
        
        # Done button
        done_button = ttk.Button(control_frame, text="Done", command=self.finish_selection)
        done_button.pack(side=tk.RIGHT)
        
        # Display initial slice
        self.update_slice()
        
    def update_slice(self, *args):
        # Get current image array based on selection
        if self.view_var.get() == "MRI":
            current_array = self.mri_array
        else:
            current_array = self.mra_array
            
        # Extract the current axial slice
        slice_idx = self.slice_var.get()
        self.current_slice = slice_idx
        image_slice = current_array[slice_idx, :, :]
        
        # Normalize to 0-255 for display
        image_slice = ((image_slice - image_slice.min()) * 255 / 
                      (image_slice.max() - image_slice.min())).astype(np.uint8)
        
        # Convert to PIL Image
        self.current_image = Image.fromarray(image_slice)
        self.photo = ImageTk.PhotoImage(self.current_image)
        
        # Update canvas
        self.canvas.config(width=self.current_image.width, height=self.current_image.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Redraw selection if exists
        if self.rect_id and self.start_x is not None:
            self.rect_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, 
                self.last_x, self.last_y, 
                outline='red', width=2
            )
    
    def start_selection(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.last_x = event.x
        self.last_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
    def update_selection(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.last_x = event.x
        self.last_y = event.y
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline='red', width=2
        )
        
    def end_selection(self, event):
        # Get coordinates in correct order
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        z = self.current_slice
        
        # Store selected bounds (in image coordinates)
        self.selected_bounds = {
            'y': (int(x1), int(x2)),  # x in display becomes y in image
            'x': (int(y1), int(y2)),  # y in display becomes x in image
            'z': (max(0, z-20), min(self.mri_array.shape[0]-1, z+20))  # ±20 slices around selection
        }
        
        # Print the number of slices included
        print(f"\nSelected region includes {self.selected_bounds['z'][1] - self.selected_bounds['z'][0]} slices")
        print(f"From slice {self.selected_bounds['z'][0]} to {self.selected_bounds['z'][1]}")
        
    def preview_selection(self):
        if self.selected_bounds is None:
            print("Please make a selection first")
            return
            
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Get the middle slice of the selection
        z_mid = (self.selected_bounds['z'][0] + self.selected_bounds['z'][1]) // 2
        
        # Show MRI with selection
        mri_slice = self.mri_array[z_mid, :, :]
        ax1.imshow(mri_slice, cmap='gray')
        ax1.add_patch(plt.Rectangle(
            (self.selected_bounds['y'][0], self.selected_bounds['x'][0]),
            self.selected_bounds['y'][1] - self.selected_bounds['y'][0],
            self.selected_bounds['x'][1] - self.selected_bounds['x'][0],
            fill=False, color='red'
        ))
        ax1.set_title('MRI Selection')
        
        # Show MRA with selection
        mra_slice = self.mra_array[z_mid, :, :]
        ax2.imshow(mra_slice, cmap='gray')
        ax2.add_patch(plt.Rectangle(
            (self.selected_bounds['y'][0], self.selected_bounds['x'][0]),
            self.selected_bounds['y'][1] - self.selected_bounds['y'][0],
            self.selected_bounds['x'][1] - self.selected_bounds['x'][0],
            fill=False, color='red'
        ))
        ax2.set_title('MRA Selection')
        
        plt.show()
        
    def finish_selection(self):
        if self.selected_bounds is None:
            print("Please select a region before proceeding.")
            return
        self.preview_selection()  # Show final selection
        if tk.messagebox.askyesno("Confirm Selection", 
                                "Is this the correct region you want to align?"):
            self.root.quit()
            self.root.destroy()
        
    def get_selection(self):
        return self.selected_bounds

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
    print("Mask statistics:")
    print(f"Size: {mask.GetSize()}")
    print(f"Sum of mask (number of selected voxels): {mask_stats.GetSum()}")
    print(f"Selected region dimensions: {bounds['z'][1]-bounds['z'][0]} x {bounds['x'][1]-bounds['x'][0]} x {bounds['y'][1]-bounds['y'][0]}")
    
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
        print("\nPerforming initial registration on full images...")
        print(f"Using transform type: {transform_type}")
        
        registration = sitk.ImageRegistrationMethod()
        
        if transform_type == 'bspline':
            # Calculate grid size from desired physical spacing
            transform_mesh_size = calculate_grid_size(fixed_image, grid_spacing)
            initial_transform = sitk.BSplineTransformInitializer(fixed_image, transform_mesh_size)
            
            actual_spacing = calculate_grid_spacing(fixed_image, transform_mesh_size[0])
            print(f"B-spline transform settings:")
            print(f"Requested grid spacing: {grid_spacing}mm")
            print(f"Actual grid spacing: {actual_spacing[0]:.1f}mm")
            print(f"Grid size: {transform_mesh_size[0]}x{transform_mesh_size[1]}x{transform_mesh_size[2]} control points")
            print(f"Image size: {fixed_image.GetSize()}")
            print(f"Image spacing: {fixed_image.GetSpacing()}")
            
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
            
            print(f"Using {transform_type} transform with {len(initial_transform.GetParameters())} parameters")
            
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
            print(f"Iteration {method.GetOptimizerIteration()}: "
                  f"Metric = {method.GetMetricValue():.6f}")
            
        registration.AddCommand(sitk.sitkIterationEvent, 
                              lambda: command_iteration(registration))
        
        # Perform registration
        final_transform = registration.Execute(fixed_image, moving_image)
        
        if transform_type != 'bspline':
            print("Final transform parameters:", final_transform.GetParameters())
        
        # Apply transform
        result = sitk.Resample(moving_image, fixed_image, final_transform)
        
        return result, final_transform
        
    except Exception as e:
        print(f"Error during initial registration: {str(e)}")
        return None, None

def perform_nonrigid_registration(fixed_image, moving_image, mask, grid_spacing=30):
    """Perform non-rigid registration on masked region."""
    try:
        print("\nPerforming non-rigid registration on selected region...")
        
        # Calculate grid size from desired physical spacing
        transform_mesh_size = calculate_grid_size(fixed_image, grid_spacing)
        print(f"Using B-spline transform with grid spacing: {grid_spacing}mm")
        print(f"Grid size: {transform_mesh_size[0]}x{transform_mesh_size[1]}x{transform_mesh_size[2]} control points")
        
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
            print(f"Iteration {method.GetOptimizerIteration()}: "
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
        print(f"Error during non-rigid registration: {str(e)}")
        return None, None

def save_image(image, file_path):
    """Save a SimpleITK image to a NIfTI file."""
    try:
        if image is None:
            raise ValueError("Cannot save None image")
            
        # Validate image before saving
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        print(f"Saving image stats:")
        print(f"Size: {image.GetSize()}")
        print(f"Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
        print(f"Mean: {stats.GetMean()}")
        
        # Ensure the image is in a proper format for saving
        if stats.GetMaximum() <= 1.0 and stats.GetMinimum() >= -1.0:
            # If normalized, scale to 0-4095 range for better visualization
            print("Rescaling normalized values to 0-4095 range...")
            rescaler = sitk.RescaleIntensityImageFilter()
            rescaler.SetOutputMinimum(0)
            rescaler.SetOutputMaximum(4095)
            image = rescaler.Execute(image)
            
            # Cast to 16-bit integer
            image = sitk.Cast(image, sitk.sitkUInt16)
        
        sitk.WriteImage(image, file_path)
        print(f"Successfully saved image to {file_path}")
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        print("Attempting to save with basic settings...")
        try:
            # Try basic save without any preprocessing
            sitk.WriteImage(image, file_path)
            print(f"Successfully saved image to {file_path}")
        except Exception as e2:
            print(f"Final error saving image: {str(e2)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Align MRA to MRI using normalized mutual information')
    parser.add_argument('--mri', required=True, help='Path to the reference MRI file (.nii.gz)')
    parser.add_argument('--mra', required=True, help='Path to the moving MRA file (.nii.gz)')
    parser.add_argument('--output', required=True, help='Path to save the aligned MRA file')
    parser.add_argument('--initial-transform', 
                      choices=['euler', 'similarity', 'versor', 'affine', 'bspline'],
                      default='euler', 
                      help='Type of transform to use for initial registration (default: euler)')
    parser.add_argument('--initial-spacing', type=float, default=50.0,
                      help='B-spline control point spacing in mm for initial registration (default: 50mm)')
    parser.add_argument('--final-spacing', type=float, default=30.0,
                      help='B-spline control point spacing in mm for final registration (default: 30mm)')
    args = parser.parse_args()
    
    # Load images
    print("Loading images...")
    mri_image = load_image(args.mri)
    mra_image = load_image(args.mra)
    
    # Resample MRA to match MRI resolution
    mra_resampled = resample_to_reference(mra_image, mri_image)
    
    # Normalize images
    print("Normalizing images...")
    mri_norm = normalize_image(mri_image)
    mra_norm = normalize_image(mra_resampled)
    
    # Perform initial registration on full images
    initial_result, initial_transform = perform_initial_registration(
        mri_norm, mra_norm, 
        transform_type=args.initial_transform,
        grid_spacing=args.initial_spacing
    )
    
    if initial_result is None:
        print("Initial registration failed. Exiting...")
        return
    
    # Show initial registration result and let user select region
    print("\nInitial registration complete. Please select region for final registration...")
    selector = RegionSelector(mri_norm, initial_result)
    selector.root.mainloop()
    
    bounds = selector.get_selection()
    if bounds is None:
        print("No region selected. Exiting...")
        return
    
    print("\nSelected bounds:", bounds)
    
    # Create mask from selected region
    print("\nCreating registration mask...")
    mask = create_mask_from_bounds(mri_norm, bounds)
    
    # Perform non-rigid registration on selected region
    print("\nPerforming final non-rigid registration on selected region...")
    aligned_region, nonrigid_transform = perform_nonrigid_registration(
        mri_norm, initial_result, mask, grid_spacing=args.final_spacing)
    
    if aligned_region is not None:
        # Save result
        print(f"\nSaving aligned image to {args.output}")
        save_image(aligned_region, args.output)
        print("Done!")
    else:
        print("Final registration failed. Please try again with a different region selection.")

if __name__ == "__main__":
    main()
