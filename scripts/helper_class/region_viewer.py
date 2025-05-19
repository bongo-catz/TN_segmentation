import SimpleITK as sitk
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import logging as log

class RegionViewer:
    def __init__(self, mri_image, mra_image, centroids, margin=40):
        self.mri_image = mri_image
        self.mra_image = mra_image
        self.centroids = centroids
        self.margin = margin
        self.confirmed = False
        
        # Get image arrays
        self.mri_array = sitk.GetArrayFromImage(mri_image) # Shape: (Z, Y, X)
        self.mra_array = sitk.GetArrayFromImage(mra_image)
        
        # Calculate bounding box
        self.bounds = self.calculate_bounding_box()
        
        # Create main window with larger size
        self.root = tk.Tk()
        self.root.title("Region Verification")
        self.root.geometry("1000x1000")  # Larger window size
        self.root.update_idletasks()
        
        # Create control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Slice information label
        self.slice_info = ttk.Label(control_frame, text="", font=('Helvetica', 12))
        self.slice_info.pack(side=tk.LEFT, padx=10)
        
        # Slice slider (on the region of the boundaries)
        self.slice_var = tk.IntVar(value=self.bounds['z'][0])
        self.slice_slider = tk.Scale(
            control_frame,
            from_=self.bounds['z'][0],
            to=self.bounds['z'][1],
            orient=tk.HORIZONTAL,
            variable=self.slice_var,
            command=self.update_slice,
            length=400,
            showvalue=False,
            resolution=1  # Ensures discrete steps
        )
        self.slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Button(button_frame, text="Confirm", command=self.confirm, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.cancel, width=10).pack(side=tk.LEFT)
        
        # Create canvas with scrollbars
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial display
        self.update_slice()
        
    def update_slice(self, *args):
        """Update display with current slice for both MRI and MRA"""
        slice_idx = int(self.slice_var.get())

        self.slice_info.config(text=f"Slice: {slice_idx}/{self.mri_array.shape[0]-1} | "
                                    f"Z-range of Bounding Box: {self.bounds['z'][0]}-{self.bounds['z'][1]}")

        # Extract slices
        mri_slice = self.normalize_slice(self.mri_array[slice_idx, :, :])
        mra_slice = self.normalize_slice(self.mra_array[slice_idx, :, :])

        # Convert to PIL images
        mri_img = Image.fromarray(mri_slice).convert('RGB')
        mra_img = Image.fromarray(mra_slice).convert('RGB')

        # Resize both images
        scale_factor = 2
        new_size = (int(mri_img.width * scale_factor), int(mri_img.height * scale_factor))
        mri_img = mri_img.resize(new_size, Image.LANCZOS)
        mra_img = mra_img.resize(new_size, Image.LANCZOS)

        # Draw overlays on MRI image
        draw = ImageDraw.Draw(mri_img)
        scale_coord = lambda x: int(x * scale_factor)

        if self.bounds['z'][0] <= slice_idx <= self.bounds['z'][1]:
            box = [
                scale_coord(self.bounds['x'][0]),
                scale_coord(self.bounds['y'][0]),
                scale_coord(self.bounds['x'][1]),
                scale_coord(self.bounds['y'][1])
            ]
            draw.rectangle(box, outline='red', width=3)

        for c in self.centroids:
            if c[0] == slice_idx:
                x_pos = scale_coord(c[2])
                y_pos = scale_coord(c[1])
                radius = scale_coord(5)
                draw.ellipse([
                    x_pos - radius, y_pos - radius,
                    x_pos + radius, y_pos + radius
                ], fill='red')
                draw.text((x_pos + radius + 5, y_pos),
                        f"({c[2]},{c[1]},{c[0]})",
                        fill="yellow",
                        font=ImageFont.load_default())

        # Stack MRI and MRA images side by side
        combined_img = Image.new('RGB', (new_size[0] * 2, new_size[1]))
        combined_img.paste(mri_img, (0, 0))
        combined_img.paste(mra_img, (new_size[0], 0))

        # Display the combined image
        self.photo = ImageTk.PhotoImage(combined_img)
        self.canvas.config(width=combined_img.width, height=combined_img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Optional: scroll to center
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def calculate_bounding_box(self):
        """Calculate bounding box around centroids with margin"""
        if not self.centroids:
            raise ValueError("No centroids provided for bounding box calculation")
            
        # Get min/max coordinates
        z_coords = [c[0] for c in self.centroids]  # Axial (slice number)
        y_coords = [c[1] for c in self.centroids]  # Coronal
        x_coords = [c[2] for c in self.centroids]  # Sagittal
        
        # Calculate bounds for each axis
        z_min = max(0, min(z_coords) - self.margin)
        z_max = min(self.mri_array.shape[0]-1, max(z_coords) + self.margin)
        
        y_min = max(0, min(y_coords) - self.margin)
        y_max = min(self.mri_array.shape[1]-1, max(y_coords) + self.margin)
        
        x_min = max(0, min(x_coords) - self.margin)
        x_max = min(self.mri_array.shape[2]-1, max(x_coords) + self.margin)

        return {
            'z': (z_min, z_max),
            'y': (y_min, y_max),
            'x': (x_min, x_max)
        }
        
    def normalize_slice(self, slice_data):
        """Normalize slice data to 0-255 with robust handling of edge cases"""
        # Handle NaN values by replacing with minimum
        if np.isnan(slice_data).any():
            slice_data = np.nan_to_num(slice_data, nan=np.nanmin(slice_data))
        
        # Handle case where all values are identical
        if np.all(slice_data == slice_data.flat[0]):
            return np.zeros_like(slice_data, dtype=np.uint8)  # Return all black
        
        # Normalize to 0-255 range
        slice_min = np.min(slice_data)
        slice_range = np.max(slice_data) - slice_min
        
        # Avoid division by zero (though already handled by identical values case)
        if slice_range == 0:
            return np.zeros_like(slice_data, dtype=np.uint8)
        
        normalized = ((slice_data - slice_min) * 255 / slice_range)
        
        # Clip values to ensure they're within 0-255 range before conversion
        return np.clip(normalized, 0, 255).astype(np.uint8)
        
    def confirm(self):
        self.confirmed = True
        self.root.quit()
        self.root.destroy()
        
    def cancel(self):
        self.confirmed = False
        self.root.quit()
        self.root.destroy()
        
    def get_mask(self):
        """Create mask from bounding box"""
        mask_array = np.zeros_like(self.mri_array)
        mask_array[
            self.bounds['z'][0]:self.bounds['z'][1],
            self.bounds['y'][0]:self.bounds['y'][1],
            self.bounds['x'][0]:self.bounds['x'][1]
        ] = 1
        
        mask_image = sitk.GetImageFromArray(mask_array)
        mask_image.CopyInformation(self.mri_image)
        return mask_image