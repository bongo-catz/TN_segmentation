import SimpleITK as sitk
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import logging as log
import tkinter.messagebox as messagebox

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
        
        # log.info the number of slices included
        log.info(f"\nSelected region includes {self.selected_bounds['z'][1] - self.selected_bounds['z'][0]} slices")
        log.info(f"From slice {self.selected_bounds['z'][0]} to {self.selected_bounds['z'][1]}")
        
    def preview_selection(self):
        if self.selected_bounds is None:
            log.info("Please make a selection first")
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
            log.info("Please select a region before proceeding.")
            return
        self.preview_selection()  # Show final selection
        if tk.messagebox.askyesno("Confirm Selection", 
                                "Is this the correct region you want to align?"):
            self.root.quit()
            self.root.destroy()
        
    def get_selection(self):
        return self.selected_bounds