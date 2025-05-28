import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Workshop - Xử lý ảnh")
        self.root.geometry("1200x800")
        
        # Biến lưu trữ ảnh
        self.original_image = None
        self.processed_image = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện người dùng"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel (bên trái)
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Image display frame (bên phải)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Nút load ảnh
        ttk.Button(control_frame, text="Chọn ảnh", command=self.load_image).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Reset ảnh gốc", command=self.reset_image).pack(pady=5, fill=tk.X)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Function 1: Color Balance
        self.setup_color_balance(control_frame)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Function 2: Histogram
        self.setup_histogram(control_frame)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Function 3-5: Filters
        self.setup_filters(control_frame)
        
        # Image display area
        self.setup_image_display(image_frame)
        
    def setup_color_balance(self, parent):
        """Function 1: Thiết lập điều khiển Color Balance"""
        color_frame = ttk.LabelFrame(parent, text="1. Color Balance", padding=5)
        color_frame.pack(fill=tk.X, pady=5)
        
        # Red channel
        ttk.Label(color_frame, text="Red:").pack()
        self.red_var = tk.DoubleVar(value=1.0)
        red_scale = ttk.Scale(color_frame, from_=0.1, to=3.0, variable=self.red_var, 
                             orient=tk.HORIZONTAL, command=self.apply_color_balance)
        red_scale.pack(fill=tk.X)
        
        # Green channel
        ttk.Label(color_frame, text="Green:").pack()
        self.green_var = tk.DoubleVar(value=1.0)
        green_scale = ttk.Scale(color_frame, from_=0.1, to=3.0, variable=self.green_var,
                               orient=tk.HORIZONTAL, command=self.apply_color_balance)
        green_scale.pack(fill=tk.X)
        
        # Blue channel
        ttk.Label(color_frame, text="Blue:").pack()
        self.blue_var = tk.DoubleVar(value=1.0)
        blue_scale = ttk.Scale(color_frame, from_=0.1, to=3.0, variable=self.blue_var,
                              orient=tk.HORIZONTAL, command=self.apply_color_balance)
        blue_scale.pack(fill=tk.X)
        
    def setup_histogram(self, parent):
        """Function 2: Thiết lập điều khiển Histogram"""
        hist_frame = ttk.LabelFrame(parent, text="2. Histogram", padding=5)
        hist_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(hist_frame, text="Hiển thị Histogram", 
                  command=self.show_histogram).pack(fill=tk.X, pady=2)
        ttk.Button(hist_frame, text="Histogram Equalization", 
                  command=self.apply_histogram_equalization).pack(fill=tk.X, pady=2)
        
    def setup_filters(self, parent):
        """Function 3-5: Thiết lập các bộ lọc"""
        filter_frame = ttk.LabelFrame(parent, text="3-5. Filters", padding=5)
        filter_frame.pack(fill=tk.X, pady=5)
        
        # Median Filter
        median_frame = ttk.Frame(filter_frame)
        median_frame.pack(fill=tk.X, pady=2)
        ttk.Label(median_frame, text="Median Filter Size:").pack(side=tk.LEFT)
        self.median_var = tk.IntVar(value=5)
        median_spin = ttk.Spinbox(median_frame, from_=3, to=15, increment=2, 
                                 textvariable=self.median_var, width=5)
        median_spin.pack(side=tk.RIGHT)
        ttk.Button(filter_frame, text="Apply Median Filter", 
                  command=self.apply_median_filter).pack(fill=tk.X, pady=2)
        
        # Mean Filter
        mean_frame = ttk.Frame(filter_frame)
        mean_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mean_frame, text="Mean Filter Size:").pack(side=tk.LEFT)
        self.mean_var = tk.IntVar(value=5)
        mean_spin = ttk.Spinbox(mean_frame, from_=3, to=15, increment=2, 
                               textvariable=self.mean_var, width=5)
        mean_spin.pack(side=tk.RIGHT)
        ttk.Button(filter_frame, text="Apply Mean Filter", 
                  command=self.apply_mean_filter).pack(fill=tk.X, pady=2)
        
        # Gaussian Filter
        gauss_frame = ttk.Frame(filter_frame)
        gauss_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gauss_frame, text="Gaussian Sigma:").pack(side=tk.LEFT)
        self.gauss_var = tk.DoubleVar(value=1.0)
        gauss_spin = ttk.Spinbox(gauss_frame, from_=0.1, to=5.0, increment=0.1, 
                                textvariable=self.gauss_var, width=5)
        gauss_spin.pack(side=tk.RIGHT)
        ttk.Button(filter_frame, text="Apply Gaussian Smoothing", 
                  command=self.apply_gaussian_filter).pack(fill=tk.X, pady=2)
        
        # Add noise button for testing
        ttk.Separator(filter_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Button(filter_frame, text="Thêm Salt & Pepper Noise", 
                  command=self.add_noise).pack(fill=tk.X, pady=2)
        
    def setup_image_display(self, parent):
        """Thiết lập khu vực hiển thị ảnh"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Original image tab
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="Ảnh gốc")
        
        # Processed image tab
        self.processed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.processed_frame, text="Ảnh đã xử lý")
        
        # Histogram tab
        self.histogram_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.histogram_frame, text="Histogram")
        
        # Image labels
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(expand=True)
        
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(expand=True)
        
    def load_image(self):
        """Tải ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.current_image = self.original_image.copy()
                self.processed_image = self.original_image.copy()
                self.display_images()
                # Reset sliders
                self.red_var.set(1.0)
                self.green_var.set(1.0)
                self.blue_var.set(1.0)
            else:
                messagebox.showerror("Lỗi", "Không thể tải ảnh!")
                
    def reset_image(self):
        """Reset về ảnh gốc"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.processed_image = self.original_image.copy()
            self.display_images()
            # Reset sliders
            self.red_var.set(1.0)
            self.green_var.set(1.0)
            self.blue_var.set(1.0)
            
    def display_images(self):
        """Hiển thị ảnh trên giao diện"""
        if self.original_image is not None:
            # Display original image
            orig_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            orig_pil = Image.fromarray(orig_rgb)
            orig_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
            orig_photo = ImageTk.PhotoImage(orig_pil)
            self.original_label.configure(image=orig_photo)
            self.original_label.image = orig_photo
            
        if self.processed_image is not None:
            # Display processed image
            proc_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            proc_pil = Image.fromarray(proc_rgb)
            proc_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
            proc_photo = ImageTk.PhotoImage(proc_pil)
            self.processed_label.configure(image=proc_photo)
            self.processed_label.image = proc_photo
            
    def apply_color_balance(self, event=None):
        """Function 1: Áp dụng color balance"""
        if self.current_image is None:
            return
            
        # Lấy giá trị từ sliders
        red_factor = self.red_var.get()
        green_factor = self.green_var.get()
        blue_factor = self.blue_var.get()
        
        # Áp dụng color balance
        balanced = self.current_image.astype(np.float32)
        balanced[:, :, 0] *= blue_factor   # Blue channel
        balanced[:, :, 1] *= green_factor  # Green channel
        balanced[:, :, 2] *= red_factor    # Red channel
        
        # Clamp values to [0, 255]
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        
        self.processed_image = balanced
        self.display_images()
        
    def show_histogram(self):
        """Function 2: Hiển thị histogram"""
        if self.processed_image is None:
            return
            
        # Clear previous histogram
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()
            
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Histogram Analysis')
        
        # Calculate histograms
        colors = ['blue', 'green', 'red']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
            axes[0, 0].plot(hist, color=color, alpha=0.7)
        axes[0, 0].set_title('RGB Histogram')
        axes[0, 0].set_xlabel('Pixel Intensity')
        axes[0, 0].set_ylabel('Frequency')
        
        # Grayscale histogram
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axes[0, 1].plot(hist_gray, color='black')
        axes[0, 1].set_title('Grayscale Histogram')
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].set_ylabel('Frequency')
        
        # Display images
        axes[1, 0].imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Current Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gray, cmap='gray')
        axes[1, 1].set_title('Grayscale Image')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.histogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def apply_histogram_equalization(self):
        """Function 2: Áp dụng histogram equalization"""
        if self.current_image is None:
            return
            
        # Convert to LAB color space for better equalization
        lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        self.processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.current_image = self.processed_image.copy()
        self.display_images()
        
    def apply_median_filter(self):
        """Function 3: Áp dụng median filter để loại bỏ salt & pepper noise"""
        if self.current_image is None:
            return
            
        kernel_size = self.median_var.get()
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
            
        self.processed_image = cv2.medianBlur(self.current_image, kernel_size)
        self.current_image = self.processed_image.copy()
        self.display_images()
        
    def apply_mean_filter(self):
        """Function 4: Áp dụng mean filter"""
        if self.current_image is None:
            return
            
        kernel_size = self.mean_var.get()
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
        self.current_image = self.processed_image.copy()
        self.display_images()
        
    def apply_gaussian_filter(self):
        """Function 5: Áp dụng Gaussian smoothing"""
        if self.current_image is None:
            return
            
        sigma = self.gauss_var.get()
        # Calculate kernel size based on sigma (rule of thumb: 6*sigma + 1)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.processed_image = cv2.GaussianBlur(self.current_image, 
                                              (kernel_size, kernel_size), sigma)
        self.current_image = self.processed_image.copy()
        self.display_images()
        
    def add_noise(self):
        """Thêm salt and pepper noise để test các filter"""
        if self.current_image is None:
            return
            
        noise = np.random.random(self.current_image.shape[:2])
        
        # Salt noise (white pixels)
        salt = noise < 0.05
        # Pepper noise (black pixels)  
        pepper = noise > 0.95
        
        noisy_image = self.current_image.copy()
        noisy_image[salt] = 255
        noisy_image[pepper] = 0
        
        self.processed_image = noisy_image
        self.current_image = noisy_image.copy()
        self.display_images()

def main():
    """Hàm main chạy ứng dụng"""
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()