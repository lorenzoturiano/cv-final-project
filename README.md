# Star Removal Model

## Objective

Develop the best model to **remove stars from astronomical images** while:
- Minimizing modifications to non-star areas (ideally no modification at all).
- Removing stars and filling the previously occupied space **plausibly and naturally**.

---

## Dataset Creation

### Option 1: Real Images (Team: Lorenzo, Luca)

1. **Input:**  
   Monochromatic photos with a light band.
   
2. **Preprocessing Steps:**
   - Apply pixel-wise transformations (e.g., `arcsinh`, histogram equalization) to better distinguish stars.
   - Convolve the image (initially Gaussian PSF approximation; real PSF depends on telescope).
   - Compute image gradients (e.g., Sobel filter) and apply thresholding to generate a **star mask**.
   - Post-process the mask (e.g., erosion, closure) to clean it up.
   
3. **Generating Ground Truth:**
   - Apply **inpainting** on the original photo using the mask to remove stars.
   - Manually fix poorly inpainted stars if necessary.

### Option 2 (Optional / Later): Artificial Stars

- Generate synthetic star images and train the model on these controlled datasets.

---

## Model Development

### Option 1: U-Net (Lead: Ric)

- Train a U-Net from scratch.
- Tune hyperparameters.
- Directly output the final cleaned image.

### Option 2: Segmentation + Inpainting

- Use a segmentation model to detect stars.
- Apply inpainting afterward to fill the star positions.

### Option 3: Object Detection + U-Net per Patch

- Detect star locations via object detection.
- Apply a local U-Net-based inpainting model on each detected patch.

---

## Evaluation Metrics

- **Background Modification:**  
  Measure how much non-star regions are changed.
- **Star Removal Rate:**  
  Percentage of stars correctly removed.
- **Image Difference:**  
  Compare original and output images quantitatively.
- **Masked Area Error:**  
  Evaluate the quality of the inpainting specifically in star regions.

---

## Contributors

- Lorenzo
- Luca
- Ric
