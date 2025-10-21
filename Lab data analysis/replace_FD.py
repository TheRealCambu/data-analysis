# import os
# import cv2
# import pytesseract
# from PIL import Image, ImageDraw, ImageFont
#
# # If on Windows, set tesseract path manually
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# root_dir = r"C:\Users\39338\Downloads"
#
# # Font for drawing text
# font = ImageFont.truetype("arial.ttf", 90)
#
# for subdir, _, files in os.walk(root_dir):
#     print(subdir)
#     for file in files:
#         if file.lower().endswith('.png'):
#             img_path = os.path.join(subdir, file)
#             img = cv2.imread(img_path)
#
#             # Run OCR with bounding boxes
#             data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
#
#             # Convert OpenCV image to PIL
#             pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             draw = ImageDraw.Draw(pil_img)
#
#             n_boxes = len(data['text'])
#             replaced = False
#             for i in range(n_boxes):
#                 text = data['text'][i].strip()
#                 if text == "FD":
#                     (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#                     # Cover old text (white rectangle)
#                     draw.rectangle([(x, y), (x + w, y + h)], fill="white")
#                     # Write new text
#                     draw.text((x - 10, y - 25), "Fast timing-square", font=font, fill="black")
#                     replaced = True
#
#             if replaced:
#                 result_path = os.path.join(subdir, f"fixed_{file}")
#                 pil_img.save(result_path)
#                 print(f"✅ Fixed: {result_path}")
#             else:
#                 print(f"⚠️ No 'FD' found in {file}")

import os
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Paths
root_dir = r"C:\Users\39338\Downloads"
font = ImageFont.truetype("arial.ttf", 88)

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Preprocessing function
def preprocess_image(cv_img, kernel_size=2, thresh=180):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)
    return dilated


# Find indexes of "FD"
def find_fd(data):
    indices = []
    for i, txt in enumerate(data['text']):
        if txt.strip().upper() == "FD":
            indices.append(i)
    return indices


# Loop over each subfolder
for subdir_name in os.listdir(root_dir):
    current_folder = os.path.join(root_dir, subdir_name)
    if not os.path.isdir(current_folder):
        continue  # skip files in root_dir

    # Create Fixed folder inside the current subfolder
    fixed_folder = os.path.join(current_folder, "Fixed")
    os.makedirs(fixed_folder, exist_ok=True)

    # Collect PNG images in the current subfolder
    image_files = [f for f in os.listdir(current_folder) if f.lower().endswith(".png")]
    total_images = len(image_files)
    print(f"\nProcessing folder: {subdir_name}")
    print(f"Total images found: {total_images}")

    for file in image_files:
        img_path = os.path.join(current_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        replaced = False
        # Try multiple preprocessing parameters
        for kernel_size in [1, 2, 3, 4]:
            for thresh in range(100, 221, 1):
                proc_img = preprocess_image(img, kernel_size=kernel_size, thresh=thresh)
                data = pytesseract.image_to_data(proc_img, output_type=pytesseract.Output.DICT)
                fd_indices = find_fd(data)
                if fd_indices:
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    for i in fd_indices:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        padding = 0
                        draw.rectangle([(x - padding, y - padding), (x + w + padding, y + h + padding)], fill="white")
                        draw.text((x - 8, y - 21), "Fast square-timing", font=font, fill="black")

                    # Save fixed image in Fixed folder
                    result_path = os.path.join(fixed_folder, file)
                    pil_img.save(result_path)
                    replaced = True
                    print(f"✅ Fixed: {file} (kernel={kernel_size}, thresh={thresh})")
                    break
            if replaced:
                break

        if not replaced:
            print(f"⚠️ Could not find 'FD' in {file} with any preprocessing parameters.")

    fixed_count = len([f for f in os.listdir(fixed_folder) if f.lower().endswith(".png")])
    print(f"Finished processing folder: {subdir_name}. Fixed {fixed_count}/{total_images} images.")
