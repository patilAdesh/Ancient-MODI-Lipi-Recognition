import cv2 as cv
import numpy as np
import os
from PIL import Image
from PIL import ImageEnhance, ImageFilter

def generate_synthetic_data(input_folder, output_folder, prefix, num_images=200):
    # Create the output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the transformation parameters
    rotation_angles = [90, 180, 30]
    zoom_factors = [0.5, 0.8, 1.2]
    blur_radius = 2
    unblur_factor = 2
    saturation_factor = 1.5
    contrast_factor = 1.5

    # Loop over all input images in the input folder
    for input_image_path in os.listdir(input_folder):
        # Check if the file is an image file
        if input_image_path.endswith('.jpg') or input_image_path.endswith('.png')or input_image_path.endswith('.jpeg'):
            # Open the input image using the PIL library
            input_image = Image.open(os.path.join(input_folder, input_image_path))

            # Convert the PIL image to a NumPy array
            input_image = np.array(input_image)

            # Loop over the number of output images to generate for this input image
            for i in range(num_images):
                # Apply the transformations
                # Rotate the image
                rotation_angle = rotation_angles[i % len(rotation_angles)]
                rotated_image = Image.fromarray(input_image).rotate(rotation_angle)

                # Zoom the image
                zoom_factor = zoom_factors[i % len(zoom_factors)]
                zoomed_image = rotated_image.resize((int(rotated_image.width * zoom_factor), int(rotated_image.height * zoom_factor)))

                # Apply blur and unblur
                blurred_image = zoomed_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                unblurred_image = blurred_image.filter(ImageFilter.UnsharpMask(radius=blur_radius, percent=unblur_factor))

                # Adjust the image saturation and contrast
                saturated_image = ImageEnhance.Color(unblurred_image).enhance(saturation_factor)
                contrasted_image = ImageEnhance.Contrast(saturated_image).enhance(contrast_factor)

                # Convert the image back to a NumPy array
                output_image = np.array(contrasted_image)

                # Save the output image using OpenCV
                output_image_path = os.path.join(output_folder, f'{os.path.splitext(input_image_path)[0]}_{i}.jpg')
                cv.imwrite(output_image_path, cv.cvtColor(output_image, cv.COLOR_RGB2BGR))

if __name__ == '__main__':
    input_folder = 'C:/Users/patil/Pictures/Dataset/52/'
    output_folder = 'C:/Users/patil/Pictures/Dataset/52/'
    prefix = 'synthetic_image'
    num_images = 200

    generate_synthetic_data(input_folder, output_folder, prefix, num_images)
