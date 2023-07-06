import imgaug.augmenters as iaa
import cv2
import os

# Specify the path to the directory containing the images
input_dir = "/Users/macbook/Desktop/project/train/images"
output_dir = "/Users/macbook/Desktop/project/train/new_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the augmentation sequence
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip images horizontally with a 50% probability
    iaa.GaussianBlur(sigma=(0, 3.0)),  # Apply Gaussian blur with a sigma between 0 and 3.0
    iaa.Affine(rotate=(-20, 20)),  # Rotate images between -20 and 20 degrees
    iaa.Resize({"height": 224, "width": 224}),  # Resize images to 224x224
])

# Loop through the images in the input directory
for image_file in os.listdir(input_dir):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        # Read the image
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        # Apply augmentation to the image
        augmented_image = augmentation(image=image)

        # Save the augmented image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, augmented_image)

        print(f"Augmented image saved: {output_path}")