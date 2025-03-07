import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 🏷️ Define Paths
TRAIN_DIR = "C:/xampp/htdocs/skin_cancer_app/dataset/train"
AUGMENTED_DIR = "C:/xampp/htdocs/skin_cancer_app/dataset_augmented"
TARGET_SAMPLES = 462  # Match the highest class count for balance

# 🔄 Data Augmentation Settings
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values (0 to 1)
    rotation_range=30,  # Rotate up to 30 degrees
    width_shift_range=0.2,  # Shift horizontally
    height_shift_range=0.2,  # Shift vertically
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Flip horizontally
    fill_mode="nearest",  # Fill missing pixels
)

# 🖼️ Augment & Save Images for Underrepresented Classes
def augment_images(class_name):
    input_dir = os.path.join(TRAIN_DIR, class_name)
    output_dir = os.path.join(AUGMENTED_DIR, class_name)
    os.makedirs(output_dir, exist_ok=True)

    existing_images = len(os.listdir(input_dir))
    images_needed = TARGET_SAMPLES - existing_images

    print(f"Processing category: {class_name}")
    print(f"Existing images: {existing_images}, Needed: {images_needed}")

    if images_needed > 0:
        print(f"Augmenting '{class_name}': Adding {images_needed} images")

        count = 0  # Track number of images generated
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            print(f"Image file: {img_path}")  # Debugging

            image = cv2.imread(img_path)

            if image is None:
                print(f"Skipping corrupt or unreadable image: {img_path}")
                continue

            image = cv2.resize(image, (150, 150))  # Resize for model input
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            i = 0  # Counter for each image augmentation
            for batch in datagen.flow(
                image, batch_size=1, save_to_dir=output_dir, save_prefix="aug", save_format="jpg"
            ):
                print(f"Saving augmented image {count+1}/{images_needed} as aug_{count}.jpg")  
                count += 1
                i += 1
                if count >= images_needed:  # Stop when required images are generated
                    return

# 🎯 Augment Only the Minority Classes
minority_classes = ["nevus"]  # Add more if needed
for cls in minority_classes:
    augment_images(cls)

# 📊 Check Dataset Distribution After Augmentation
def check_dataset_distribution(dataset_dir):
    print("\n📌 Checking dataset distribution:")
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            print(f"Category: {category}, Images: {len(os.listdir(category_path))}")

# Verify dataset after augmentation
check_dataset_distribution(AUGMENTED_DIR)

# 🔄 Real-Time Data Augmentation for Model Training
def get_train_generator():
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    return train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
    )

# 💡 Usage in train_model.py
if __name__ == "__main__":
    print("\n✅ Dataset augmentation completed! Now you can train the model.")
    train_generator = get_train_generator()
