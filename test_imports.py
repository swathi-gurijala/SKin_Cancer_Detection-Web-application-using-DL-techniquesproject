try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
