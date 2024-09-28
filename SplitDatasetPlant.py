import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, train_ratio=0.8):
    # Crear las carpetas de destino si no existen
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterar sobre cada clase en el directorio fuente
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        # Verifica que sea un directorio
        if os.path.isdir(class_path):
            # Listar imágenes en la clase
            images = os.listdir(class_path)
            random.shuffle(images)  # Mezclar imágenes

            # Calcular el número de imágenes para entrenamiento y validación
            num_train = int(len(images) * train_ratio)
            train_images = images[:num_train]
            val_images = images[num_train:]

            # Crear carpetas para la clase en train y val
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            # Mover imágenes a las carpetas correspondientes
            for image in train_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))
            for image in val_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_name, image))

    print("Dataset dividido en train y val.")

# Definir los directorios
source_directory = 'PlantVillage'  # Cambia esto si es necesario
train_directory = 'PlantVillage2/train'
val_directory = 'PlantVillage2/val'

# Llamar a la función
split_dataset(source_directory, train_directory, val_directory)
