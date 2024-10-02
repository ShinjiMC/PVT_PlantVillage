# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
from timm.models import create_model
from datasets import build_dataset
import utils
import pvt
import pvt_v2
import os

class IndexToWord:
    def __init__(self, root_path):
        # Carga las clases desde la carpeta del dataset
        self.classes = sorted(os.listdir(root_path))  # Obtén las clases y ordénalas alfabéticamente
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Mapa de clase a índice
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}  # Mapa de índice a clase

    def index_to_word(self, index):
        # Obtiene el nombre de la clase a partir del índice
        return self.idx_to_class.get(index, "Índice no válido")

    def word_to_index(self, word):
        # Obtiene el índice de la clase a partir del nombre
        return self.class_to_idx.get(word, "Palabra no válida")

# Uso del transformador


def main():
    # Define paths
    checkpoint_path = 'checkpoints/pvt_v2_b2_li/checkpoint.pth'
    image_path = 'test/Strawberry___healthy.jpg'
    image_path2 = 'test/Tomato_Target_Spot.jpg'

    # Set device to CUDA
    device = torch.device('cuda')

    # Initialize model
    model = create_model(
        'pvt_v2_b2_li',  # Specify the model
        pretrained=False,
        num_classes=1000,  # Adjust this according to your dataset
    )

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Load and preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of the model
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])

    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the predicted class index

    root_path = './PlantVillage'
    transformer = IndexToWord(root_path)
    index = predicted.item() + 8
    print(f'Predicted class index: {index}')
    class_name = transformer.index_to_word(index)
    print(f'Índice {index} corresponde a la clase: {class_name}')
    
    #image2
    image2 = Image.open(image_path2).convert('RGB')
    image2 = preprocess(image2).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        output2 = model(image2)
        _, predicted2 = torch.max(output2, 1)
        
    index2 = predicted2.item() + 8
    print(f'Predicted class index: {index2}')
    class_name2 = transformer.index_to_word(index2)
    print(f'Índice {index2} corresponde a la clase: {class_name2}')

if __name__ == '__main__':
    main()
