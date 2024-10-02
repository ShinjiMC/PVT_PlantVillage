import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageTk
from torchvision import transforms
from timm.models import create_model
from datasets import build_dataset
import utils
import pvt
import pvt_v2
import os
import sys
import tkinter as tk
from tkinter import filedialog, Label, Button

class IndexToWord:
    def __init__(self, root_path):
        # Carga las clases desde la carpeta del dataset
        self.classes = sorted(os.listdir(root_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def index_to_word(self, index):
        # Obtiene el nombre de la clase a partir del índice
        return self.idx_to_class.get(index, "Índice no válido")

class PlantPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Enfermedades de Plantas")
        self.root.geometry("800x600")

        self.index_to_word = IndexToWord('./PlantVillage')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

        self.title_label = Label(root, text="Clasificación de Enfermedades de Plantas", font=("Arial", 20))
        self.title_label.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.prediction_label = Label(root, text="Predicción: ", font=("Arial", 16))
        self.prediction_label.pack(pady=10)

        self.upload_button = Button(root, text="Subir Imagen", command=self.upload_image)
        self.upload_button.pack(pady=20)
        
        self.tk_image = None

    def load_model(self):
        checkpoint_path = 'checkpoints/pvt_v2_b2_li/checkpoint.pth'
        model = create_model('pvt_v2_b2_li', pretrained=False, num_classes=1000)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        return preprocess(image).unsqueeze(0).to(self.device)

    def predict_image(self, image_path):
        image_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
            index = predicted.item() + 8  # Ajuste del índice si es necesario
            return self.index_to_word.index_to_word(index)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Imágenes", "*.*")])
        if file_path:
            predicted_class = self.predict_image(file_path)
            self.display_image(file_path)
            self.prediction_label.config(text=f'Predicción: {predicted_class}')

    def display_image(self, img_path):
        image = Image.open(img_path)
        image.thumbnail((400, 400))
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image

if __name__ == '__main__':
    root = tk.Tk()
    app = PlantPredictionApp(root)
    root.mainloop()