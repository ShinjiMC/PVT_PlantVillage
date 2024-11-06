import numpy as np
import time
import torch
import argparse
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
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class IndexToWord:
    def __init__(self, root_path):
        # Carga las clases desde la carpeta del dataset
        self.classes = sorted(os.listdir(root_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def index_to_word(self, index):
        return self.idx_to_class[index] if index in self.idx_to_class else 'Desconocido'

class PlantPredictionApp:
    def __init__(self, root, model_name, checkpoint_dir):
        self.root = root
        self.root.title("Predicción de Enfermedades de Plantas")
        self.root.geometry("800x600")

        self.index_to_word = IndexToWord('./Potato_Status/val')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.model = self.load_model()

        self.title_label = Label(root, text="Clasificación de Enfermedades de Plantas", font=("Arial", 20))
        self.title_label.pack(pady=10)
        self.model_label = Label(root, text=model_name, font=("Arial", 20))
        self.model_label.pack(pady=10)
        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.prediction_label = Label(root, text="Predicción: ", font=("Arial", 16))
        self.prediction_label.pack(pady=10)

        self.probabilities_label = Label(root, text="Probabilidades: ", font=("Arial", 14))
        self.probabilities_label.pack(pady=10)

        self.upload_button = Button(root, text="Subir Imagen", command=self.upload_image)
        self.upload_button.pack(pady=20)
        
        self.tk_image = None

    def load_model(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.model_name}/checkpoint.pth')
        model = create_model(self.model_name, pretrained=False, num_classes=3)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        t = []
        size = int((256 / 224) * 224)
        t.append(transforms.Resize(size, interpolation=3),)
        t.append(transforms.CenterCrop(224))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        preprocess = transforms.Compose(t)
        image = Image.open(image_path).convert('RGB')
        return preprocess(image).unsqueeze(0).to(self.device)

    def predict_image(self, image_path):
        image_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            index = predicted.item()  # Ajuste del índice si es necesario
            probabilities_percent = (probabilities[0] * 100).tolist()
            prediction_text = self.index_to_word.index_to_word(index)
            
            probs_text = "\n".join(
                f"{self.index_to_word.index_to_word(i)}: {prob:.2f}%" 
                for i, prob in enumerate(probabilities_percent)
            )
            
            self.prediction_label.config(text=f'Predicción: {prediction_text}')
            self.probabilities_label.config(text=f'Probabilidades:\n{probs_text}')
            return prediction_text

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
    parser = argparse.ArgumentParser(description="Aplicación de Clasificación de Enfermedades de Plantas")
    parser.add_argument('--model_name', type=str, required=False, default= 'pvt_v2_b2_li',help="Nombre del modelo (por ejemplo, 'pvt_v2_b2_li')")
    parser.add_argument('--checkpoint_dir', type=str, required=False, default= 'checkpoints/', help="Directorio donde se encuentran los checkpoints")
    
    args = parser.parse_args()
    root = tk.Tk()
    app = PlantPredictionApp(root, args.model_name, args.checkpoint_dir)
    root.mainloop()