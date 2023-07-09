import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import keras
import numpy as np
import pkg_resources

# Load the trained model
model_file = pkg_resources.resource_filename(__name__, 'modir.h5')
model = keras.models.load_model(model_file)

# Create the modi_to_marathi dictionary by reading from the mapping file
mapping_file = pkg_resources.resource_filename(__name__, 'modi_to_marathi.txt')
with open(mapping_file, 'r', encoding='utf-8') as file:
    mapping = file.read().splitlines()
modi_to_marathi = {int(line.split(':')[0]):line.split(':')[1] for line in mapping}


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Modi Lipi to Marathi Transcription")
        self.geometry("1080x720")

        # Add a header label with the converter name and the college name
        self.header = tk.Label(self, text="Modi to Marathi Converter\nSanjeevan Engineering and Technology Institute, Panhala", font=("Arial", 16))
        # Center the header label
        self.header.pack(pady=10)

        self.label = tk.Label(self, text="Select Modi Lipi Image")
        # Center the label
        self.label.pack(pady=10)

        self.upload_button = tk.Button(self, text="Upload", command=self.open_image)
        # Center the upload button
        self.upload_button.pack(pady=10)

        self.transcript_button = tk.Button(self, text="Transcript", command=self.transcribe_image)
        # Center the transcript button
        self.transcript_button.pack(pady=10)

        self.save_button = tk.Button(self, text="Save Result", command=self.save_result)
        # Center the save button
        self.save_button.pack(pady=10)

        self.exit_button = tk.Button(self, text="Exit", command=self.destroy)
        # Center the exit button
        self.exit_button.pack(pady=10)

        self.result_label = tk.Label(self)
        # Increase the font size of the result label
        self.result_label.configure(font=("Arial", 15))
        # Center the result label
        self.result_label.pack(pady=10)

        self.percentage_label = tk.Label(self)
        # Increase the font size of the percentage label
        self.percentage_label.configure(font=("Arial", 15))
        # Center the percentage label
        self.percentage_label.pack(pady=10)


    def open_image(self):
        file_name = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.xpm;*.jpg;*.bmp"), ("All Files", "*.*")])
        if file_name:
            image = Image.open(file_name)
            photo = ImageTk.PhotoImage(image)
            self.label.configure(image=photo)
            self.label.image = photo
            # Center the image label
            self.label.pack(pady=10)
            self.file_name = file_name

    def transcribe_image(self):
        img = cv2.imread(self.file_name,0)
        img = cv2.resize(img,(96,96))
            
        # reshaping image for model
        img = img.reshape((1,96,96,1)).astype('float32')
            
        # converting to range between 0-1
        img = img/255.0
            
        result = model.predict(img)
            
        perc = np.amax(result)
        pred = np.argmax(result[0])
            
        self.result_label.configure(text="Transcription: " + modi_to_marathi[pred])
        self.percentage_label.configure(text="Prediction Rate: {:.2f}%".format(perc*100))



    def save_result(self):
        file_name = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_name:
            with open(file_name, "w", encoding='utf-8') as f:
                f.write(self.result_label.cget("text"))

if __name__ == '__main__':
    window = MainWindow()
    window.mainloop()
