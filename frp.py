import tkinter as tk
from tkinter import filedialog
import subprocess
import sys

def open_file_dialog():
    filename = filedialog.askopenfilename()
    if filename:
        run_face_recognition(filename)

def run_face_recognition(image_path):
    # Script location
    script_path = "./recognition.py"
    
    # Run script with the selected image
    subprocess.run([sys.executable, script_path, "--test", "-f", image_path])

root = tk.Tk()
root.geometry("500x500")  # Window size
root.configure(bg="black")  
label = tk.Label(root, text="FAC3 LAB", font=("Arial", 44), fg="green", bg="black")
label.place(relx=0.5, rely=0.5, anchor='center')  # Text Position

open_file_button = tk.Button(root, text="Open File", command=open_file_dialog, font=("Arial", 24))
open_file_button.pack(side="bottom", pady=20)  # Button Position

root.mainloop()
