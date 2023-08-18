import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import ImageTk, Image
from Tester import pred
from Fourier import fourier
from torchvision import transforms
root = tk.Tk()
root.title("AI Classifier")
root.geometry("1500x1100")

MyTransform = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop((512, 512)),
])

def open_image():
    file_path = filedialog.askopenfilename()
    # for i in range(100):
        # file_path = "./all_data/square_diffusion_data/sq_0 ({}).png".format(i+1)
    
    img = Image.open(file_path).convert("RGB")

    crop_img = MyTransform(img)

    res = pred(crop_img)
    # res[1].save("./all_data/modified/{}.png".format(i+200))
    # print("Image shape:", res[1].size, type(res[1]))

    display_image(crop_img, res[1], crop_img)
    print_sentence("AI generated probability: %.3f"%(float(res[0][0])))

f_res = np.zeros((400, 400, 3), dtype=np.uint8)
f_res = transforms.ToPILImage()(f_res)
# Define a function to display the image in the UI
def display_image(img, res, crop_img):
    global f_res
    resize_ratio = 400 / max(img.size)
    img = img.resize((int(img.size[0]*resize_ratio), int(img.size[1]*resize_ratio)), Image.Resampling.NEAREST) 
    resize_ratio = 400 / max(res.size)
    res = res.resize((int(res.size[0]*resize_ratio), int(res.size[1]*resize_ratio)), Image.Resampling.NEAREST)
    low_pass = fourier(crop_img, True)
    f_img = fourier(img)
    # f_res = np.array(f_res).astype(np.float32)
    # f_res += np.array(fourier(res), dtype=np.float32)/3.
    # f_res = Image.fromarray(f_res.astype(np.uint8))
    f_res = fourier(res)
    f_low = fourier(low_pass)
    resize_ratio = 400 / max(low_pass.size)
    low_pass = low_pass.resize((int(low_pass.size[0]*resize_ratio), int(low_pass.size[1]*resize_ratio)), Image.Resampling.NEAREST)
    f_low = f_low.resize((int(f_low.size[0]*resize_ratio), int(f_low.size[1]*resize_ratio)), Image.Resampling.NEAREST)
    img_tk = ImageTk.PhotoImage(img) 
    f_img_tk = ImageTk.PhotoImage(f_img)
    res_tk = ImageTk.PhotoImage(res)
    f_res_tk = ImageTk.PhotoImage(f_res)
    low_tk = ImageTk.PhotoImage(low_pass)
    f_low_tk = ImageTk.PhotoImage(f_low)


    image_label.config(image=img_tk)
    f_image_label.config(image=f_img_tk)
    image_label2.config(image=res_tk)
    f_res_label.config(image=f_res_tk)
    image_label3.config(image=low_tk)
    f_low_label.config(image=f_low_tk)

    image_label.image = img_tk
    f_image_label.image = f_img_tk
    image_label2.image = res_tk
    f_res_label.image = f_res_tk
    image_label3.image = low_tk
    f_low_label.image = f_low_tk


def print_sentence(sentence):
    sentence_label.config(text=sentence, font=("Courier", 15))

# Create the "Open Image" button
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.grid(row=0, column=0, columnspan=3)

# Create a label to display the image
image_label = tk.Label(root)
image_label.grid(row=1, column=0)

image_label2 = tk.Label(root)
image_label2.grid(row=1, column=1)

image_label3 = tk.Label(root)
image_label3.grid(row=1, column=2)

f_image_label = tk.Label(root)
f_image_label.grid(row=2, column=0)

f_res_label = tk.Label(root)
f_res_label.grid(row=2, column=1)

f_low_label = tk.Label(root)
f_low_label.grid(row=2, column=2)



# Create a label to display the sentence
sentence_label = tk.Label(root, text="")
sentence_label.grid(row=3, column=0, columnspan=3)

# Run the main loop
root.mainloop()
