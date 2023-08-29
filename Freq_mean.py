from Fourier import fourier
from torchvision import transforms
import numpy as np
from PIL import Image
import os

MyTransform = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop((512, 512)),
])

def to_freq(path):
    img = Image.open(path).convert("RGB")
    # img.save("./freq_img/{}".format(os.path.basename(path)))
    crop_img = MyTransform(img)
    f_img = fourier(crop_img)
    f_img = np.array(f_img).astype(np.float32)
    return f_img

def get_mean(path):
    dir = os.listdir(path)[-100:]
    # dir = np.random.choice(dir, 100)
    size = len(dir)
    mean_freq = np.zeros((512, 512, 3), dtype=np.float32)
    for file in dir:
        file_path = os.path.join(path, file)
        freq = to_freq(file_path)
        mean_freq += freq/size
    return mean_freq

for i in range(1):
    mean_freq = get_mean("./all_data/croped_human")
    mean_freq2 = get_mean("./all_data/white_background")
    mean_freq = Image.fromarray(mean_freq.astype(np.uint8))
    mean_freq2 = Image.fromarray(mean_freq2.astype(np.uint8))
    mean_freq.save("./freq_img/mean_freq_human_100_{}.png".format(i))
    mean_freq2.save("./freq_img/mean_freq_diffusion_100_{}.png".format(i))
