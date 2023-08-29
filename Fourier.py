import numpy as np
from torchvision import transforms
def fourier(img, iff:bool=False):
    if not iff:
        # img = img.convert('L')
        img = transforms.ToTensor()(img)/255
    else:
        img = transforms.ToTensor()(img)
    f_img = np.fft.fft2(img)
    f_img = np.fft.fftshift(f_img)
    # print(f_img.shape)
    if not iff:
        f_img = np.log(np.abs(f_img)+1)
    else:
        # low pass filter
        filter_freq_end = 200
        filter_freq_start = 0
        mask = np.ones_like(f_img)
        print(mask.shape)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                dis = (i-mask.shape[1]//2)**2 + (j-mask.shape[2]//2)**2
                # mask[:,i, j] = np.exp(-(dis / filter_freq_end**2) ** (2 * 3)) #expotential filter
                if dis > filter_freq_end**2:
                # if((i-mask.shape[1]//2)**2>filter_freq_end**2 or (j-mask.shape[2]//2)**2> filter_freq_end**2):
                    mask[:,i,j] = np.max(1/(filter_freq_end**2)*dis - 1, 0)

        f_img = f_img * mask
        # ifft
        f_img = np.fft.ifftshift(f_img)
        f_img = np.fft.ifft2(f_img)
        f_img = np.abs(f_img)
        f_img[f_img>1] = 1
        f_img[f_img<0] = 0

    f_img = np.transpose(f_img, (1,2,0))*255
    f_img = f_img.astype(np.uint8)
    # print(f_img.shape)
    f_img = transforms.ToPILImage()(f_img)
    return f_img