from keras.layers import *
from keras.models import *
import cv2
import numpy as np
from keras.models import load_model


def calculate_psnr(compressed_img, decompressed_img):
    mse = np.mean((compressed_img - decompressed_img) ** 2)
    if mse == 0:
        return ("PSNR Error - MSE = 0")
    else:
        psnr = 10 * np.log10(255 ** 2 / mse)
        return psnr


def calculate_ssim(input_img, compressed_img):

    img_avg = np.mean(input_img)
    compressed_img_avg = np.mean(compressed_img)

    img_var = np.var(input_img)
    compressed_img_var = np.var(compressed_img)

    covar = np.cov(input_img.flatten(), compressed_img.flatten())[0][1]

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * img_avg * compressed_img_avg + c1) * (2 * covar + c2)
    denominator = (img_avg ** 2 + compressed_img_avg ** 2 + c1) * (img_var + compressed_img_var + c2)
    ssim = numerator / denominator
    return ssim

def calculate_NAE(img_arr, compressed_img_arr):
    img1 = np.array(img_arr)
    img2 = np.array(compressed_img_arr)
    diff = np.abs(img1 - img2)
    nae = np.sum(diff) / (img1.size * 255)
    return nae


def downsample(input_image):
    yuv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YUV)

    yuv_downsampled = cv2.resize(yuv_image, (input_image.shape[1] // 2, input_image.shape[0] // 2))
    downsampled_img = cv2.cvtColor(yuv_downsampled, cv2.COLOR_YUV2RGB)

    cv2.imwrite(f"Images/Compressed/{img_name}_compressed.png", downsampled_img)

    return downsampled_img

def upsample(rgb_downsampled_image):

    model = load_model("model.h5", compile=False)

    YCrCb_img = cv2.cvtColor(rgb_downsampled_image, cv2.COLOR_RGB2YCrCb)
    
    y_channel = YCrCb_img[:,:,0]
    u_channel = YCrCb_img[:,:,1]
    v_channel = YCrCb_img[:,:,2]

    y_interpolate = cv2.resize(y_channel, (256,256), interpolation=cv2.INTER_AREA)
    y_input = np.expand_dims(y_interpolate, axis=0)

    y_upsampled = model.predict(y_input)
    u_upsampled = cv2.resize(u_channel, (512, 512), interpolation=cv2.INTER_AREA)
    v_upsampled = cv2.resize(v_channel, (512, 512), interpolation=cv2.INTER_AREA)
    uv_upsampled = np.stack((v_upsampled, u_upsampled), axis=-1)
    yuv_upsampled = np.concatenate((y_upsampled[0], uv_upsampled), axis=2)

    conversion_matrix = np.array([[1.164, 0.000, 1.596],
                                [1.164, -0.392, -0.813],
                                [1.164, 2.017, 0.000]])
    yuv_upsampled = yuv_upsampled.astype(np.float32)
    yuv_upsampled[:, :, 1:] -= 128
    rgb_upsampled = np.dot(yuv_upsampled, conversion_matrix.T)
    rgb_upsampled = np.clip(rgb_upsampled, 0, 255)
    rgb_upsampled = rgb_upsampled.astype(np.uint8)

    cv2.imwrite(f'Images/Upsampled/{img_name}_upsampled.png', rgb_upsampled)
    return rgb_upsampled

img_name = "landscape.png"
input_img = cv2.imread(f'Images/{img_name}', cv2.IMREAD_COLOR)
downsampled_img = downsample(input_img)
upsampled_image = upsample(downsampled_img)

psnr = calculate_psnr(input_img,upsampled_image)
ssim = calculate_ssim(input_img,upsampled_image)
nae = calculate_NAE(input_img,upsampled_image)


print(f"SSIM: {ssim}")
print(f"PSNR: {psnr}")
print(f"NAE: {nae}")