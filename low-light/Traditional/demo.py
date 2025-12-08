import argparse
import cv2
import math
import numpy as np


# HE
def hsv_he(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv_img[..., 2]
    value_he = cv2.equalizeHist(value)
    hsv_img[..., 2] = value_he
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return rgb_img


# CLAHE
def hsv_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv_img[..., 2]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    value_clahe = clahe.apply(value)
    hsv_img[..., 2] = value_clahe
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return rgb_img


# DarkChannel
class DarkChannel:
    def __init__(self, sz=15):
        self._sz = sz

    def _dark_channel(self, im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self._sz, self._sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def _atm_light(self, im, dark):
        [h, w] = im.shape[:2]
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort()
        indices = indices[imsz-numpx::]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def _transmission_estimate(self, im, A):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)

        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind]/A[0, ind]

        transmission = 1 - omega*self._dark_channel(im3)
        return transmission

    def _guided_filter(self, im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I*mean_p

        mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I*mean_I

        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a*im + mean_b
        return q

    def _transmission_refine(self, im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)/255
        r = 60  # default 60
        eps = 0.0001
        t = self._guided_filter(gray, et, r, eps)

        return t

    def _recover(self, im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind]-A[0, ind])/t + A[0, ind]

        return res

    def run(self, image):
        rever_img = 255 - image
        rever_img_bn = rever_img.astype('float64')/255
        dark = self._dark_channel(rever_img_bn)
        A = self._atm_light(rever_img_bn, dark)
        te = self._transmission_estimate(rever_img_bn, A)
        t = self._transmission_refine(image, te)
        J = self._recover(rever_img_bn, t, A, 0.1)

        rever_res_img = (1-J)*255

        return rever_res_img


# MSR
def numpy_log(x):
    return np.log(x + 1)


def mean_std_normalize(result, dynamic=2.0):
    mean = np.mean(result, axis=(0, 1))
    stdvar = np.sqrt(np.var(result, axis=(0, 1)))
    min_value = mean - dynamic * stdvar
    max_value = mean + dynamic * stdvar
    result = (result - min_value) / (max_value - min_value)
    result = 255 * np.clip(result, 0, 1)
    return result.astype("uint8")


def MSR(low_light, sigmas=[10, 50, 100], weights=[0., 0, 0], dynamic=2):
    weights = np.array(weights) / np.sum(weights)
    low_light = low_light.astype("float32")
    log_I = numpy_log(low_light)
    log_Ls = [cv2.GaussianBlur(log_I, (0, 0), sig) for sig in sigmas]
    log_R = weights[0] * (log_I - log_Ls[0])
    for i in range(1, len(weights)):
        log_R += weights[i] * (log_I - log_Ls[i])
    temp = np.exp(log_R)
    return mean_std_normalize(temp, dynamic)

# part2 
def get_ksize(sigma):
    # opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0

    return int(((sigma - 0.8)/0.15) + 2.0)

def get_gaussian_blur(img, ksize=0, sigma=5):
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)

    # Gaussian 2D-kernel can be seperable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    # if ksize >= 11, then convolution is computed by applying fourier transform
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

def ssr(img, sigma):
    # Single-scale retinex of an image
    # SSR(x, y) = log(I(x, y)) - log(I(x, y)*F(x, y))
    # F = surrounding function, here Gaussian
    epsilon = 1e-10
    return np.log10(img + epsilon) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0 + epsilon)

def msr2(img, sigma_scales=[15, 80, 250]):
    # Multi-scale retinex of an image
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales

    msr = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr += ssr(img, sigma)

    # divide MSR by weights of each scale
    # here we use equal weights
    msr = msr / len(sigma_scales)

    # computed MSR could be in range [-k, +l], k and l could be any real value
    # so normalize the MSR image values in range [0, 255]
    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return msr


# MSRCR
def MSRCR(low_light, sigmas=[15, 80, 200], weights=[0.33, 0.33, 0.34], alpha=128, dynamic=2.0):
    assert len(sigmas) == len(weights), "scales are not consistent !"
    weights = np.array(weights) / np.sum(weights)
    # 图像转成 float 处理
    low_light = low_light.astype("float32")
    # 转到 log 域
    log_I = numpy_log(low_light)
    # 每个尺度下做高斯模糊, 提取不同的平滑层, 作为光照图的估计
    log_Ls = [cv2.GaussianBlur(log_I, (0, 0), sig) for sig in sigmas]
    # 多个尺度的 MSR 叠加
    log_R = np.stack([weights[i] * (log_I - log_Ls[i])
                        for i in range(len(sigmas))])
    log_R = np.sum(log_R, axis=0)
    # 颜色恢复
    norm_sum = numpy_log(np.sum(low_light, axis=2))
    result = log_R * (numpy_log(alpha * low_light) -
                      np.atleast_3d(norm_sum))
    # result = numpy.exp(result)
    # 标准化
    return mean_std_normalize(result, dynamic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Low light enhancement"
    )
    parser.add_argument(
        "-i", "--image_name", default=None, type=str,
        help="image name"
    )
    parser.add_argument(
        "-m", "--method", default='darkChannel', type=str,
        help="method name: he|clahe|darkChannel|msr|msrcr"
    )
    parser.add_argument(
        "-s", "--save_image", default=None, type=str,
        help="save image name"
    )

    opt = parser.parse_args()
    image_name = opt.image_name
    method = opt.method
    save_image = opt.save_image

    img = cv2.imread(image_name)
    if method == "he":
        res_img = hsv_he(img)
        cv2.imwrite(save_image, res_img)
    elif method == "clahe":
        res_img = hsv_clahe(img)
        cv2.imwrite(save_image, res_img)
    elif method == "darkChannel":
        dark_channel = DarkChannel(15)
        res_img = dark_channel.run(img)
        cv2.imwrite(save_image, res_img)
    elif method == "msr":
        res_img = MSR(img, [5, 25, 50, 75, 100], [0.2, 0.2, 0.2, 0.3, 0.1])
        cv2.imwrite(save_image, res_img)
    elif method == "msr2":
        # img = img.astype(np.float64)
        res_img = msr2(img)
        cv2.imwrite(save_image, res_img)
    elif method == "msrcr":
        res_img = MSRCR(
            img,
            sigmas=[5, 25, 50, 75, 100],
            weights=[0.2, 0.2, 0.2, 0.3, 0.1],
            alpha=128,
            dynamic=2.0)
        cv2.imwrite(save_image, res_img)
    else:
        print(f"[ERROR]: method {method} not in [he, clahe, darkChannel, msr, msrcr]")

# eg:
# python3 demo.py -i "images/test_001.jpg" -s "results/test_001_he.jpg" -m he