import numpy as np
import imageio
import matplotlib.pyplot as plt
import time


def DFT(vec):
    N = vec.shape[0]

    fourier_1d = np.zeros((N), dtype=complex)

    for u in range(N):
        for x in range(N):
            fourier_1d[u] += vec[x] * (
                np.cos((2 * np.pi * u * x / N)) - (1j * np.sin((2 * np.pi * u * x / N)))
            )

    return fourier_1d


def FFT(vec):
    vec = np.asarray(vec, dtype=complex)

    N = vec.shape[0]

    if N == 1:
        return DFT(vec)
    else:
        vec_even = FFT(vec[::2])
        vec_odd = FFT(vec[1::2])
        W_u_2k = np.cos(2 * np.pi * np.arange(N) / N) - 1j * np.sin(
            2 * np.pi * np.arange(N) / N
        )

        F_u = vec_even + vec_odd * W_u_2k[: N // 2]

        F_u_M = vec_even + vec_odd * W_u_2k[N // 2 :]

        fourier_vec = np.concatenate([F_u, F_u_M])

    return fourier_vec


# def FFT_2D(img):
#     fourier_img = np.zeros((img.shape), dtype=complex)
#     N, M = img.shape[:2]
#     for i in range(img.shape[0]):
#         fourier_img[i, :] = FFT(img[i, :])
#
#     for j in range(img.shape[1]):
#         fourier_img[:, j] = FFT(fourier_img[:, j])
#
#     return fourier_img


def FFT_2D(img):
    # zero-pad image to nearest power of 2 in both dimensions
    N, M = img.shape[:2]
    N_padded = 2 ** int(np.ceil(np.log2(N)))
    M_padded = 2 ** int(np.ceil(np.log2(M)))
    img_padded = np.pad(img, ((0, N_padded - N), (0, M_padded - M)), mode="constant")

    # compute 2D FFT of padded image
    fourier_img = np.zeros(img_padded.shape, dtype=complex)
    for i in range(img_padded.shape[0]):
        fourier_img[i, :] = FFT(img_padded[i, :])
    for j in range(img_padded.shape[1]):
        fourier_img[:, j] = FFT(fourier_img[:, j])

    # return Fourier transform of original image
    return fourier_img[:N, :M]


def shift_image(image):
    new_image = np.zeros((image.shape))

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_image[x, y] = image[x, y] * ((-1) ** (x + y))

    return new_image


def IFFT(fourier_vec):
    fourier_conj = np.conjugate(fourier_vec)

    fourier_conj = FFT(fourier_conj)

    inv_fourier = np.conjugate(fourier_conj)

    inv_fourier = inv_fourier / fourier_vec.shape[0]

    return inv_fourier


def IFFT_2D(fourier_img):
    inv_fourier_img = np.zeros(fourier_img.shape, dtype=complex)

    for i in range(fourier_img.shape[0]):
        inv_fourier_img[i, :] = IFFT(fourier_img[i, :])

    for j in range(fourier_img.shape[1]):
        inv_fourier_img[:, j] = IFFT(inv_fourier_img[:, j])

    return np.real(inv_fourier_img)


def log_transform(fourier_img):
    fourier_spectrum = np.zeros((fourier_img.shape))
    for i in range(fourier_img.shape[0]):
        for j in range(fourier_img.shape[1]):
            fourier_spectrum[i, j] = np.sqrt(
                fourier_img[i, j].real ** 2 + fourier_img[i, j].imag ** 2
            )

    # Log transform of image
    fourier_img = (255 / np.log10(1 + 255)) * np.log10(
        1 + (255 / (np.max(fourier_spectrum)) * fourier_spectrum)
    )

    return fourier_img, fourier_spectrum


def fourier_padding(img):
    N, M = img.shape
    padded_image = np.zeros((N * 2, M * 2))


def fourier_conv(img, kernel):
    n, m = kernel.shape

    shifted_img = shift_image(img)
    fourier_img = FFT_2D(shifted_img)
    print("image in fourier domain")

    padded_kernel = np.zeros((img.shape))
    padded_kernel[:n, :m] = kernel[:, :]

    shifted_kernel = shift_image(padded_kernel)
    fft_kernel = FFT_2D(shifted_kernel)

    conv_img = fft_kernel * fourier_img

    inv_conv_img = IFFT_2D(conv_img)

    inv_conv_img = shift_image(inv_conv_img)

    return inv_conv_img


# path = "/home/gregz/Files/CNN/data/luna.JPG"
# image = imageio.imread(path, as_gray=True)
#
# plt.imshow(image, cmap="gray", vmin=0, vmax=255, aspect="auto")
# plt.show()
# shifted_image = shift_image(image)
#
# plt.imshow(shifted_image, cmap="gray", vmin=0, vmax=255, aspect="auto")
# plt.show()
#
# start_time = time.time()
#
# fourier_img = FFT_2D(shifted_image)
#
# log_img, fourier_spec = log_transform(fourier_img)
#
# plt.imshow(fourier_img.real, cmap="gray", vmin=0, vmax=255, aspect="auto")
# plt.show()
#
# inv_fourier_img = IFFT_2D(fourier_img)
# inv_fourier_img = shift_image(inv_fourier_img)
#
# plt.imshow(inv_fourier_img, cmap="gray", vmin=0, vmax=255, aspect="auto")
# plt.show()
#
#
# np_fourier_img = np.fft.fft2(image)
# np_fourier_img = np.fft.fftshift(np_fourier_img)
#
# plt.imshow(np_fourier_img.real, cmap="gray", vmin=0, vmax=255, aspect="auto")
# plt.show()
