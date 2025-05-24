NAME = 'SAModel'

import numpy as np
import random


'''-------------------------------------------Вычисление наблюдаемых ФРТ---------------------------------------------'''


def PSF_calc(SLIT, SEEING, SEP, pix_size):

    M = 30  # Масштаб картинки

    # Параметры изображения
    width = 200 * M
    height = 150 * M

    # Реальная ширина щели
    slit_arcsec = SLIT

    # Параметры атмосферного размытия
    sigma = 5 * M  # Дисперсия для немасштабированного изображения
    seeing = SEEING  # Размер атмосферных изображений (arcsec)

    # Параметры подобласти, содержащей центральный максимум
    region_width = 40 * M
    region_height = 40 * M

    # Параметры разделения объектов вдоль щели
    sep = SEP  # Смещение спектра 2 относительно спектра 1 (arcsec)

    # Параметры изображения
    center_x = width // 2
    center_y = height // 2

    # Размер и seeing для немасштабированного изображения
    size_unscaled = 3 * sigma
    seeing_unscaled = 2 * sigma * np.sqrt(2 * np.log(2))

    # Параметры щели на модельном изображении
    slit_width_image = int(slit_arcsec * seeing_unscaled / seeing)
    slit_position = width // 2

    # Параметры разделения объектов вдоль щели
    offset_pix = sep * seeing_unscaled / seeing  # Смещение спектра 2 относительно спектра 1 на кадре (pix)

    # Создание изображения
    image = np.zeros((height, width))
    # Массив для гауссовой функции
    x, y = np.indices((height, width))
    # Гауссова функция
    gaussian = np.exp(-((x - center_y) ** 2 + (y - center_x) ** 2) / (2 * (sigma ** 2)))
    # Добавление гауссианы к изображению
    image += gaussian

    # Создание маски для ФРТ
    mask = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            if slit_position - slit_width_image // 2 <= x <= slit_position + slit_width_image // 2:
                mask[y, x] = 1

    # Вычисление ФРТ
    diffraction_pattern = image * mask


    # Вычисление координат углов подобласти
    x_start = center_x - region_width // 2
    x_end = center_x + region_width // 2
    y_start = center_y - region_height // 2
    y_end = center_y + region_height // 2

    # Извлечение подобласти изображения ФРТ
    region = np.abs(diffraction_pattern)[y_start:y_end, x_start:x_end]
    region_shifted = np.abs(diffraction_pattern)[int(y_start + offset_pix):int(y_end + offset_pix), x_start:x_end]

    # Вычисление масштаба области ФРТ на ПЗС
    y_pixels = int(size_unscaled * pix_size * seeing / M) * 2  # Длина масштабируемого участка в пикселях
    x_pixels = int(y_pixels * region_width / region_height)
    scale_x = region_width // x_pixels
    scale_y = region_height // y_pixels

    PSF = np.zeros((y_pixels, x_pixels))
    for i in range(y_pixels):
        for j in range(x_pixels):
            PSF[i, j] = region[i * scale_y: (i + 1) * scale_y, j * scale_x: (j + 1) * scale_x].mean()

    Summ = 0
    for y in range(y_pixels):
        for x in range(y_pixels):
            Summ += PSF[y, x]

    PSF = PSF / Summ  # Нормировка

    PSF_shifted = np.zeros((y_pixels, x_pixels))
    for i in range(y_pixels):
        for j in range(x_pixels):
            PSF_shifted[i, j] = region_shifted[i * scale_y: (i + 1) * scale_y, j * scale_x: (j + 1) * scale_x].mean()

    Summ2 = 0
    for y in range(y_pixels):
        for x in range(y_pixels):
            Summ2 += PSF_shifted[y, x]

    PSF_shifted = PSF_shifted / Summ2  # Нормировка

    return(PSF + 1e-6, PSF_shifted + 1e-6)  # 1e-6 - регуляризующая добавка


'''---------------------------------------------Моделирование спектров-----------------------------------------------'''


# Вычисление континуума по закону Планка
def planck_function(wavelength, T):
    h = 6.62607015e-34  # Постоянная Планка (Дж·с)
    c = 299792458  # Скорость света (м/с)
    k = 1.380649e-23  # Постоянная Больцмана (Дж/К)
    wavelength = wavelength * 1e-10  # Перевод в метры
    return (2 * h * c ** 2 / wavelength ** 5) * (1 / (np.exp(h * c / (wavelength * k * T)) - 1))


def ExampleSpectrum(T, CRVAL1, image_size_X, CDELT1, N_emission, N_absorption,
                    sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                    MaxMag_emission, MaxMag_absorption, MaxADU):

    wavelength_max = CRVAL1 + image_size_X / CDELT1
    wavelengths = np.arange(CRVAL1, wavelength_max, CDELT1)

    continuum = planck_function(wavelengths, T)

    # Определение линий излучения
    lines_emission = []
    for t in range(N_emission):
      line_wavelength_emission = random.randint(CRVAL1, int(wavelength_max ))
      magnitude_emission = random.uniform(MinMag_emission, MaxMag_emission)  # Величина линий (случайная)
      lines_emission.append((line_wavelength_emission, magnitude_emission))

    # Добавление линий излучения к спектру
    for line_wavelength, magnitude in lines_emission:
      gaussian = np.exp(-((wavelengths - line_wavelength) / sigma_emission) ** 2 / 2)
      continuum *= (1 + magnitude * gaussian)

    # Определение линий поглощения
    lines_absorption = []
    for t in range(N_absorption):
      line_wavelength_absorption = random.randint(CRVAL1, int(wavelength_max))
      magnitude_absorption = random.uniform(MinMag_absorption, MaxMag_absorption)  # Величина линий (случайная)
      lines_absorption.append((line_wavelength_absorption, magnitude_absorption))

    # Добавление линий поглощения к спектру
    for line_wavelength, magnitude in lines_absorption:
      gaussian = np.exp(-((wavelengths - line_wavelength) / sigma_absorption) ** 2 / 2)
      continuum *= (1 - magnitude * gaussian)

    # Создание массивов для сохранения результатов
    intensity = continuum / max(continuum) * MaxADU

    return(wavelengths, intensity)


'''-----------------------------------------------Моделирование кадра------------------------------------------------'''


# Вариант без использования известных спектров
def Frame_model(slit, seeing, sep, pix_size, T1, T2, CRVAL1, image_size_X, image_size_Y, Y_est, CDELT1,
                N_emission, N_absorption, sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                MaxMag_emission, MaxMag_absorption, MaxADU1, MaxADU2, RN, BN):

    PSF_kernel_1, PSF_kernel_2 = PSF_calc(slit, seeing, sep, pix_size)

    PSF_width = len(PSF_kernel_1[0])
    PSF_heigth = len(PSF_kernel_1)


    Lambda1, Int1 = ExampleSpectrum(T1, CRVAL1, image_size_X, CDELT1, N_emission, N_absorption,
                                           sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                                           MaxMag_emission, MaxMag_absorption, MaxADU1)

    spectrum1 = np.column_stack((Lambda1, Int1))
    for k1 in range(image_size_X):
        spectrum1[k1, 0] = int(round((Lambda1[k1] - CRVAL1) / CDELT1)) + PSF_width / 2

    Lambda2, Int2 = ExampleSpectrum(T2, CRVAL1, image_size_X, CDELT1, N_emission, N_absorption,
                                           sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                                           MaxMag_emission, MaxMag_absorption, MaxADU2)

    spectrum2 = np.column_stack((Lambda2, Int2))
    for k2 in range(image_size_X):
        spectrum2[k2, 0] = int(round((Lambda2[k2] - CRVAL1) / CDELT1)) + PSF_width / 2


    # Создание первого изображения
    image1 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image1[int(i + Y_est - PSF_heigth / 2), j + x] += Int1[x] * PSF_kernel_1[i][j]

    # Создание второго изображения
    image2 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image2[int(i + Y_est - PSF_heigth / 2), j + x] += Int2[x] * PSF_kernel_2[i][j]

    # Итоговое изображение без шумов
    image = image1 + image2
    image = image[:, int(PSF_width / 2): - int(PSF_width / 2)]

    # Итоговое изображение с шумами
    image = image + BN  # Добавление фона
    image = np.random.poisson(image)  # Добавление пуассоновского шума
    # Добавление шума считывания
    for i in range(image_size_X):
        for j in range(image_size_Y):
            R = random.randint(0, RN)
            image[j, i] += R

    return(image, PSF_kernel_1, Lambda1, Int1, Lambda2, Int2)


# Вариант с использованием известных спектров
def Frame_model_known_spec(slit, seeing, sep, pix_size, CRVAL1, image_size_X, image_size_Y, Y_est, CDELT1, RN, BN,
                           Lambda1, Int1, Lambda2, Int2):

    PSF_kernel_1, PSF_kernel_2 = PSF_calc(slit, seeing, sep, pix_size)

    PSF_width = len(PSF_kernel_1[0])
    PSF_heigth = len(PSF_kernel_1)

    spectrum1 = np.column_stack((Lambda1, Int1))
    for k1 in range(image_size_X):
        spectrum1[k1, 0] = int(round((Lambda1[k1] - CRVAL1) / CDELT1)) + PSF_width / 2


    spectrum2 = np.column_stack((Lambda2, Int2))
    for k2 in range(image_size_X):
        spectrum2[k2, 0] = int(round((Lambda2[k2] - CRVAL1) / CDELT1)) + PSF_width / 2


    # Создание первого изображения
    image1 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image1[int(i + Y_est - PSF_heigth / 2), j + x] += Int1[x] * PSF_kernel_1[i][j]

    # Создание второго изображения
    image2 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image2[int(i + Y_est - PSF_heigth / 2), j + x] += Int2[x] * PSF_kernel_2[i][j]

    # Итоговое изображение без шумов
    image = image1 + image2
    image = image[:, int(PSF_width / 2): - int(PSF_width / 2)]

    # Итоговое изображение с шумами
    image = image + BN  # Добавление фона
    image = np.random.poisson(image)  # Добавление пуассоновского шума
    # Добавление шума считывания
    for i in range(image_size_X):
        for j in range(image_size_Y):
            R = random.randint(0, RN)
            image[j, i] += R

    return(image, PSF_kernel_1)
