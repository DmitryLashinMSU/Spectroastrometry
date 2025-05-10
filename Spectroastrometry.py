NAME = 'Spectroastrometry'

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from matplotlib import pyplot as plt


'''--------------------------------------------Функции для аппроксимации---------------------------------------------'''


''' Аппроксимация "положительной" гауссианой '''


def G_fit(coord, I):

    # Начальные значения параметров
    P0 = np.zeros(5)
    P0[0] = np.sum(I)  # Полный интеграл
    P0[2] = np.mean(coord)  # Оценка центра
    P0[1] = np.sum((coord - P0[2]) ** 2 * I) / P0[0]  # Оценка стандартного отклонения
    P0[3] = 5  # Начальный фон

    # Подгонка с использованием МНК
    result = least_squares(L, P0, args=(coord, I))
    P = result.x
    return P


# Функция потерь
def L(x, coord, I):
    return I - G(coord, x)


# Аппроксимирующая функция
def G(coord, P):
    return P[0] / 2 * (erf((coord - P[2]) / (P[1] * np.sqrt(2))) +
                       erf((P[2] - coord + 1) / (P[1] * np.sqrt(2)))) + P[3]


''' Аппроксимация "отрицательной" гауссианой '''


def neg_G_fit(coord, I):

  # Начальные значения параметров
  P0 = np.zeros(5)
  I = -1 * I
  P0[0] = -1 * np.sum(np.max(I) - I)  # Глубина поглощения
  P0[2] = np.mean(coord)  # Оценка центра
  P0[1] = np.sqrt(np.sum((coord - P0[2]) ** 2 * (np.min(I) - I)) / np.sum((np.min(I) - I)))
  P0[3] = np.max(I)  # Начальный фон
  m_est = np.sum((coord - P0[2]) * (np.min(I) - I)) / np.sum((coord - P0[2]) ** 2)
  P0[4] = m_est

  # Подгонка с использованием МНК
  result = least_squares(neg_L, P0, args=(coord, I))
  P = result.x
  return P


# Функция потерь
def neg_L(x, coord, I):
    return neg_G(coord, x) - I


# Аппроксимирующая функция
def neg_G(coord, P):
    return - P[0] / 2 * (erf((coord - P[2]) / (P[1] * np.sqrt(2))) +
                         erf((P[2] - coord + 1) / (P[1] * np.sqrt(2)))) - P[3]  # + P[4] * coord


'''----------------------------------Извлечение спектроастрометрической информации-----------------------------------'''


''' Определение наблюдаемого смещения и FWHM '''


def Center_search(CRVAL1, CDELT1, x_start, x_end, area, Y_est, image):
    # Срез спектра
    X0 = x_start  # Начальный пиксель
    X1 = x_end  # Конечный пиксель
    X_count = 0  # Счетчик для заполнения массивов в цикле

    LAMBDA = np.zeros(X1 - X0)
    CENTER = np.zeros(X1 - X0)
    ERRORBAR = np.zeros(X1 - X0)
    SPEC = np.zeros(X1 - X0)
    FWHM = np.zeros(X1 - X0)

    while X_count < (X1 - X0):
      slice_I = np.zeros(2 * area)  # Срез спектра на заданной координате X_count
      coord = np.zeros(2 * area)
      for j in range(2 * area):
        slice_I[j] = image[Y_est - area + j][x_start + X_count]
        coord[j] = Y_est - area + j
        SPEC[X_count] += image[Y_est - area + j][ x_start + X_count]

      # Применение аппроксимации к модели:
      P = G_fit(coord, slice_I)

      FWHM[X_count] = (2 * np.sqrt(2 * np.log(2)) * P[1])
      SNR = P[0] / P[3]
      error = 0.6 * FWHM[X_count] / SNR
      LAMBDA[X_count] = CRVAL1 + (x_start + X_count) * CDELT1
      CENTER[X_count] = P[2]
      ERRORBAR[X_count] = error

      X_count += 1

    return LAMBDA, SPEC, CENTER, FWHM, ERRORBAR


''' Оценка фактического смещения по линиям'''


def Line_center_search(image, x_line, x_cont, Y_est, area):

 # area - полуширина области поперек направления дисперсии, в которой проводится аппроксимация спектра
    Coord_center = np.zeros(2 * area)
    I_center = np.zeros(2 * area)
    for j in range(2 * area):
        I_center[j] = image[Y_est - area + j][x_line]
        Coord_center[j] = Y_est - area + j

    Coord_cont = np.zeros(2 * area)
    I_cont = np.zeros(2 * area)

    for j in range(2 * area):
        I_cont[j] = image[Y_est - area + j][x_cont]
        Coord_cont[j] = Y_est - area + j

    # Выделение линии

    Line = np.abs(I_center - I_cont)
    P4 = G_fit(Coord_center, Line)

    FWHM = (2 * np.sqrt(2 * np.log(2)) * P4[1])
    SNR = P4[0] / P4[3]
    error = 0.6 * FWHM / SNR

    return P4, error


'''-------------------------------------------------Обработка кадров-------------------------------------------------'''

''' Медианная фильтрация '''

def Median_images(images):
    # Проверка на одинаковый размер изображений
    shape = images[0].shape
    if not all(img.shape == shape for img in images):
      raise ValueError("Все изображения должны иметь одинаковый размер.")

    stacked_images = np.stack(images, axis=-1)

    median_image = np.median(stacked_images, axis=-1)

    return median_image


''' Усреднение '''


def Average_images(images):
    # Проверка на одинаковый размер изображений
    shape = images[0].shape
    if not all(img.shape == shape for img in images):
      raise ValueError("Все изображения должны иметь одинаковый размер.")

    stacked_images = np.stack(images, axis=-1)

    average_image = np.mean(stacked_images, axis=-1)

    return average_image


''' Устранение тренда методом скользящего среднего '''


def Detrend(window, CENTER):

    CENTER_rolling = np.zeros_like(CENTER)
    rolling_mean = np.zeros_like(CENTER)

    for i in range(len(CENTER)):
        start_index = max(0, i - window // 2)
        end_index = min(len(CENTER), i + window // 2 + 1)

        rolling_mean[i] = np.mean(CENTER[start_index:end_index])

        CENTER_rolling[i] = CENTER[i] - rolling_mean[i]

    return CENTER_rolling, rolling_mean


''' Устранение степенного тренда'''


def remove_polynomial_trend(data, n):
    x = np.arange(len(data))
    coefficients = np.polyfit(x, data, n)
    polynomial_trend = np.polyval(coefficients, x)
    detrended_data = data - polynomial_trend
    return detrended_data


'''------------------------------------------------Отображение данных------------------------------------------------'''


''' 3D график для выбранной части кадра '''


def show_3D(image, x_start, x_end, y_start, y_end):

    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)

    # Фильтрация данных
    x_min, x_max = x_start, x_end
    y_min, y_max = y_start, y_end

    x_mask = (x >= x_min) & (x <= x_max)
    y_mask = (y >= y_min) & (y <= y_max)

    mask = x_mask & y_mask

    x_filtered = x[mask]
    y_filtered = y[mask]
    image_filtered = image[mask]

    # Преобразование обратно в 2D массив для plot_surface (необходимо для правильного отображения)
    x_filtered = x_filtered.reshape(image[y_min:y_max+1, x_min:x_max+1].shape)
    y_filtered = y_filtered.reshape(image[y_min:y_max+1, x_min:x_max+1].shape)
    image_filtered = image[y_min:y_max+1, x_min:x_max+1]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_filtered, y_filtered, image_filtered / 1000, cmap='viridis')
    ax.set_xlabel('X, pix')
    ax.set_ylabel('Y, pix')
    ax.set_zlabel('I, ADU * 1e3')
    plt.show()
