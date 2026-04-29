# nmf_core.py
# Ручная реализация неотрицательной матричной факторизации (NMF)
# по Т. Сегарану, глава 10.

import numpy as np


def difcost(a, b):
    """
    Вычисляет сумму квадратов разностей между элементами двух матриц.
    Это целевая функция, которую NMF стремится минимизировать.
    """
    dif = 0
    # Цикл по строкам и столбцам матрицы
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            # Суммируем квадраты разностей
            dif += pow(a[i, j] - b[i, j], 2)
    return dif


def factorize(v, pc=10, iter=50):
    """
    Факторизация матрицы v на матрицы w (веса) и h (признаки).

    Параметры:
      v  — исходная матрица (numpy matrix)
      pc — число признаков (компонент)
      iter — максимальное число итераций

    Возвращает:
      w, h — матрицы весов и признаков
    """
    ic = np.shape(v)[0]  # число строк (документов)
    fc = np.shape(v)[1]  # число столбцов (слов)

    # Инициализация матриц весов и признаков случайными значениями
    w = np.matrix([[np.random.random() for j in range(pc)] for i in range(ic)])
    h = np.matrix([[np.random.random() for j in range(fc)] for i in range(pc)])

    # Выполняем операцию не более iter раз
    for i in range(iter):
        wh = w * h

        # Вычисляем текущую разность
        cost = difcost(v, wh)

        # Выходим из цикла, если матрица уже факторизована
        if cost == 0:
            break

        # Обновляем матрицу признаков (h)
        hn = (np.transpose(w) * v)
        hd = (np.transpose(w) * w * h)
        hd = np.where(hd == 0, 1e-10, hd)  # защита от деления на ноль
        h = np.matrix(np.array(h) * np.array(hn) / np.array(hd))

        # Обновляем матрицу весов (w)
        wn = (v * np.transpose(h))
        wd = (w * h * np.transpose(h))
        wd = np.where(wd == 0, 1e-10, wd)  # защита от деления на ноль
        w = np.matrix(np.array(w) * np.array(wn) / np.array(wd))

    return w, h
