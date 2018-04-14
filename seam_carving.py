import time
from functools import wraps

import cv2 as cv
import numpy as np


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" % (function.__name__,
                                                     str(t1 - t0)))
        return result

    return function_timer


# @fn_timer
def sobel_energy(img):
    blurred = cv.GaussianBlur(img, (3, 3), 0, 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    dx = cv.Sobel(
        gray,
        cv.CV_64F,
        1,
        0,
        ksize=3)
    dy = cv.Sobel(
        gray,
        cv.CV_64F,
        0,
        1,
        ksize=3)
    return cv.add(np.absolute(dx), np.absolute(dy))


# @fn_timer
def canny_energy(img):
    blurred = cv.GaussianBlur(img, (5, 5), 0, 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    energy = cv.Canny(gray, 10, 30)
    energy = cv.GaussianBlur(energy, (3, 3), 0, 0)
    return energy


# @fn_timer
def get_energy(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    energies = np.zeros((height, width), np.float16)
    gray = np.insert(gray, height, 0, axis=0)
    gray = np.insert(gray, width, 0, axis=1)
    for x in range(height):
        energies[x] += np.fabs(gray[x, :-1] - gray[x, 1:])
        energies[x] += np.fabs(gray[x] - gray[x + 1])[:-1]
    return energies[:-1, :-1]


# @fn_timer
def cumulate(energy):
    height, width = energy.shape
    energy = energy.astype("float64")
    energy = np.insert(energy, width, 1e6, axis=1)
    energy = np.insert(energy, 0, 1e6, axis=1)
    height, width = energy.shape
    # energy[0, :] = 0
    for i in range(0, height - 1):
        x1 = energy[i, 0:-2]
        x2 = energy[i, 1:-1]
        x3 = energy[i, 2:]
        xmin = np.append([x1], [x2, x3], axis=0)
        xmin = np.transpose(xmin)
        xmin = xmin.min(1)
        energy[i + 1, 1:-1] += xmin
    return energy[:, 1:-1]


def vertical_seam(energies):
    height, width = energies.shape[:2]
    previous = 0
    seam = []
    for i in range(height - 1, -1, -1):
        row = energies[i, :]
        if i == height - 1:
            previous = np.argmin(row)
        else:
            left = row[previous - 1] if previous - 1 >= 0 else 1e6
            middle = row[previous]
            right = row[previous + 1] if previous + 1 < width else 1e6
            previous = previous + np.argmin([left, middle, right]) - 1
        seam.append([previous, i])
    return seam


def draw_seam(img, seam):
    cv.polylines(img, np.int32([np.asarray(seam)]), False, (255, 255, 0))
    cv.imshow('seam', img)
    cv.waitKey(1)
    # cv.waitKey(0)


def draw_horizontal_seam(img, seam):
    cv.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 255))
    img = cv.transpose(img)
    cv.imshow('seam', img)
    cv.waitKey(1)
    # cv.waitKey(0)


def remove_seam(img, seam):
    height, width, bands = img.shape
    removed = np.zeros((height, width - 1, bands), np.uint8)
    for x, y in seam:
        removed[y, 0:x] = img[y, 0:x]
        removed[y, x:width - 1] = img[y, x + 1:width]
    return removed


def resize(img, width=None, height=None):
    result = img
    img_height, img_width = img.shape[:2]
    dy = img_height - height if img_height - height > 0 else 0
    dx = img_width - width if img_width - width > 0 else 0

    for i in range(dx):
        energies = cumulate(sobel_energy(result))
        # energies = cumulate(canny_energy(result))
        # energies = cumulate(get_energy(result))
        seam = vertical_seam(energies)
        draw_seam(result, seam)
        result = remove_seam(result, seam)

    result = cv.transpose(result)

    for i in range(dy):
        energies = cumulate(sobel_energy(result))
        # energies = cumulate(canny_energy(result))
        # energies = cumulate(get_energy(result))
        seam = vertical_seam(energies)
        draw_horizontal_seam(result, seam)
        result = remove_seam(result, seam)

    result = np.transpose(result, (1, 0, 2))
    cv.imwrite('resized.jpg', result)
    cv.imshow('seam', result)


@fn_timer
def run(img):
    img_height, img_width = img.shape[:2]

    resize(
        img,
        width=img_width - int(float(img_width) / 5),
        height=img_height - int(float(img_height) / 8))


if __name__ == '__main__':
    img = cv.imread('img/t.jpg')
    cv.namedWindow('origin', cv.WINDOW_NORMAL)
    cv.imshow('origin', img)
    cv.namedWindow('seam', cv.WINDOW_NORMAL)

    # cv.namedWindow('1', cv.WINDOW_NORMAL)
    # cv.imshow('1', canny_energy(img))
    # cv.namedWindow('11', cv.WINDOW_NORMAL)
    # cv.imshow('11', cumulate(canny_energy(img)).astype("uint8"))

    cv.namedWindow('sobel energy', cv.WINDOW_NORMAL)
    cv.imshow('sobel energy', sobel_energy(img).astype("uint8"))
    # cv.namedWindow('22', cv.WINDOW_NORMAL)
    # cv.imshow('22', cumulate(sobel_energy(img)).astype("uint8"))

    run(img)

    while (1):
        k = cv.waitKey(0) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()
