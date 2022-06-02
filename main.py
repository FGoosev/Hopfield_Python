import numpy as np
import random
from PIL import Image
import os
import re


def convertMatrixToVector(x):
    m = x.shape[0] * x.shape[1]
    tmp = np.zeros(m)
    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp[c] = x[i, j]
            c += 1
    return tmp


def createWeight(x):
    if len(x.shape) != 1:
        print("Введенные данные - не вектор!")
        return
    else:
        w = np.zeros([len(x), len(x)])
        for i in range(len(x)):
            for j in range(i, len(x)):
                if i == j:
                    w[i, j] = 0
                else:
                    w[i, j] = x[i] * x[j]
                    w[j, i] = w[i, j]
    print(w)
    return w


def readImg(file, size, threshold = 145):
    img = Image.open(file).convert(mode="L")
    img = img.resize(size)
    imgArray = np.asarray(img, dtype=np.uint8)
    x = np.zeros(imgArray.shape, dtype=np.float16)
    x[imgArray > threshold] = 1
    x[x == 0] = -1
    return x

def arrayToImg(data, outfile = None):
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    image = Image.fromarray(y, mode="L")
    if outfile is not None:
        image.save(outfile)
    return image

def update(w, yVec, theta=0.5, time=100):
    for s in range(time):
        m = len(yVec)
        i = random.randint(0,m-1)
        u = np.dot(w[i][:], yVec) - theta

        if u > 0:
            yVec[i] = 1
        elif u < 0:
            yVec[i] = -1
    return yVec

def hopfield(train_files, test_files, theta=0.5, time=1000, size=(100,100), threshold=60, current_path=None):

    print("Загрузка изображения и создание весов матрицы....")
    num_files = 0
    for path in train_files:
        x = readImg(file=path, size=size, threshold=threshold)
        x_vec = convertMatrixToVector(x)
        if num_files == 0:
            w = createWeight(x_vec)
            num_files = 1
        else:
            tmp_w = createWeight(x_vec)
            w = w + tmp_w
            num_files += 1

    print("Веса созданы!!")
    #Import test data
    counter = 0
    for path in test_files:
        y = readImg(file=path, size=size, threshold=threshold)
        oshape = y.shape
        #y_img = arrayToImg(y)
        #y_img.show()
        print("Загрузка тестовых данных")

        yVec = convertMatrixToVector(y)
        print("Обновление...")
        yVec_after = update(w=w, yVec=yVec, theta=theta, time=time)
        yVec_after = yVec_after.reshape(oshape)
        if current_path is not None:
            outfile = current_path+"/after_"+str(counter)+".jpeg"
            after = arrayToImg(yVec_after, outfile=outfile)
            after.show()
        else:
            after_img = arrayToImg(yVec_after, outfile=None)
            after_img.show()
        counter += 1



current_path = os.getcwd()
train_paths = []
path = current_path+"/train/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-]*.jp[e]*g', i):
        train_paths.append(path+i)


test_paths = []
path = current_path+"/test/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g', i):
        test_paths.append(path+i)


hopfield(train_files=train_paths, test_files=test_paths, theta=0.5, time=20000, size=(100,100), threshold=60, current_path=current_path)