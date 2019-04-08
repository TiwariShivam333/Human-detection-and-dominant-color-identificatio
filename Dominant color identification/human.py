import numpy as np
import cv2
import math
import os


class ImageProcessing:
    inputSize = 0
    def grayscale(self, img):
        h = img.shape[0]
        w = img.shape[1]
        image = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                image[i][j] = round(img[i][j][2] * 0.299 + img[i][j][1] * 0.587 + img[i][j][0] * 0.114)
        return image

    def prewittOp(self, img):
        prewittOpHorz = np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]])

        prewittOpVert = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]])

        h = img.shape[0]
        w = img.shape[1]

        HorzGrad = np.zeros((h, w))
        VertGrad = np.zeros((h, w))

        for i in range(1, h - 1, 1):
            for j in range(1, w - 1, 1):
                x = 0
                for k in range(3):
                    for l in range(3):
                        x = x + ((img[i - 1 + k][j - 1 + l]) * prewittOpHorz[k][l])
                HorzGrad[i][j] = x / 3

        for i in range(1, h - 1, 1):
            for j in range(1, w - 1, 1):
                x = 0
                for k in range(3):
                    for l in range(3):
                        x = x + (img[i - 1 + k][j - 1 + l] * prewittOpVert[k][l])
                VertGrad[i][j] = x / 3

        GradAngle = np.zeros((h, w))

        for i in range(1, h - 1, 1):
            for j in range(1, w - 1, 1):
                if HorzGrad[i][j] == 0 and VertGrad[i][j] == 0:
                    GradAngle[i][j] = 0
                elif HorzGrad[i][j] == 0 and VertGrad[i][j] != 0:
                    GradAngle[i][j] = 90
                else:
                    x = math.degrees(math.atan(VertGrad[i][j] / HorzGrad[i][j]))
                    if x < 0:
                        x = 360 + x
                    if x >= 170 or x < 350:
                        x = x - 180
                    GradAngle[i][j] = x

        GradMagnitude = np.zeros((h, w), dtype='int')

        for i in range(1, h - 1, 1):
            for j in range(1, w - 1, 1):
                x = math.pow(HorzGrad[i][j], 2) + math.pow(VertGrad[i][j], 2)
                GradMagnitude[i][j] = int(round(math.sqrt(x / 2)))
        return GradAngle, GradMagnitude

    def hogc(self, GradAngle, GradMagnitude):
        h = GradAngle.shape[0]
        w = GradAngle.shape[1]

        cellHistogram = np.zeros((int(h / 8), int(w * 9 / 8)))

        tHist = np.zeros((1, 9))

        for i in range(0, h - 7, 8):
            for j in range(0, w - 7, 8):
                tHist = tHist * 0
                for k in range(8):
                    for l in range(8):
                        angle = GradAngle[i + k][j + l]
                        if -10 <= angle < 0:
                            dist = 0 - angle
                            tHist[0][0] = tHist[0][0] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][8] = tHist[0][8] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 0 <= angle < 20:
                            dist = angle
                            tHist[0][0] = tHist[0][0] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][1] = tHist[0][1] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 20 <= angle < 40:
                            dist = angle - 20
                            tHist[0][1] = tHist[0][1] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][2] = tHist[0][2] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 40 <= angle < 60:
                            dist = angle - 40
                            tHist[0][2] = tHist[0][2] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][3] = tHist[0][3] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 60 <= angle < 80:
                            dist = angle - 60
                            tHist[0][3] = tHist[0][3] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][4] = tHist[0][4] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 80 <= angle < 100:
                            dist = angle - 80
                            tHist[0][4] = tHist[0][4] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][5] = tHist[0][5] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 100 <= angle < 120:
                            dist = angle - 100
                            tHist[0][5] = tHist[0][5] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][6] = tHist[0][6] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 120 <= angle < 140:
                            dist = angle - 120
                            tHist[0][6] = tHist[0][6] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][7] = tHist[0][7] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 140 <= angle < 160:
                            dist = angle - 140
                            tHist[0][7] = tHist[0][7] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][8] = tHist[0][8] + dist * GradMagnitude[i + k][j + l] / 20
                        elif 160 <= angle < 170:
                            dist = angle - 160
                            tHist[0][8] = tHist[0][8] + (20 - dist) * GradMagnitude[i + k][j + l] / 20
                            tHist[0][0] = tHist[0][0] + dist * GradMagnitude[i + k][j + l] / 20
                cellHistogram[int(i / 8)][int(j * 9 / 8):int(j * 9 / 8 + 9)] = tHist
        return cellHistogram

    def hogblock(self, cellHistogram):
        h = cellHistogram.shape[0]
        w = cellHistogram.shape[1]

        hist = np.empty((int(h - 1), int((w / 9 - 1) * 36)))
        tHist1 = np.zeros((1, 36))

        for i in range(0, h - 1, 1):
            for j in range(0, w - 17, 9):
                l2Norm = 0
                for k in range(2):
                    for l in range(18):
                        l2Norm = l2Norm + math.pow(cellHistogram[i + k][j + l], 2)
                l2Norm = math.sqrt(l2Norm)
                x = 0
                for k in range(2):
                    for l in range(18):
                        if l2Norm == 0:
                            tHist1[0][x] = 0
                        else:
                            tHist1[0][x] = cellHistogram[i + k][j + l] / l2Norm
                        x = x + 1
                hist[i][int(j * 36 / 9):int(j * 36 / 9 + 36)] = tHist1
        hist = hist.flatten()
        ImageProcessing.inputSize = hist.shape[0]
        return hist


def reLu(num):
    if num <= 0:
        return 0
    else:
        return num


def reLuDeriv(num):
    if num <= 0:
        return 0
    else:
        return 1


class train:
    wt1 = None
    wt2 = None
    hiddenlayersize = 0
    hiddeninput = None
    flag = 0
    counter = 0
    squareErr = 0
    epochs = 0
    preVertr = None
    err = 0

    def training(self, hist):
        if train.flag == 0:
            train.hiddenlayersize = 64
            print("Training.....")
            train.wt1 = np.random.randn(train.hiddenlayersize, ImageProcessing.inputSize)
            train.wt1 = np.multiply(train.wt1, math.sqrt(2 / int(ImageProcessing.inputSize + train.hiddenlayersize)))
            train.wt2 = np.random.randn(train.hiddenlayersize)
            train.wt2 = train.wt2 * math.sqrt(1 / int(train.hiddenlayersize))
            train.flag = 1

        train.hiddeninput = np.empty(train.hiddenlayersize)

        train.hiddeninput = np.matmul(train.wt1, hist)
        sigmoidInput = np.matmul(list(map(reLu, train.hiddeninput)), train.wt2)
        observedOutput = 1 / (1 + np.exp(-sigmoidInput))
        return observedOutput

    def backpropagation(self, hist, observedOutput, labels):

        err = labels - observedOutput
        x = (-err) * observedOutput * (1 - observedOutput)

        a = np.multiply(train.wt2, x)
        b = np.multiply(a, list(map(reLuDeriv, train.hiddeninput)))
        change1 = np.matmul(b.reshape(train.hiddenlayersize, 1), hist.reshape(1, ImageProcessing.inputSize))
        train.wt1 = np.subtract(train.wt1, np.multiply(change1, 0.1))

        change2 = np.multiply(list(map(reLu, train.hiddeninput)), 0.1 * x)
        train.wt2 = np.subtract(train.wt2, change2)

        train.squareErr = train.squareErr + math.pow(err, 2)
        train.counter = train.counter + 1
        if train.counter == 20:
            train.counter = 0
            train.epochs = train.epochs + 1
            if train.epochs == 1:
                train.preVertr = train.squareErr
            else:
                train.err = train.preVertr - train.squareErr
                train.preVertr = train.squareErr
            train.squareErr = 0


class NeuralNetworkTest:
    def testImage(self, hist):
        hiddeninput = np.matmul(train.wt1, hist)
        sigmoidInput = np.matmul(list(map(reLu, hiddeninput)), train.wt2)
        observedOutput = 1 / (1 + np.exp(-sigmoidInput))
        return observedOutput



images = []
posimg=os.listdir('train_pos')
negimg=os.listdir('train_neg')
for i in range(10):
    images.append('/train_pos/' + posimg[i])
    images.append('/train_neg/' + negimg[i])
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

obj1 = ImageProcessing()
obj2 = train()
obj3 = NeuralNetworkTest()
counter = 0
flag = 0
while train.err > 0.0005 or train.epochs <= 50:
    for i in range(20):
        if counter < 20:
            img = cv2.imread('.' + images[i])
            image = obj1.grayscale(img)
            GradMagnitude, GradAngle = obj1. prewittOp(image)
            cellHistogram = obj1.hogc(GradMagnitude, GradAngle)
            tHist1 = obj1.hogblock(cellHistogram)
            if flag == 0:
                flag = 1
                hist = np.empty((20, ImageProcessing.inputSize))
            hist[i][:] = tHist1
            counter = counter + 1
        observedOutput = obj2.training(hist[i][:])
        obj2.backpropagation(hist[i][:], observedOutput, labels[i])

print("Training Complete! :D")
print('Total epochs :', train.epochs)
print('Final error :', train.preVertr)
print()

testImages = []
pos = os.listdir('test_pos')
neg = os.listdir('test_neg')
for i in range(5):
    testImages.append('/test_pos/' + pos[i])
    testImages.append('/test_neg/' + neg[i])
labelsTest = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

correct = 0
wrong = 0
for i in range(10):
    imgTest = cv2.imread('.' + testImages[i])
    imageTest = obj1.grayscale(imgTest)
    GradMagnitudeTest, GradAngleTest = obj1.prewittOp(imageTest)
    cellHistogramTest = obj1.hogc(GradMagnitudeTest, GradAngleTest)
    histTest = obj1.hogblock(cellHistogramTest)
    observed = obj3.testImage(histTest)

    print('Image ->',(i+1), testImages[i])

    if observed >= 0.5:
        print('Human is present in the image.')
    else:
        print('Human is not present in the image.')
    print()

    if 0 < abs(labelsTest[i] - observed) < 0.5:
        correct += 1
    if 0.5 < abs(observed - labelsTest[i]) < 1:
        wrong += 1

print("Correct Predictions :", correct)
print("Incorrect Predictions :", wrong)
