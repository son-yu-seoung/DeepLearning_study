from PIL import Image
import os
from glob import glob
import numpy as np
from sklearn.utils import shuffle

class dataSet():

    def __init__(self, globalPath):
        self.globalPath = globalPath
        self.x = []
        self.y = []

        #in splitData
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
    
    def imageRead(self, path):
        x = Image.open(path)
        y = path.split('\\')[-2]
        #.\\dataset\\1\\1.jpg
        # print(x,y)
        return x, int(y)-1

    #실제로 모든 데이터를 읽어들이는 함수
    def getFilesInFolder(self, path):
        #모든 경로들을 다 가져와서 result에 넣음
        result = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]
        # print(result)

        for localPath in result:
            img, target = self.imageRead(localPath)
            self.x.append(img)
            self.y.append(target)
        # print(len(self.x), len(self.y))
        return self.x, self.y
    
    def resizeAll(self, X, Y, dim):
        
        resizedX = []
        resizedY = []

        N = len(X)

        for i in range(N):
            resized = X[i].resize((dim, dim))
            npImg = np.array(resized)

            if len(npImg.shape) == 3:
                resizedX.append(npImg)
                resizedY.append(Y[i])
           # print(npImg.shape)
        
        self.x = np.array(resizedX)
        self.y = np.array(resizedY)
        self.y = np.reshape(self.y, (-1, 1))
        #print(self.x.shape, self.y.shape)
    
    def splitDataset(self, ratio):
        train_idx = int(len(self.x) * ratio)
        print(train_idx)
        self.train_x, self.train_y = self.x[:train_idx, :, :, :], self.y[:train_idx, :]

        self.test_x, self.test_y = self.x[train_idx:, :, :, :], self.y[train_idx:, :]

        return self.train_x, self.train_y, self.test_x, self.test_y

    def shuffleData(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x, y = shuffle(x, y)
        return x, y

    #normalize Z-transform
    def normZT(self, x):
        x = (x - np.mean(x) / np.std(x))
        return x
    
    def normMinMax(self, x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

    def load_data(self, dim, ratio):
        self.getFilesInFolder(self.globalPath) #전체 데이터 가져옴
        self.resizeAll(self.x, self.y, dim) # numpy화 되어 있음
        self.x, self.y = self.shuffleData(self.x, self.y) #데이터 섞기
        self.splitDataset(ratio) #훈련용, 시험용으로 쪼개기
        self.train_x = self.normZT(self.train_x) #train 정규화
        self.test_x = self.normZT(self.test_x) #test 정규화

        return self.train_x, self.train_y, self.test_x, self.test_y



# globalPath = 'C:\\2021-son\\AnimalProject\\dataset\\'
# ds = dataSet(globalPath)
# train_x, train_y, test_x, test_y = ds.load_data(64, 0.8)
# np.save("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\train_X", train_x)
# np.save("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\train_Y", train_y)
# np.save("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\test_X", test_x)
# np.save("C:\\2021-son\\AnimalProject\\dataset\\Numpy\\test_Y", test_y)








