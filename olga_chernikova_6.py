import scipy.io
from sklearn.cross_validation import train_test_split
import numpy as np
import math
import random
from PIL import Image


def scale_to_unit_interval(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]
    H, W = img_shape
    Hs, Ws = tile_spacing

    dt = X.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                this_x = X[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    this_img = scale_to_unit_interval(
                        this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)
                c = 1
                if output_pixel_vals:
                    c = 255
                out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
    return out_array

def visualize_mnist(train_X):
    images = train_X[0:2500, :]
    image_data = tile_raster_images(images,
                                    img_shape=[28,28],
                                    tile_shape=[50,50],
                                    tile_spacing=(2,2))
    im_new = Image.fromarray(np.uint8(image_data))
    im_new.save('mnist.png')



def f(x):
    return 1/(1 + math.exp(-x))

def df(x):
    return f(x)*(1 - f(x))

class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        #w[номер слоя][номер нейрона в слое][номер нейрона в следующем слое] = вес
        #нейрон с -1 имеет последний номер в каждом слое. 
        self.w = []
        for i in range(self.num_layers - 1):
            self.w.append(np.zeros((self.layers[i] + 1, self.layers[i + 1])))

        self.val = []
        self.dv = []
        self.err = []
        for i in range(self.num_layers):
            self.val.append(np.zeros(self.layers[i]))
            self.dv.append(np.zeros(self.layers[i]))
            self.err.append(np.zeros(self.layers[i]))

    def train(self, X, y, max_iter=10000, learning_rate=1):
        for i in range(self.num_layers - 1):
            self.w[i] = np.random.rand(self.layers[i] + 1, self.layers[i + 1]) * (1/(len(X[0]))) - 1/(2*len(X[0]))

        
        for j in range(max_iter):
            if j % 100 == 0:
                print(j)
            i = np.random.randint(0, len(X))
            self.forward(X[i])
            self.backward(X[i], y[i])

            for h in range(self.num_layers - 1):
                for n1 in range(self.layers[h]):
                    self.w[h][n1] = (self.w[h][n1] - learning_rate * (self.err[h + 1] * self.dv[h + 1]) * self.val[h][n1])
                        
            

    def forward(self, X):
        for i in range(len(X)):
            self.val[0][i] = X[i]

        for i in range(1, self.num_layers):
            for j in range(self.layers[i]):
                cur_val = self.w[i - 1][self.layers[i - 1]][j]
                cur_val += np.sum(self.w[i - 1].T[j][:-1] * self.val[i - 1])
               
                self.dv[i][j] = df(cur_val)
                self.val[i][j] = f(cur_val)
        

    def backward(self, X, y):
        ans = [0]*len(self.val[-1])
        ans[y] = 1
        for i in range(len(self.val[-1])):
            self.err[-1][i] = self.val[-1][i] - ans[i]

        for i in range(len(self.val) - 2, 0, -1):
            for j in range(len(self.val[i])):
                self.err[i][j] = np.sum((self.err[i + 1] * self.dv[i + 1]) * self.w[i][j])
    

    def predict(self, X):
        ans = []
        print(len(X))
        for x in X:
            self.forward(x)
            res = 0
            for i in range(len(self.val[-1])):
                if self.val[-1][i] >= self.val[-1][res]:
                    res = i
            ans.append(res)
        return ans


def quality(y1, y2):
    cntr = 0
    for i in range(len(y1)):
        if y1[i] == y2[i]:
            cntr += 1
    print(cntr)
    return cntr/len(y1)
        

dataset = scipy.io.loadmat('mnist-original.mat')
trainX, testX, trainY, testY = train_test_split(
    dataset['data'].T / 255.0, dataset['label'].squeeze().astype("int0"), test_size = 0.3)

visualize_mnist(trainX)

for cnt_mid in range(5, 50, 5):
    nn = NeuralNetwork([trainX.shape[1], cnt_mid, cnt_mid, 10])
    nn.train(trainX, trainY)

    print(cnt_mid, quality(nn.predict(testX), testY))
