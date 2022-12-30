# Подключение библиотек
from os import listdir, environ

import tensorflow
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D, \
    concatenate, Embedding, multiply, ReLU, Concatenate  # Базовые слои keras
from keras.layers.advanced_activations import LeakyReLU  # LeakyReLU - функция активации
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose  # Сверточные слои keras
from keras.models import Sequential, Model  # Формат модели keras
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
from tensorflow import random_normal_initializer

import matplotlib.pyplot as plt  # Отображение графиков
import matplotlib.ticker as ticker

import numpy as np  # Numpy массивы
from keras_preprocessing.image import load_img, img_to_array
from numpy import asarray, log10
from sklearn.utils import shuffle

environ["CUDA_VISIBLE_DEVICES"]="-1"

img_size_h = img_size_w = 128
savedImgDir = 'EpochProgress'
generateMapTo = 'UnityRenderer\\Diploma_Work\\Assets\\generatedMaps'
dataset = '.\\DataGenerator\\DataGenerator\\out\\mountains_128x128'


def load_data(path):
    x_train = list()
    temp_x = load_images_from_dir(path + '\\mountains')
    x_train.extend(temp_x)

    return (asarray(x_train))


# load all images in a directory into memory
def load_images_from_dir(path):
    pic_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + '\\' + filename, color_mode='grayscale')
        # convert to numpy array
        pixels = img_to_array(pixels)
        pixels = (pixels.astype(np.float32) - 127.5) / 127.5
        pic_list.append(pixels)

    return asarray(pic_list)


class GAN():
    def __init__(self, isConv):
        self.img_rows = img_size_h  # Высота изображения в пикселях в MNIST
        self.img_cols = img_size_w  # Ширина изображения в пикселях в MNIST
        self.channels = 1  # Количество каналов (глубина изображения)
        self.latent_dim = img_size_h*img_size_w*self.channels  # Размерность скрытого пространства
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        if (isConv):
            self.generator = self.build_conv_generator()
            self.discriminator = self.build_conv_discriminator(self.img_shape)
        else:
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator(self.img_shape)

        self.build_GAN()

    def build_GAN(self):
        optimizer = adam_v2.Adam(learning_rate=0.0002, beta_1=0.5)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(img_size_w, img_size_h, self.channels), name="GAN_In")  # Создаем слой Input размерностью latent_dim. На входе подается шум, а на выходе - сгенерированные изображения.
        #z = Input(img_size_w, img_size_h, self.channels)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False  # Замораживаем обучение дискриматора, чтобы в объединенной модели тренировался только генератор

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(
            img)  # Создаем модель дискриминатора-критика. На вход подаются сгенерированные изображения, а на выходе вероятность распознавания исходных изображений

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid,
                              name="GAN")  # В объединенной модели на входе шум, а на выходе вероятность насколько генератор-творец обманул дискриминатора-критика
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.combined.summary()

    def build_conv_discriminator(self, img_shape):  # Функция создания сверточного дискрминатора
        model = Sequential()  # Инициализируем модель currDisc

        print(img_shape)
        model.add(Dense(1, input_shape=img_shape, name="Discriminator_In"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(6, (3, 3), strides=1,
                         padding="same"))  # Создаем слой  Conv2D (размерность входных данных (img_shape), ядро свертки = 4, окно свертки = (3,3))
        model.add(LeakyReLU(alpha=0.2))  # Добавляем слой активационной функции с параметром 0.2
        model.add(Conv2D(6, (5, 5), strides=1,
                         padding="same"))  # Создаем слой  Conv2D (размерность входных данных (img_shape), ядро свертки = 4, окно свертки = (3,3))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(6, (7, 7), strides=1,
                         padding="same"))  # Создаем слой  Conv2D (размерность входных данных (img_shape), ядро свертки = 4, окно свертки = (3,3))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())  # Добавляем слой Flatten ()
        model.add(Dense(12))  # Добавляем Dense-слой на 512 нейронов
        model.add(LeakyReLU(alpha=0.2))  # Добавляем слой активационной функции с параметром 0.2
        model.add(Dense(1, activation='sigmoid',
                        name="Discriminator_Out"))  # Добавляем Dense-слой c 1 нейроном с активационной функцией sigmoid, поскольку нам нужно категорировать входноые изображения на два класса 1 - из тестовой выборки и 0 - сформирован генератором.

        img = Input(
            shape=img_shape)  # Создаем слой Input (записываем входные данные размерностью (img_rows, img_cols, channels) в img)
        validity = model(
            img)  # Записываем в переменную validity значение, возвращаемое currDisc'ом с входным параметром img

        model.summary()
        plot_model(model, to_file='discriminator_conv_arch.png', show_shapes=True, show_layer_names=True)
        discriminator_conv = Model(inputs=img,
                                   outputs=validity)  # Создаем модель discriminator_conv (входные данные: img, выходные данные: validity)

        return discriminator_conv


    def downsample(self, filters, size, apply_batchnorm = True):
        initializer = random_normal_initializer(0., 0.02)

        result = Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(BatchNormalization())

        result.add(LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = random_normal_initializer(0., 0.02)

        result = Sequential()
        result.add(
            Conv2DTranspose(filters, size, strides=2,
                            padding='same',
                            kernel_initializer=initializer,
                            use_bias=False))

        result.add(BatchNormalization())

        if apply_dropout:
            result.add(Dropout(0.5))

        result.add(ReLU())

        return result

    def build_conv_generator(self):

        inputs = Input(shape=[128, 128, 1], name="GAN_In")

        down_stack = [
            self.downsample(64, 4, apply_batchnorm = False),  # (batch_size, 128, 128, 64)
            self.downsample(128, 4),  # (batch_size, 64, 64, 128)
            self.downsample(256, 4),  # (batch_size, 32, 32, 256)
            self.downsample(512, 4),  # (batch_size, 16, 16, 512)
            self.downsample(512, 4),  # (batch_size, 8, 8, 512)
            self.downsample(512, 4),  # (batch_size, 4, 4, 512)
            self.downsample(512, 4),  # (batch_size, 2, 2, 512)
            #self.downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout = True),  # (batch_size, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout = True),  # (batch_size, 4, 4, 1024)
            #self.upsample(512, 4, apply_dropout = True),  # (batch_size, 8, 8, 1024)
            self.upsample(512, 4),  # (batch_size, 16, 16, 1024)
            self.upsample(256, 4),  # (batch_size, 32, 32, 512)
            self.upsample(128, 4),  # (batch_size, 64, 64, 256)
            self.upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = random_normal_initializer(0., 0.02)
        last = Conv2DTranspose(1, 4, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            #x = Concatenate()([x, skip])

        x = last(x)

        model = Model(inputs=inputs, outputs=x)

        model.summary()

        plot_model(model, to_file='generator_conv_arch.png', show_shapes=True, show_layer_names=True)

        return model



    def train(self, epochs, batch_size=128, save_interval=1000, save_images=False, tendention_С=None):
        X_train = load_data(dataset)
        # X_train = X_train.reshape(-1, img_size_w, img_size_h, 1)
        X_train = shuffle(X_train)

        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5                  # Масштабируем значение в диапазон от -1 до 1, поскоьлку активационная функция tanh, у которого значения лежат от -1 до +1
        # X_train = np.expand_dims(X_train, axis = 3)                             # Добавляем третью размерность для X_train ((28,28) => (28,28,1))

        valid = np.ones((batch_size, 1))  # Создаем массив единиц длинной batch_size
        fake = np.zeros((batch_size, 1))  # Создаем массив нулей длинной batch_size

        # tendention_plot, axs, tendention_R, tendention_C = self.create_tendention_plot(save_interval, epochs)
        curr_colomn = 0
        d_loss_list = []
        g_loss_list = []
        acc_list = []
        loss_acc_plt_step = 5
        step_summ_g_loss = 0
        step_summ_d_loss = 0
        step_summ_acc = 0
        gen_imgs = []

        for epoch in range(epochs):

            # ---Train Discriminator---
            idx = np.random.randint(0, X_train.shape[0],
                                    batch_size)  # Выбираем случайным образом batch_size картинок из исходной обучающей выбрки для тренировки дискриминатора
            imgs = X_train[idx]  # В переменную imgs записываем значение из X_train с индексами из idx

            gen_imgs.clear()
            for i in range(batch_size):
                noise_img = np.random.normal(0, 1, self.latent_dim)  # Формируем массив размерностью (batch_size, self.latent_dim) из нормально распределенных значений
                noise_img = noise_img.reshape(img_size_h, img_size_w, self.channels)
                gen_imgs.append(self.generator.predict(noise_img))  # Формируем массив изображений с помощью входной переменной generator


            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(imgs,
                                                            valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,
                                                            fake)
            d_loss = np.add(d_loss_real,
                            d_loss_fake) / 2

            # --- Train Generator ---
            noise = np.random.normal(0, 1, (batch_size,
                                            self.latent_dim))
            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch(x=noise,
                                                  y=valid)

            step_summ_g_loss += d_loss[0]
            step_summ_d_loss += g_loss
            step_summ_acc += 100 * d_loss[1]

            if (epoch % loss_acc_plt_step == 0):
                acc_list = np.append(acc_list, step_summ_acc / loss_acc_plt_step)
                d_loss_list = np.append(d_loss_list, step_summ_d_loss / loss_acc_plt_step)
                g_loss_list = np.append(g_loss_list, step_summ_g_loss / loss_acc_plt_step)
                step_summ_g_loss = 0
                step_summ_d_loss = 0
                step_summ_acc = 0

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[1], g_loss))  # Plot the progress

            if (epoch % save_interval == 0) or (
                    epoch == epochs - 1):  # Выводим/сохраняем изображения каждые save_interval эпох и в конце цикла
                self.save_imgs(epoch, save_images)
                # self.save_tendention(tendention_plot, axs, tendention_C, curr_colomn)
                curr_colomn += 1
            # self.close_tendention(tendention_plot)

        x_lin = np.linspace(0, epochs - 1, epochs // loss_acc_plt_step)
        fig, axs = plt.subplots(2, figsize=(300, 10), dpi=100)

        # fig.subplots_adjust(bottom=spacing)
        axs[0].plot(x_lin, log10(d_loss_list), linewidth=1, color='b', label='d_loss')
        axs[0].plot(x_lin, log10(g_loss_list), linewidth=1, color='r', label='g_loss')
        axs[0].set(ylabel='loss_value', xlabel='epoch')
        axs[0].legend(loc='upper right')
        axs[0].xaxis.set_major_locator(ticker.MultipleLocator(200))
        axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(50))
        axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.15))
        axs[0].grid()

        axs[1].plot(x_lin, acc_list, 'red')
        axs[1].set(ylabel='discriminator accuracy (%)', xlabel='epoch')
        axs[1].xaxis.set_major_locator(ticker.MultipleLocator(200))
        axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(50))
        axs[1].yaxis.set_major_locator(ticker.MultipleLocator(25))
        axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(10))
        axs[1].grid()
        fig.tight_layout()
        fig.savefig('acc_and_loss.png')

        self.GenPredict(5)

    def save_imgs(self, epoch, save_images=False):
        r, c = 5, 5  # Параметры вывода: r - количество строк, c - количество столбцов
        noise = np.random.normal(0, 1, (r * c,
                                        self.latent_dim))  # Создаем вектор размерностью (r*c, latent_dim (25,100)) из нормально распределенных значений
        gen_imgs = self.generator.predict(
            noise)  # Формируем генератором картинку на основании случайного входного сигнала
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5  # Выполняем обратное преобразование полученных значений изображения в диапазон от 0 до 1
        # Функция активации tanh, поэтому значения на выходе генератора лежат в диапазоне от -1 до + 1.
        fig, axs = plt.subplots(r, c)  # Создаем окно вывода r на c (5 на 5) ячеек
        cnt = 0  # Порядковый номер картинки
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0],
                                 cmap='gray')  # Записываем в axs[i,j] сгенерироввнное изображение из gen_img с индексом cnt
                axs[i, j].axis('off')  # Отключаем печать названия осей
                cnt += 1  # Увеличиваем индекс изображения
        # plt.show()                                                              # Выводим сгенерированные изображения
        if (save_images == True):
            fig.savefig(savedImgDir + '\\gan_generated_image_epoch_%d.png' % epoch)  # Сохраняем изображения
        plt.close()

    def create_tendention_plot(self, save_interval, epoch_count):
        r = 1 + epoch_count // save_interval
        c = 1 + epoch_count % save_interval  # Параметры вывода: r - количество строк, c - количество столбцов
        print(r, c)
        fig, axs = plt.subplots(r, c)
        return fig, axs, r, c

    def save_tendention(self, fig, axs, all_colomns, cur_colomn):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_img = self.generator.predict(noise)
        gen_img = 0.5 * gen_img + 0.5

        curY, curX = cur_colomn // all_colomns, cur_colomn % all_colomns

        axs[curY, curX].imshow(gen_img,
                               cmap='gray')  # Записываем в axs[i,j] сгенерироввнное изображение из gen_img с индексом cnt
        axs[curY, curX].axis('off')  # Отключаем печать названия осей
        # Увеличиваем индекс изображения
        # plt.show()                                                              # Выводим сгенерированные изображения

    def close_tendention(self, fig):
        fig.savefig('tendence.png')  # Сохраняем изображения
        plt.close()

    def save(self, name):
        self.generator.save(name + '\\gen')
        self.discriminator.save(name + '\\dis')

    def load(self, dir):
        self.generator.load_weights(dir + '\\gen')
        self.generator.load_weights(dir + '\\dis')

    def GenPredict(self, count):
        noise = np.random.normal(0, 1,
                                 self.latent_dim)  # Создаем вектор размерностью (r*c, latent_dim (25,100)) из нормально распределенных значений
        for i in range(count):
            gen_img = self.generator.predict(noise)
            fig = plt.plot(gen_img)
            fig.savefig(generateMapTo + "\\predictedImg_" + str(i) + ".png")


# gan = GAN(None, None)
# gan.train(epochs=20000, batch_size=128, save_interval = 250, save_images = True)

def trainGan():
    gan = GAN(True)
    gan.train(epochs=80000, batch_size=256, save_interval=300, save_images=True)
    gan.save('gan_weights')


def loadGanAndGen():
    loaded_model = tensorflow.keras.models.load('gan_weights')
    loaded_model.GenPredict(3)


trainGan()
# loadGanAndGen()
