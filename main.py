def build_conv_discriminator(self, img_shape):  # Функция создания сверточного дискрминатора
    model = Sequential()  # Инициализируем модель currDisc

    print(img_shape)
    model.add(Dense(6, input_shape=img_shape, name="Discriminator_In"))
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

def build_conv_generator(self):
    noise_shape = (self.latent_dim,)
    model = Sequential()
    model.add(Dense(img_size_h * img_size_h * 6, use_bias=False, input_shape=noise_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((img_size_h, img_size_h, 6)))
    assert model.output_shape == (None, img_size_h, img_size_w, 6)  # Note: None is the batch size

    model.add(Conv2DTranspose(6, (7, 7), strides=1, padding='same', use_bias=False))
    print(model.output_shape)
    assert model.output_shape == (None, img_size_h, img_size_w, 6)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(6, (5, 5), strides=1, padding='same', use_bias=False))
    print(model.output_shape)
    assert model.output_shape == (None, img_size_h, img_size_w, 6)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(6, (3, 3), strides=1, padding='same', use_bias=False, activation='tanh'))
    print(model.output_shape)
    assert model.output_shape == (None, img_size_h, img_size_w, 6)

    model.add(Dense(12, activation='tanh'))

    model.add(Dense(1, activation='tanh'))
    model.add(Reshape(self.img_shape, name="Generator_Out"))

    model.summary()

    plot_model(model, to_file='generator_conv_arch.png', show_shapes=True,
               show_layer_names=True)  # Записываем в переменную img значение, возвращаемое generator'ом  с входным параметром noise

    return model