from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import zipfile
import numpy as np
from PIL import Image
import os
import xml.etree.ElementTree as ET


class DogGenerator:
    def __init__(self,image_width,image_height,channels):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels

        self.image_shape = (self.image_width,self.image_height,self.channels)
        self.random_noise_dimension = 100
        optimizer = Adam(0.0002,0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        self.generator = self.build_generator()
        random_input = Input(shape=(self.random_noise_dimension,))
        generated_image = self.generator(random_input)
        self.discriminator.trainable = False
        validity = self.discriminator(generated_image)
        self.combined = Model(random_input,validity)
        self.combined.compile(loss="binary_crossentropy",optimizer=optimizer)
        self.trainingImages = ""
    def get_training_data(self,datafolder):
        print("Loading training data...")

        training_data = []
        #Finds all files in datafolder
        filenames = os.listdir(datafolder)

        for filename in filenames:
            path = os.path.join(datafolder,filename)
            image = Image.open(path)
            image = image.resize((self.image_width,self.image_height),Image.ANTIALIAS)
            pixel_array = np.asarray(image)

            training_data.append(pixel_array)
        training_data = np.reshape(training_data,(-1,self.image_width,self.image_height,self.channels))
        print (training_data.shape)
        return training_data


    def build_generator(self):
        model = Sequential()

        model.add(Dense(256*4*4,activation="relu",input_dim=self.random_noise_dimension))
        model.add(Reshape((4,4,256)))
        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels,kernel_size=3,padding="same"))
        model.add(Activation("tanh"))
        print ('build_generator : ')
        model.summary()
        input = Input(shape=(self.random_noise_dimension,))
        generated_image = model(input)
        return Model(input,generated_image)


    def build_discriminator(self):
        model = Sequential()
        print ('ttt',self.image_shape)
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        print ('build_discriminator : ')
        model.summary()
        input_image = Input(shape=self.image_shape)
        validity = model(input_image)

        return Model(input_image, validity)

    def train(self, epochs,batch_size):
        training_data = self.trainingImages
        training_data = training_data / 127.5 - 1.
        labels_for_real_images = np.ones((batch_size,1))
        labels_for_generated_images = np.zeros((batch_size,1))

        for epoch in range(epochs):
            indices = np.random.randint(0,training_data.shape[0],batch_size)
            real_images = training_data[indices]
            random_noise = np.random.normal(0,1,(batch_size,self.random_noise_dimension))
            generated_images = self.generator.predict(random_noise)
            discriminator_loss_real = self.discriminator.train_on_batch(real_images,labels_for_real_images)
            discriminator_loss_generated = self.discriminator.train_on_batch(generated_images,labels_for_generated_images)
            discriminator_loss = 0.5 * np.add(discriminator_loss_real,discriminator_loss_generated)

            generator_loss = self.combined.train_on_batch(random_noise,labels_for_real_images)
            print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, discriminator_loss[0], 100*discriminator_loss[1], generator_loss))

    def save_images3(self,  samples=10,zipFileName='images.zip'):
        zipf = zipfile.ZipFile(outFile, 'w', zipfile.ZIP_DEFLATED)
        zipf.close()
        noise = np.random.normal(0, 1, size=[samples, self.random_noise_dimension])
        images = self.generator.predict(noise)
        for i in range(images.shape[0]):
            image = images[i, :, :, :]
            image = np.reshape(image, [self.image_height,self.image_width,self.channels])
            fileName1 = 'dogFake_' + str(i) + '.png'
            image = Image.fromarray((127.5*(0.5*image+0.5)).astype('uint8'))
            image.save(fileName1, 'PNG')
            z = zipfile.ZipFile(zipFileName, "a", zipfile.ZIP_DEFLATED)
            z.write(fileName1)
            z.close()
            os.remove(fileName1)

    def crop(self,trainingDataLoc):
        ROOT = trainingDataLoc
        # list of all image file names in all-dogs
        IMAGES = os.listdir(ROOT + 'all-dogs')
        # list of all the annotation directories, each directory is a dog breed
        breeds = os.listdir(ROOT + 'Annotation/Annotation/')

        idxIn = 0;
        namesIn = []
        imagesIn = np.zeros((25000, 64, 64, 3))

        # CROP WITH BOUNDING BOXES TO GET DOGS ONLY
        # iterate through each directory in annotation
        for breed in breeds:
            # iterate through each file in the directory
            for dog in os.listdir(ROOT + 'annotation/Annotation/' + breed):
                try:
                    img = Image.open(ROOT + 'all-dogs/all-dogs/' + dog + '.jpg')
                except:
                    continue
                # Element Tree library allows for parsing xml and getting specific tag values
                tree = ET.parse(ROOT + 'annotation/Annotation/' + breed + '/' + dog)
                # take a look at the print out of an xml previously to get what is going on
                root = tree.getroot()  # <annotation>
                objects = root.findall('object')  # <object>
                for o in objects:
                    bndbox = o.find('bndbox')  # <bndbox>
                    xmin = int(bndbox.find('xmin').text)  # <xmin>
                    ymin = int(bndbox.find('ymin').text)  # <ymin>
                    xmax = int(bndbox.find('xmax').text)  # <xmax>
                    ymax = int(bndbox.find('ymax').text)  # <ymax>
                    w = np.min((xmax - xmin, ymax - ymin))
                    img2 = img.crop((xmin, ymin, xmin + w, ymin + w))
                    img2 = img2.resize((64, 64), Image.ANTIALIAS)
                    imagesIn[idxIn, :, :, :] = np.asarray(img2)
                    namesIn.append(breed)
                    idxIn += 1

        self.trainingImages = imagesIn

if __name__ == '__main__':
    outFile = 'images.zip'
    trainingDataLoc = 'C://Work//dataset//generative-dog-images//generative-dog-images//'

    doggenerator = DogGenerator(64,64,3)
    doggenerator.crop(trainingDataLoc=trainingDataLoc)

    doggenerator.train(epochs=100, batch_size=32)

    doggenerator.save_images3(samples=10,zipFileName=outFile)

    print ('end')