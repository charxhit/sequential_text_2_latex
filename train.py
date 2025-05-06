import config
import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot as plt


class TrainSym:

    def __init__(self, name, epochs=10, batch_size=128, save_path=Path.cwd()):
        self.imgs = None
        self.labels = None
        self.pickle_loc = config.pickle_loc
        self._labels_to_code = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.dir_to_label = config.dic
        self.save_path = save_path
        self.name = name

        self._preprocess_data()

    def _preprocess_data(self):
        # We need to convert our labels to numeric values, so we create our arrays to do that here.
        self._labels_to_code = {label: num for num, label in enumerate(self.dir_to_label)}

    def load(self):
        while True:
            try:
                with open(self.pickle_loc / "imgs.pickle", 'rb') as f:
                    imgs = pickle.load(f)
                with open(self.pickle_loc / "labels.pickle", 'rb') as f:
                    labels = pickle.load(f)
            except FileNotFoundError:
                print("No saved pickle for imgs and labels found, creating now")
                self.save()
            else:
               return imgs, labels

    def save(self):
        imgs, labels = self._load_data(config.dataset)
        self.pickle_loc.mkdir(exist_ok=True, parents=True)
        with open(self.pickle_loc / "imgs.pickle", 'wb') as f:
            pickle.dump(imgs, f)
        with open(self.pickle_loc / "labels.pickle", 'wb') as f:
            pickle.dump(labels, f)

    def _load_data(self, dataDir):
        imgs = []
        labels = []
        for key, value in self.dir_to_label.items():
            path: Path = dataDir / key
            for img in path.iterdir():
                if not img.is_file():
                    continue
                try:
                    img = cv2.imread(img, cv2.COLOR_BGR2GRAY)
                    imgs.append(img)
                    labels.append(value)
                except Exception as e:
                    print(e)

        return imgs, labels

    def train(self):
        print("Loading data...")
        self.imgs, self.labels = self.load()
        print("Data loaded")

        # Here x_train, y_train is the input/output for training, with input containing imgs and output having labels
        # (in latex)
        x_train, x_test, y_train, y_test = train_test_split(self.imgs, self.labels, test_size=0.33,
                                                            stratify=self.labels,
                                                            random_state=42)
        # Normalize imgs (divide by 255 basically)
        x_train = self.normalize_pixels(x_train)
        x_test = self.normalize_pixels(x_test)

        # Convert output labels to numeric codes for consistency with keras model
        y_train = [self._labels_to_code[label] for label in y_train]
        y_test = [self._labels_to_code[label] for label in y_test]

        # COnvert our outputs to ndarrays
        y_train, y_test = np.array(y_train), np.array(y_test)

        print("Creating model...")
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(45, 45, 1)))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(2025, activation='relu'))
        model.add(Dense(79, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("Beginning training...")
        x = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2, validation_split=0.1)

        model.save(self.save_path / self.name)


    @staticmethod
    def normalize_pixels(imgs):
        return tf.keras.utils.normalize(imgs, axis=1)



t = TrainSym('test')
t.train()