import h5py
import numpy as np
from PIL import Image
import cv2
import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
from common.utils import unfold_label, shuffle_data
from collections import Counter


class BatchImageGenerator:
    def __init__(self, flags, stage, file_path, b_unfold_label):
        if stage not in ["train", "val", "test"]:
            assert ValueError("invalid stage!")
        self.flags = flags
        self.configuration(flags, stage, file_path)
        self.load_data(b_unfold_label)

    def configuration(self, flags, stage, file_path):
        self.batch_size = flags.batch_size
        self.current_index = -1
        self.file_path = file_path
        self.stage = stage

    def normalize(self, inputs):
        # the mean and std used for the normalization of
        # the inputs for the pytorch pretrained model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # norm to [0, 1]
        inputs = inputs / 255.0

        inputs_norm = []
        for item in inputs:
            item = np.transpose(item, (2, 0, 1))
            item_norm = []
            for c, m, s in zip(item, mean, std):
                c = np.subtract(c, m)
                c = np.divide(c, s)
                item_norm.append(c)

            item_norm = np.stack(item_norm)
            inputs_norm.append(item_norm)

        inputs_norm = np.stack(inputs_norm)

        return inputs_norm

    def create_imbalance(self, data, labels, ratio):
        statistics = Counter(labels)
        print(list(statistics.values()))
        major_classes = [0, 1, 2]
        minor_classes = [3, 4, 5, 6]
        n_major = np.array(list(statistics.values())).min()

        n_minor = int(n_major / ratio)
        print("Ratio: {:.4f}, n_major/n_minor={}/{}".format(ratio, n_major, n_minor))

        class_dict = {}
        for i in range(len(labels)):
            image, label = data[i], labels[i]
            if class_dict.get(label, None) is None:
                class_dict[label] = [image]
            else:
                class_dict[label].append(image)

        if n_major > np.array(list(statistics.values())).min():
            raise Exception("Not enough samples")
        new_images = []
        new_labels = []
        for c in major_classes:
            new_images.extend(class_dict[c][0:n_major])
            new_labels.extend([c] * n_major)
        for c in minor_classes:
            new_images.extend(class_dict[c][0:n_minor])
            new_labels.extend([c] * n_minor)
        new_images, new_labels = np.stack(new_images), np.array(new_labels)
        return new_images, new_labels

    def load_data(self, b_unfold_label):
        file_path = self.file_path
        f = h5py.File(file_path, "r")
        self.images = np.array(f["images"])
        self.labels = np.array(f["labels"])
        f.close()

        def resize(x):
            x = x[
                :, :, [2, 1, 0]
            ]  # we use the pre-read hdf5 data file from the download page and need to change BRG to RGB
            return np.array(Image.fromarray(obj=x, mode="RGB").resize(size=(224, 224)))
            # return cv2.resize(src=x, dsize=(224, 224, 3))

        # resize the image to 224 for the pretrained model
        self.images = np.array(list(map(resize, self.images)))

        # norm the image value
        self.images = self.normalize(self.images)

        assert np.max(self.images) < 5.0 and np.min(self.images) > -5.0

        # shift the labels to start from 0
        self.labels -= np.min(self.labels)
        if self.flags.imbalanced_class == True and "train" in file_path:
            new_images, new_labels = self.create_imbalance(
                self.images, self.labels, self.flags.imbalance_ratio
            )
            self.images = new_images
            self.labels = new_labels

        if b_unfold_label:
            self.labels = unfold_label(
                labels=self.labels, classes=len(np.unique(self.labels))
            )
        assert len(self.images) == len(self.labels)

        self.file_num_train = len(self.labels)
        print("data num loaded:", self.file_num_train)

        if self.stage == "train":
            self.images, self.labels = shuffle_data(
                samples=self.images, labels=self.labels
            )

    def get_images_labels_batch(self):
        images = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.images, self.labels = shuffle_data(
                    samples=self.images, labels=self.labels
                )

            images.append(self.images[self.current_index])
            labels.append(self.labels[self.current_index])

        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels

    def shuffle(self):
        self.file_num_train = len(self.labels)
        self.current_index = 0
        self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
