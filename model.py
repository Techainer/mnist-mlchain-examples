import cv2
import numpy as np
import tensorflow as tf

from utils import load_graph


class MNISTClassifier(object):
    """
    MINIST Classifier that can classifiy handwriten digit
    """
    def __init__(self, model_path):
        """
        Init MNIST Classifier.

        Args:
            model_path (str): Path to a pretrained (.pb) model.
        """
        self.graph = load_graph(model_path)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.input = self.graph.get_tensor_by_name('net/input/X:0')
        self.output = self.graph.get_tensor_by_name('net/output/predict:0')

    def predict(self, image: np.ndarray):
        """
        Run inference on an input image.

        Args:
            image (numpy.ndarray): An input image.

        Returns:
            Class number of input image.
        """
        if len(image.shape) == 3 and image.shape[2] != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.reshape(image, (1, 28, 28, 1))
        res = self.sess.run(self.output, feed_dict={self.input: image})
        return np.argmax(res)
