import cv2
import numpy as np
import tensorflow as tf
from mlchain.server import TemplateResponse
from PIL import Image
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
        dummy_image = np.random.rand(1, 28, 28, 1)
        self.sess.run(self.output, feed_dict={self.input: dummy_image})
        print("Loaded model!")


    def predict(self, image: np.ndarray):
        """
        Run inference on an input image.

        Args:
            image (numpy.ndarray): An input image.

        Returns:
            A dictionary contain the prediction result
            {
                'output' (int): Class number of input image.
                'confidence' (float): Confidence level of the model.
            }
        """
        if len(image.shape) == 3 and image.shape[-1] != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (1, 28, 28, 1))
        res = self.sess.run(self.output, feed_dict={self.input: image})
        return {
            'output': int(np.argmax(res)),
            'confidence': float(np.max(res))
        }


    def predict_frontend(self, image: Image.Image):
        image = Image.composite(image, Image.new('RGB', image.size, 'white'), image)
        image = image.convert('L')
        image = image.resize((28, 28), Image.ANTIALIAS)
        image = 1 - np.array(image, dtype=np.float32) / 255.0
        return self.predict(image)


    def frontend(self): 
        return TemplateResponse('index.html')
