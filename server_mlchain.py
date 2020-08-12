from model import MNISTClassifier
from mlchain.base import ServeModel

minist_model = MNISTClassifier(model_path='weights/pretrained_mnist.pb')
serve_model = ServeModel(minist_model)