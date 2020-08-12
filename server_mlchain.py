from mlchain.base import ServeModel
from model import MNISTClassifier

minist_model = MNISTClassifier(model_path='weights/pretrained_mnist.pb')

serve_model = ServeModel(minist_model)