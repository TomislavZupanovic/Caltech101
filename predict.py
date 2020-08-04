from source.model import Model
from source.preprocess import process_data
from source.utils import random_image

train_data, valid_data, test_data, train_dataset = process_data('source/data/Dataset')
model = Model()  # Instantiate model
model.load_weights('model_Caltech101')
model.predict(random_image())




