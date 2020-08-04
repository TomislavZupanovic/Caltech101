from source.model import Model
from source.visualize import plot_losses
from source.preprocess import process_data

train_data, valid_data, test_data, train_dataset = process_data('source/data/Dataset')
model = Model()  # Instantiate model
model.build('CES', 0.001, train_dataset)
losses = model.fit(50, train_data, valid_data)
model.evaluate(1, test_data)
plot_losses(losses)


