from keras.models import Model
from keras.layers import Dense

class Actor:
    def __init__(self, shared_model, output_dim):
        y = Dense(output_dim, activation='relu')(shared_model.output)
        self.model = Model(shared_model.input, y)

    def fit(self):
        pass

    def predict(self):
        pass