import tensorflow as tf

class Actor:
    def __init__(self, shared_model, output_dim):
        y = tf.keras.layers.Dense(output_dim, activation='relu')(shared_model.output)
        self.model = tf.keras.Model(shared_model.input, y)

    def fit(self, X, y, lock=None):
        if lock: lock.acquire()
        self.model.compile(lr=0.01, optimizer='rmsprop', loss='mean_squared_error')
        self.model.fit(X, y, epochs=100, batch_size=100)
        if lock: lock.release()

    def predict(self, X, lock=None):
        if lock: lock.acquire()
        self.model.predict(X)
        if lock: lock.release()