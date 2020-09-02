import numpy as np

class MyPerceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
       self.lr = learning_rate
       self.epochs = n_iterations
       self.b = None   #bias
       self.w = None  #weights

    def fit(self, x, y):
        self.w = np.ones(x.shape[1])
        self.b = 0

        for i in range(self.epochs):
            y_hat = self.activation(np.dot(self.w, x.T) + self.b)

            dw = - np.dot((y - y_hat), x) / x.shape[0]
            db = -np.mean(y - y_hat)
            self.w = self.w - (self.lr * dw)
            self.b = self.b - (self.lr * db)

        print(self.w)
        print(self.b)

    def activation(self, z):
        return 1/(1 + np.exp(-z))

    def predict(self, x):
        y_pred = []
        for i in range(x.shape[0]):
            conversion = (self.activation(np.dot(self.w, x[i].T) + self.b) > 0.5).astype(int)
            y_pred.append(conversion)
        return np.array(y_pred)

