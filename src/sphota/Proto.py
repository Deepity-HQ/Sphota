import torch

class Proto:
    def __init__(self, front, seed=0):
        self.front, self.keys, self.y = front, torch.zeros(0, front.dim), []

    def _s(self, x):
        return (self.keys @ self.front.features(x[None])[0]).tolist()

    def predict(self, x):
        return self.y[int(np.argmax(self._s(x)))]

    def class_scores(self, x):
        return dict(zip(self.y, self._s(x)))

    def learn(self, x, c, pred):
        if c not in self.y:
            self.keys = torch.cat([self.keys, self.front.features(x[None])]); self.y.append(c)

    def items(self):
        return len(self.y)

