from dataclasses import dataclass


@dataclass
class Optimizer:
    def intialize(params):
        ...

    def step(opt_params, params, grads):
        ...


@dataclass
class SGD(Optimizer):
    lr: float

    def intialize(params):
        return dict()

    def step(self, params, grads):
        for p, g in zip(params, grads):
            for k in p.keys():
                p[k] -= lr * g[k]
        return params, {}
