from dataclasses import dataclass

import jax.numpy as jnp


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


@dataclass
class Adam(Optimizer):
    lr: float
    beta_one: float = 0.9
    beta_two: float = 0.999
    epsilon: float = 1e-8

    @classmethod
    def _create_params(cls, p):
        return {"m": jnp.zeros_like(p), "v": jnp.zeros_like(p), "t": 1}

    @classmethod
    def intialize(cls, params):
        opt_params = []
        for p_dict in params:
            empty = {k: cls._create_params(p) for (k, p) in p_dict.items()}
            opt_params.append(empty)
        return opt_params

    def step(self, params, grads, opt_params):
        for o, p, g in zip(opt_params, params, grads):
            for k in p.keys():
                o[k]["m"] = self.beta_one * o[k]["m"] + (1 - self.beta_one) * g[k]
                o[k]["v"] = self.beta_two * o[k]["v"] + (1 - self.beta_two) * (
                    g[k] ** 2
                )

                t = o[k]["t"]

                m_t = o[k]["m"] / (1 - self.beta_one**t)
                v_t = o[k]["v"] / (1 - self.beta_two**t)

                o[k]["t"] += 1

                p[k] -= self.lr * (m_t / (jnp.sqrt(v_t) + self.epsilon))

        return params, opt_params
