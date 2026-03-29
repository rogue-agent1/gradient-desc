#!/usr/bin/env python3
"""Gradient descent variants: SGD, Adam, RMSprop, momentum."""
import sys, math, random

class Optimizer:
    def step(self, params, grads): raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr; self.momentum = momentum; self.velocity = None
    def step(self, params, grads):
        if self.velocity is None: self.velocity = [0.0] * len(params)
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]
        return params

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.m = None; self.v = None; self.t = 0
    def step(self, params, grads):
        if self.m is None: self.m = [0.0]*len(params); self.v = [0.0]*len(params)
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * grads[i]**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return params

class RMSprop(Optimizer):
    def __init__(self, lr=0.001, decay=0.9, eps=1e-8):
        self.lr = lr; self.decay = decay; self.eps = eps; self.cache = None
    def step(self, params, grads):
        if self.cache is None: self.cache = [0.0]*len(params)
        for i in range(len(params)):
            self.cache[i] = self.decay * self.cache[i] + (1-self.decay) * grads[i]**2
            params[i] -= self.lr * grads[i] / (math.sqrt(self.cache[i]) + self.eps)
        return params

def numerical_grad(fn, params, eps=1e-5):
    grads = []
    for i in range(len(params)):
        old = params[i]
        params[i] = old + eps; fp = fn(params)
        params[i] = old - eps; fm = fn(params)
        params[i] = old; grads.append((fp - fm) / (2 * eps))
    return grads

def rosenbrock(params):
    return sum(100*(params[i+1]-params[i]**2)**2 + (1-params[i])**2 for i in range(len(params)-1))

def sphere_fn(params): return sum(x**2 for x in params)

def main():
    import argparse
    p = argparse.ArgumentParser(description="Gradient descent optimizer")
    p.add_argument("--fn", choices=["rosenbrock","sphere"], default="rosenbrock")
    p.add_argument("--opt", choices=["sgd","adam","rmsprop"], default="adam")
    p.add_argument("--dim", type=int, default=5); p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.001); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)
    fn = {"rosenbrock": rosenbrock, "sphere": sphere_fn}[args.fn]
    opt = {"sgd": SGD(lr=args.lr, momentum=0.9), "adam": Adam(lr=args.lr), "rmsprop": RMSprop(lr=args.lr)}[args.opt]
    params = [random.uniform(-2, 2) for _ in range(args.dim)]
    print(f"Optimizing {args.fn} ({args.dim}D) with {args.opt.upper()} (lr={args.lr})")
    print(f"Initial: f={fn(params):.6f}")
    for step in range(args.steps):
        grads = numerical_grad(fn, params)
        params = opt.step(params, grads)
        if step % (args.steps // 10) == 0:
            print(f"Step {step:5d}: f={fn(params):.6f}")
    print(f"\nFinal: f={fn(params):.6f}")
    print(f"Solution: [{', '.join(f'{x:.4f}' for x in params)}]")

if __name__ == "__main__": main()
