#!/usr/bin/env python3
"""Gradient descent optimizer. Zero dependencies."""
import math

def gradient_descent(grad_fn, x0, lr=0.01, epochs=1000, tol=1e-8):
    x = list(x0) if isinstance(x0, (list,tuple)) else [x0]
    history = [x[:]]
    for _ in range(epochs):
        g = grad_fn(x)
        if not isinstance(g, (list,tuple)): g = [g]
        new_x = [xi - lr*gi for xi, gi in zip(x, g)]
        if sum((a-b)**2 for a,b in zip(x, new_x)) < tol**2: break
        x = new_x; history.append(x[:])
    return x[0] if len(x)==1 else x, history

def adam(grad_fn, x0, lr=0.001, epochs=1000, beta1=0.9, beta2=0.999, eps=1e-8):
    x = list(x0) if isinstance(x0, (list,tuple)) else [x0]
    m = [0]*len(x); v = [0]*len(x)
    for t in range(1, epochs+1):
        g = grad_fn(x)
        if not isinstance(g, (list,tuple)): g = [g]
        for i in range(len(x)):
            m[i] = beta1*m[i]+(1-beta1)*g[i]
            v[i] = beta2*v[i]+(1-beta2)*g[i]**2
            m_hat = m[i]/(1-beta1**t)
            v_hat = v[i]/(1-beta2**t)
            x[i] -= lr*m_hat/(math.sqrt(v_hat)+eps)
    return x[0] if len(x)==1 else x

def numerical_gradient(fn, x, eps=1e-5):
    if isinstance(x, (int,float)):
        return (fn(x+eps)-fn(x-eps))/(2*eps)
    grad = []
    for i in range(len(x)):
        xp = x[:]; xm = x[:]
        xp[i] += eps; xm[i] -= eps
        grad.append((fn(xp)-fn(xm))/(2*eps))
    return grad

if __name__ == "__main__":
    # Minimize x^2
    result, _ = gradient_descent(lambda x: [2*x[0]], [5.0], lr=0.1)
    print(f"Min of x^2: x={result:.6f}")
