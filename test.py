from gradient_desc import gradient_descent, adam, numerical_gradient
r, hist = gradient_descent(lambda x: [2*x[0]], [5.0], lr=0.1, epochs=100)
assert abs(r) < 0.01
r2, _ = gradient_descent(lambda x: [2*(x[0]-3)], [0.0], lr=0.1)
assert abs(r2-3) < 0.01
r3 = adam(lambda x: [2*x[0]], [5.0], lr=0.1, epochs=500)
assert abs(r3) < 0.1
g = numerical_gradient(lambda x: x**2, 3.0)
assert abs(g-6) < 0.01
g2 = numerical_gradient(lambda x: x[0]**2+x[1]**2, [3.0, 4.0])
assert abs(g2[0]-6)<0.01 and abs(g2[1]-8)<0.01
print("gradient_desc tests passed")
