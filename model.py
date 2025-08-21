w,b, m, lr = 0.1, 0.1, 20, 0.002

x = [14.98, 38.03, 29.28, 23.95, 6.24, 6.24, 2.32, 34.65, 24.04, 28.32, 0.82, 38.80, 33.30, 8.49, 7.27, 7.34, 12.17, 20.99, 17.28, 11.65]

y = [29.90, 82.63, 59.02, 45.83, 24.81, 16.35, 9.98, 67.17, 50.37, 62.20, 0.89, 84.47, 68.59, 20.53, 16.54, 28.93, 29.27, 41.69, 43.67, 22.19]

y_hat = []

cost = None
w_step = None
b_step = None


# Functions

def new_predictions():
    global y_hat
    y_hat = [(w * xi) + b for xi in x]

def new_cost():
    global cost
    cost = (1 / (2*m) * (sum((y_hat - y_actual) ** 2 for y_actual, y_hat in zip(y, y_hat))))

def deriv_w():
    global w_step
    w_step = 1 / m * sum((y_hat - y_actual) * xi for y_hat, y_actual, xi in zip(y_hat, y, x))

def deriv_b():
    global b_step
    b_step = 1 / m * sum((y_hat - y_actual) for y_hat, y_actual, in zip(y_hat, y) )

def w_update():
    global w
    w = w - (lr * w_step)

def b_update():
    global b
    b = b - (lr * b_step)


# Gradient Descent

a = 0

while (a < 1000000):
    new_predictions(); new_cost()
    print(f"New predictions for y_hat: {y_hat}")
    print(f"New cost function value is {cost}")
    deriv_w(); deriv_b()
    print(f"The derivate of the cost function with respect to w is {w_step}, and with respect to b, it is {b_step}")
    b_update(); w_update()
    print(f"The new value of b is {b}, and the new value of w is {w}")
    a += 1
