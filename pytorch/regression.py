import numpy as np

#y=wx+b,计算误差函数
def compute_error_for_line_given_points(w,b,points):
    totalError = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (w * x + b - y) ** 2
        return totalError/float(len(points))

#计算每一步的梯度
def step_gradient(w_current, b_current, points, learningRate):
    w_gradient = 0.0
    b_gradient = 0.0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        w_gradient += -(2/N) * (y - ((w_current * x) + b_current)) * x
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
    w_new = w_current - (learningRate * w_gradient)
    b_new = b_current - (learningRate * b_gradient)
    return [w_new,b_new]

#循环迭代计算梯度
def gradient_descent_runner(w_start, b_start, points, learningRate, epho):
    w = w_start
    b = b_start
    points = np.array(points)
    for i in range(epho):
        [w, b] = step_gradient(w, b, points, learningRate)
    return [w,b]


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    #points = [[1,5],[2,10],[3,15],[4,21],[5,24],[6,30]]
    points = np.array(points[1:,:])
    learningRate = 1e-4
    w_initial = 100*np.random.random()-50.0
    b_initial = 100*np.random.random()-50.0
    epho = 10000

    print("Starting gradient descent at w = {0}, b = {1}, error = {2}"
          .format(w_initial, b_initial,
                  compute_error_for_line_given_points(w_initial,b_initial,points)))

    print("Running...")
    [w,b] = gradient_descent_runner(w_initial, b_initial, points, learningRate, epho)

    print("After {0} iterations w = {1}, b = {2}, error = {3} "
          .format(epho, w, b, compute_error_for_line_given_points(w,b,points)))


if __name__ == '__main__':
    run()

