def grad_f_xi(x, i):
    if i == 0:
        return 100 * (8 * x[0] - 2 * (x[1] + x[2] + x[3] + x[4])) + 2 * x[0] - 8
    else:
        return 200 * (x[i] - x[0])

def grad_g_xi(x, i):
    return 2 * (i + 1) * x[i]

def grad_f(x):
    res = []
    for i in range(0, 5):
        res.append(grad_f_xi(x, i))
    return res

def grad_g(x):
    res = []
    for i in range(0, 5):
        res.append(grad_g_xi(x, i))
    return res
