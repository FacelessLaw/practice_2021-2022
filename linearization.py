import numpy as np
import pandas as pd
from scipy.optimize import linprog
from matplotlib import pyplot as plt

import simplex_method
import utils

def linear_minimize(f, x0, g_list, eps=1e-4, dx=1e-5,
                    maxstep=1e1, step_decrement=0.5, linear_method='simplex',
                    print_linprog=False):
    '''
    solve problem:
        f(x) -> min
        g(x) <= 0 for g in g_list
    by linearization method
        
    arguments:
        f - callable, function to minimize
        x0 - initial guess
        g_list - list of callables, limitation functions
        eps - float, criterion to stop if ||x_k - x_k-1|| < eps
        dx - float or array of same shape as x0, precision to calculate gradient
        maxstep - float or array of same shape as x0, linprog boundary parameter: |x^i_k - x^i_k-1| < maxstep
        step_decrement - float or array of same shape as maxstep, rate to decrease maxstep
        linear_method - string, 'simplex' or 'interior-point', 
            see scipy.optimize.linprog documentation for more info
        print_linprog - boolean, if True then print linear programming result info
            
    returns:
        x_min - problem solution
    '''
    x = np.asarray(x0, dtype=np.float64)
    x0 = x + 10
    niter = 0
    while np.linalg.norm(x - x0) > eps:
        '''
        f(x) ~ f(x_k) + <grad f(x_k), x - x_k> --> min  <==>  
               <grad f(x_k), x> --> min
        g(x) ~ g(x_k) + <grad g(x_k), x - x_k> <= 0  <==>  
               <grad g(x_k), x> <= <grad g(x_k), x_k> - g(x_k)
        '''        
        c = np.array(utils.grad_f(x))
        A_ub = np.array([utils.grad_g(x) for g in g_list])
        b_ub = (A_ub * x).sum(axis=1) - np.array([g(x) for g in g_list])
        bounds = np.c_[x - maxstep, x + maxstep]
        
        x0 = x
        # need for debug purposes
        # res = linprog(c, A_ub, b_ub, method=linear_method, bounds=bounds) 
        res = simplex_method.linprog(c, A_ub, b_ub, bounds=bounds)
        if print_linprog:
            print(res)
        niter += res.nit
        x = res.x
        
        maxstep *= step_decrement
        # print(x0)
        # print(x)
        # print(np.linalg.norm(x - x0))
        # print(eps)
        # print("\n")
        
    return x, niter

if __name__ == '__main__':
    f = lambda x: 100*((x[1:] - x[0])**2).sum() + (x[0] - 4)**2
    g1 = lambda x: sum([(i+1)*v**2 for i, v in enumerate(x)]) - 79
    
    x0 = [3, 2, 4, 2.5, 3.1]

    # niter_all = []
    # eps_all = np.linspace(0.01, 0.1, num=10)
    # for eps in eps_all:
    #     x_min, niter = linear_minimize(f, x0, [g1], maxstep=10, step_decrement=0.99, eps=eps)
    #     niter_all.append(niter)
        
    # plt.style.use('ggplot')
    # plt.plot(eps_all, niter_all)
    # plt.savefig('plot.png')

    curr_eps = 1e-4
    step = 1e-1
    t1 = {
        "x0_1": [],
        "x0_2": [],
        "x0_3": [],
        "x0_4": [],
        "x0_5": [],
        "iterations": [],
        "xmin_1": [],
        "xmin_2": [],
        "xmin_3": [],
        "xmin_4": [],
        "xmin_5": [],
        "f(xmin)": [],
        "||x0-xmin||": []
    }
    for i in range(0, 101):
        x1 = -10 + step * i
        old_x0_1 = x0[0]
        x0[0] = x1
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t1["x0_1"].append(x0[0])
        t1["x0_2"].append(x0[1])
        t1["x0_3"].append(x0[2])
        t1["x0_4"].append(x0[3])
        t1["x0_5"].append(x0[4])
        t1["iterations"].append(niter)
        t1["xmin_1"].append(x_min[0])
        t1["xmin_2"].append(x_min[1])
        t1["xmin_3"].append(x_min[2])
        t1["xmin_4"].append(x_min[3])
        t1["xmin_5"].append(x_min[4])
        t1["f(xmin)"].append(f(x_min))
        t1["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[0] = old_x0_1
    
    for i in range(0, 101):
        x2 = -10 + step * i
        old_x0_2 = x0[1]
        x0[1] = x2
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t1["x0_1"].append(x0[0])
        t1["x0_2"].append(x0[1])
        t1["x0_3"].append(x0[2])
        t1["x0_4"].append(x0[3])
        t1["x0_5"].append(x0[4])
        t1["iterations"].append(niter)
        t1["xmin_1"].append(x_min[0])
        t1["xmin_2"].append(x_min[1])
        t1["xmin_3"].append(x_min[2])
        t1["xmin_4"].append(x_min[3])
        t1["xmin_5"].append(x_min[4])
        t1["f(xmin)"].append(f(x_min))
        t1["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[1] = old_x0_2
    
    for i in range(0, 101):
        x3 = -10 + step * i
        old_x0_3 = x0[2]
        x0[2] = x3
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t1["x0_1"].append(x0[0])
        t1["x0_2"].append(x0[1])
        t1["x0_3"].append(x0[2])
        t1["x0_4"].append(x0[3])
        t1["x0_5"].append(x0[4])
        t1["iterations"].append(niter)
        t1["xmin_1"].append(x_min[0])
        t1["xmin_2"].append(x_min[1])
        t1["xmin_3"].append(x_min[2])
        t1["xmin_4"].append(x_min[3])
        t1["xmin_5"].append(x_min[4])
        t1["f(xmin)"].append(f(x_min))
        t1["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[2] = old_x0_3
    
    for i in range(0, 101):
        x4 = -10 + step * i
        old_x0_4 = x0[3]
        x0[3] = x4
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t1["x0_1"].append(x0[0])
        t1["x0_2"].append(x0[1])
        t1["x0_3"].append(x0[2])
        t1["x0_4"].append(x0[3])
        t1["x0_5"].append(x0[4])
        t1["iterations"].append(niter)
        t1["xmin_1"].append(x_min[0])
        t1["xmin_2"].append(x_min[1])
        t1["xmin_3"].append(x_min[2])
        t1["xmin_4"].append(x_min[3])
        t1["xmin_5"].append(x_min[4])
        t1["f(xmin)"].append(f(x_min))
        t1["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[3] = old_x0_4
    
    for i in range(0, 101):
        x5 = -10 + step * i
        old_x0_5 = x0[4]
        x0[4] = x5
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t1["x0_1"].append(x0[0])
        t1["x0_2"].append(x0[1])
        t1["x0_3"].append(x0[2])
        t1["x0_4"].append(x0[3])
        t1["x0_5"].append(x0[4])
        t1["iterations"].append(niter)
        t1["xmin_1"].append(x_min[0])
        t1["xmin_2"].append(x_min[1])
        t1["xmin_3"].append(x_min[2])
        t1["xmin_4"].append(x_min[3])
        t1["xmin_5"].append(x_min[4])
        t1["f(xmin)"].append(f(x_min))
        t1["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[4] = old_x0_5
    table1 = pd.DataFrame(t1)
    table1.to_excel('./table1.xlsx')

    step = 1
    t2 = {
        "x0_1": [],
        "x0_2": [],
        "x0_3": [],
        "x0_4": [],
        "x0_5": [],
        "iterations": [],
        "xmin_1": [],
        "xmin_2": [],
        "xmin_3": [],
        "xmin_4": [],
        "xmin_5": [],
        "f(xmin)": [],
        "||x0-xmin||": [],
    }
    for i in range(0, 101):
        x1 = -50 + step * i
        old_x0_1 = x0[0]
        x0[0] = x1
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t2["x0_1"].append(x0[0])
        t2["x0_2"].append(x0[1])
        t2["x0_3"].append(x0[2])
        t2["x0_4"].append(x0[3])
        t2["x0_5"].append(x0[4])
        t2["iterations"].append(niter)
        t2["xmin_1"].append(x_min[0])
        t2["xmin_2"].append(x_min[1])
        t2["xmin_3"].append(x_min[2])
        t2["xmin_4"].append(x_min[3])
        t2["xmin_5"].append(x_min[4])
        t2["f(xmin)"].append(f(x_min))
        t2["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[0] = old_x0_1
    
    for i in range(0, 101):
        x2 = -50 + step * i
        old_x0_2 = x0[1]
        x0[1] = x2
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t2["x0_1"].append(x0[0])
        t2["x0_2"].append(x0[1])
        t2["x0_3"].append(x0[2])
        t2["x0_4"].append(x0[3])
        t2["x0_5"].append(x0[4])
        t2["iterations"].append(niter)
        t2["xmin_1"].append(x_min[0])
        t2["xmin_2"].append(x_min[1])
        t2["xmin_3"].append(x_min[2])
        t2["xmin_4"].append(x_min[3])
        t2["xmin_5"].append(x_min[4])
        t2["f(xmin)"].append(f(x_min))
        t2["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[1] = old_x0_2
    
    for i in range(0, 101):
        x3 = -50 + step * i
        old_x0_3 = x0[2]
        x0[2] = x3
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t2["x0_1"].append(x0[0])
        t2["x0_2"].append(x0[1])
        t2["x0_3"].append(x0[2])
        t2["x0_4"].append(x0[3])
        t2["x0_5"].append(x0[4])
        t2["iterations"].append(niter)
        t2["xmin_1"].append(x_min[0])
        t2["xmin_2"].append(x_min[1])
        t2["xmin_3"].append(x_min[2])
        t2["xmin_4"].append(x_min[3])
        t2["xmin_5"].append(x_min[4])
        t2["f(xmin)"].append(f(x_min))
        t2["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[2] = old_x0_3
    
    for i in range(0, 101):
        x4 = -50 + step * i
        old_x0_4 = x0[3]
        x0[3] = x4
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t2["x0_1"].append(x0[0])
        t2["x0_2"].append(x0[1])
        t2["x0_3"].append(x0[2])
        t2["x0_4"].append(x0[3])
        t2["x0_5"].append(x0[4])
        t2["iterations"].append(niter)
        t2["xmin_1"].append(x_min[0])
        t2["xmin_2"].append(x_min[1])
        t2["xmin_3"].append(x_min[2])
        t2["xmin_4"].append(x_min[3])
        t2["xmin_5"].append(x_min[4])
        t2["f(xmin)"].append(f(x_min))
        t2["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[3] = old_x0_4
    for i in range(0, 101):
        x5 = -50 + step * i
        old_x0_5 = x0[4]
        x0[4] = x5
        x_min, niter = linear_minimize(
            f, x0, [g1],
            maxstep=10,
            step_decrement=0.99,
            eps=curr_eps
        )
        t2["x0_1"].append(x0[0])
        t2["x0_2"].append(x0[1])
        t2["x0_3"].append(x0[2])
        t2["x0_4"].append(x0[3])
        t2["x0_5"].append(x0[4])
        t2["iterations"].append(niter)
        t2["xmin_1"].append(x_min[0])
        t2["xmin_2"].append(x_min[1])
        t2["xmin_3"].append(x_min[2])
        t2["xmin_4"].append(x_min[3])
        t2["xmin_5"].append(x_min[4])
        t2["f(xmin)"].append(f(x_min))
        t2["||x0-xmin||"].append(np.linalg.norm(x_min - x0))
        x0[4] = old_x0_5
    table2 = pd.DataFrame(t2)
    table2.to_excel('./table2.xlsx')
