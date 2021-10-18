import numpy as np
from scipy.optimize import approx_fprime as grad
from scipy.optimize import linprog
from matplotlib import pyplot as plt

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
        c = grad(x, f, epsilon=dx)
        A_ub = np.array([grad(x, g, epsilon=dx) for g in g_list])
        b_ub = (A_ub * x).sum(axis=1) - np.array([g(x) for g in g_list])
        bounds = np.c_[x - maxstep, x + maxstep]
        
        x0 = x
        res = linprog(c, A_ub, b_ub, method=linear_method, bounds=bounds)
        if print_linprog:
            print(res)
        niter += res.nit
        x = res.x
        
        maxstep *= step_decrement
        print(x0)
        print(x)
        print(np.linalg.norm(x - x0))
        print(eps)
        print("\n")
        
    return x, niter

if __name__ == '__main__':
    f = lambda x: 100*((x[1:] - x[0])**2).sum() + (x[0] - 3)**2
    g1 = lambda x: (x**2).sum() - x[2]
    g2 = lambda x: x[2] - 1
    x0 = [3] * 5
    
    niter_all = []
    eps_all = np.linspace(0.01, 0.1, num=10)
    for eps in eps_all:
        x_min, niter = linear_minimize(f, x0, [g1, g2], maxstep=10, step_decrement=0.99, eps=eps)
        niter_all.append(niter)
        
    plt.style.use('ggplot')
    plt.plot(eps_all, niter_all)
    plt.savefig('plot.png')