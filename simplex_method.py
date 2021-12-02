import numpy as np
import simplex_utils

from collections import namedtuple
from copy import deepcopy

_LPProblem = namedtuple('_LPProblem', 'c A_ub b_ub A_eq b_eq bounds x0')

def _unscale(x, C, b_scale):
    try:
        n = len(C)
    except TypeError:
        n = len(x)

    return x[:n]*b_scale*C

def _postsolve(x, postsolve_args, complete=False):
    (c, A_ub, b_ub, A_eq, b_eq, bounds, x0), revstack, C, b_scale = postsolve_args
    x = _unscale(x, C, b_scale)

    n_x = bounds.shape[0]
    if not complete and bounds is not None:  # bounds are never none, probably
        n_unbounded = 0
        for i, bi in enumerate(bounds):
            lbi = bi[0]
            ubi = bi[1]
            if lbi == -np.inf and ubi == np.inf:
                n_unbounded += 1
                x[i] = x[i] - x[n_x + n_unbounded - 1]
            else:
                if lbi == -np.inf:
                    x[i] = ubi - x[i]
                else:
                    x[i] += lbi
    # all the rest of the variables were artificial
    x = x[:n_x]

    for rev in reversed(revstack):
        x = rev(x)

    fun = x.dot(c)
    slack = b_ub - A_ub.dot(x)  # report slack for ORIGINAL UB constraints

    return x, fun, slack

def _pivot_col(T, tol=1e-9, bland=False):
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    if bland:
        # ma.mask is sometimes 0d
        return True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
    return True, np.ma.nonzero(ma == ma.min())[0][0]


def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):
    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
    if ma.count() == 0:
        return False, np.nan
    mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
    q = mb / ma
    min_rows = np.ma.nonzero(q == q.min())[0]
    if bland:
        return True, min_rows[np.argmin(np.take(basis, min_rows))]
    return True, min_rows[0]

def _apply_pivot(T, basis, pivrow, pivcol):
    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] = T[pivrow] / pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]

def _solve_simplex(T, n, basis, postsolve_args,
                   maxiter=1000, tol=1e-9, phase=2, nit0=0,
                   ):
    nit = nit0
    status = 0
    message = ''
    complete = False

    if phase == 1:
        m = T.shape[1]-2
    elif phase == 2:
        m = T.shape[1]-1
    else:
        raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")

    if phase == 2:
        for pivrow in [row for row in range(basis.size)
                       if basis[row] > T.shape[1] - 2]:
            non_zero_row = [col for col in range(T.shape[1] - 1)
                            if abs(T[pivrow, col]) > tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(T, basis, pivrow, pivcol)
                nit += 1

    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)

    while not complete:
        # Find the pivot column
        pivcol_found, pivcol = _pivot_col(T, tol, False)
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            # Find the pivot row
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, False)
            if not pivrow_found:
                status = 3
                complete = True

        if not complete:
            if nit >= maxiter:
                # Iteration limit exceeded
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol)
                nit += 1
    return nit, status

def _linprog_simplex(c, c0, A, b, postsolve_args,
                     maxiter=1000, tol=1e-9):
    
    status = 0
    n, m = A.shape
    # All constraints must have b >= 0.
    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1
    # As all constraints are equality constraints the artificial variables
    # will also be basic variables.
    av = np.arange(n) + m
    basis = av.copy()

    # Format the phase one tableau by adding artificial variables and stacking
    # the constraints, the objective row and pseudo-objective row.
    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    nit1, status = _solve_simplex(T, n, basis,
                                  postsolve_args=postsolve_args,
                                  maxiter=maxiter, tol=tol, phase=1
                                  )
    # if pseudo objective is zero, remove the last row from the tableau and
    # proceed to phase 2
    nit2 = nit1
    if abs(T[-1, -1]) < tol:
        # Remove the pseudo-objective row from the tableau
        T = T[:-1, :]
        # Remove the artificial variable columns from the tableau
        T = np.delete(T, av, 1)
    else:
        # Failure to find a feasible starting point
        status = 2

    if status == 0:
        # Phase 2
        nit2, status = _solve_simplex(T, n, basis,
                                      postsolve_args=postsolve_args,
                                      maxiter=maxiter, tol=tol, phase=2,
                                      nit0=nit1
                                      )

    solution = np.zeros(n + m)
    solution[basis[:n]] = T[:n, -1]
    x = solution[:m]

    return x, status, int(nit2)

def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=None, x0=None):

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0)

    iteration = 0
    complete = False
    undo = []

    lp_o = deepcopy(lp)

    # Solve trivial problem, eliminate variables, tighten bounds, etc.
    c0 = 0  # we might get a constant term in the objective
    C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    if not complete:
        A, b, c, c0, x0 = simplex_utils._get_Abc(lp, c0)
        x, status, iteration = _linprog_simplex(
            c, c0=c0, A=A, b=b, postsolve_args=postsolve_args)

    # Eliminate artificial variables, re-introduce presolved variables, etc.
    x, fun, slack = _postsolve(x, postsolve_args, complete)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': None,
        'status': status,
        'message': '',
        'nit': iteration,
        'success': status == 0}

    return simplex_utils.OptimizeResult(sol)