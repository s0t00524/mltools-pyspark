import cvxpy as cp
import numpy as np
x = cp.Variable(4)

objective = cp.Minimize(4 * x[1] + 3 * x[2] + x[3])
constraints = [
    x[0] == 0,
    x >= 0,
    x[1] - x[2] + x[3] >= 1,
    x[1] + 2 * x[2] - 3 * x[3] >= 2,
]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
print(objective.value)