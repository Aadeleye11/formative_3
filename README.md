## Part 3: Gradient Descent Manual Calculation

### Objective
Manually compute three updates of the gradient descent algorithm for the parameters `m` and `b` in a linear regression model, using matrix multiplication (not scalar operations).

### Model
y = m1x1 + m2x2 + b

### Given
- X = [(1, 3), (4, 10)]
- targets = [5, 6]
- Initial m = [-1, 2], Initial b = 1
- Learning rate (α) = 0.01
- n = 2 (number of samples)

### Gradient of the MSE Cost Function
J = (1/n) * ||y_hat - target||²
e = y_hat - target = X.m + b - target
∂J/∂m = (2/n) * Xᵀ.e
∂J/∂b = (2/n) * Σe
Since n = 2, 2/n = 1, so:
∇m = Xᵀ.e
∇b = Σe
### Iterations Summary

| Iteration | m | b | Error (e) |
|-----------|---|---|-----------|
| Start | [-1, 2] | 1 | — |
| 1 | [-1.45, 0.87] | 0.88 | [1, 11] |
| 2 | [-1.3316, 1.1808] | 0.9318 | [-2.96, -2.22] |
| 3 | [-1.3696, 1.0952] | 0.9362 | [-1.8574, 1.4134] |

Each iteration followed the same 4 steps: **Predict → Compute Error → Compute Gradient → Update m and b.** Full handwritten steps for each iteration are included in the manual calculation document.

### Observed Trend
Across the three iterations, the error steadily decreases in magnitude (from [1, 11] down to [-1.86, 1.41]), and `m` and `b` are converging toward values that reduce the MSE. This confirms gradient descent is moving the parameters in the correct direction to minimize the cost function.

---

## Part 4: Gradient Descent in Code

### Objective
Convert the Part 3 manual calculations into Python code, using SciPy to verify the gradient and Matplotlib to visualize the results.

### Approach
- **Analytical gradient**: implemented directly from the formulas derived in Part 3 (`∇m = Xᵀ.e`, `∇b = Σe`).
- **SciPy verification**: `scipy.optimize.approx_fprime` is used to numerically compute the derivative of the MSE cost function, confirming the analytical gradient is correct at each step.
- **Update loop**: each iteration explicitly shows Predict → Error → MSE → Gradient (analytical & SciPy) → Update, matching the manual calculation step-by-step (no hidden/abstracted "train" function).

### Output
Running the script reproduces the exact same values as the manual calculations:

| Iteration | m | b |
|-----------|---|---|
| 1 | [-1.45, 0.87] | 0.88 |
| 2 | [-1.3316, 1.1808] | 0.9318 |
| 3 | [-1.3696, 1.0952] | 0.9362 |

### Visualizations
- **`params_over_iterations.png`** — shows how m1, m2, and b change across the 3 iterations.
- **`error_over_iterations.png`** — shows the MSE dropping from 61.0 down to ~2.7 across the iterations, visually confirming convergence.

---

## Conclusion
The manual calculations in Part 3 and the Python implementation in Part 4 produced identical results, confirming the correctness of our gradient descent derivation. The steadily decreasing error across iterations demonstrates that the model is successfully learning better parameters with each update.
