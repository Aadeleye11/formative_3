## Part 1: EM Algorithm (Gaussian Mixture)

Implemented **Expectation-Maximization from scratch** (no scikit-learn) on Galton father + child heights, labels hidden. Functions map to the math: `e_step`, `m_step`, `log_likelihood`.

**Tracking table (init + first 2 iterations):**

| Iter | μ₁ | μ₂ | σ₁² | σ₂² | π₁ | π₂ | Log-Lik |
|---|---|---|---|---|---|---|---|
| 0 | 66.00 | 70.00 | 10.96 | 10.96 | 0.500 | 0.500 | −4930.52 |
| 1 | 66.40 | 69.52 | 9.70 | 7.39 | 0.497 | 0.503 | −4873.52 |
| 2 | 66.37 | 69.53 | 10.23 | 6.76 | 0.494 | 0.506 | −4869.37 |

**Draw a line at the mean? No.** It ignores the overlap, unequal variances, and unequal priors; pile-means are biased by stolen tails. The twist: fathers/children are only ~2.5″ apart, so the data is near-unimodal and EM (maximising *likelihood*, not accuracy) finds a broad + narrow fit that doesn't recover the true split. Neither method separates well. **Mixtures separate only what is separable.** On a controlled separable case, EM hit 99% vs the naive 91%.

`classify(height)` outputs live posteriors: P(child) vs P(taller class).
# Part 2: Bayesian Probability – IMDb Movie Reviews

## Overview

In this part of the project, we applied **Bayes' Theorem** to predict whether a movie review is **positive** based on the presence of specific keywords. We used the **IMDb Movie Reviews Dataset** and implemented the solution using **basic Python only**, without any machine learning libraries.

## Objective

Our goal was to calculate the probability that a review is positive given that it contains a particular keyword:

[
P(\text{Positive} \mid \text{Keyword})
]

We selected the following positive keywords:

* **excellent**
* **amazing**
* **wonderful**

## Bayes' Theorem

We used the formula:

[
P(\text{Positive}|\text{Keyword})=
\frac{P(\text{Keyword}|\text{Positive}) \times P(\text{Positive})}
{P(\text{Keyword})}
]

Where:

* **Prior –** (P(\text{Positive})): The probability that any review is positive.
* **Likelihood –** (P(\text{Keyword}|\text{Positive})): The probability that a positive review contains the selected keyword.
* **Marginal –** (P(\text{Keyword})): The probability that the keyword appears in all reviews.
* **Posterior –** (P(\text{Positive}|\text{Keyword})): The final probability that a review is positive if it contains the keyword.

## Results
Prior P(Positive) = 0.5000

----------|----------------------|--------------|------------------------
Keyword   |  P(keyword|Positive) |  P(keyword)  |      P(Positive|keyword)
----------|----------------------|--------------|------------------------
excellent |  0.1147              |  0.0710      |      0.8074
amazing   | 0.0672               | 0.0432       |     0.7780
wonderful |  0.0903              |  0.0556      |      0.8122

| Keyword   | Posterior Probability |
| --------- | --------------------: |
| excellent |                80.74% |
| amazing   |                77.80% |
| wonderful |                81.22% |

These results show that reviews containing these words are highly likely to express positive sentiment.

## Implementation

Our Python program:

1. Loaded the IMDb dataset.
2. Counted positive reviews to calculate the **Prior**.
3. Counted how often each keyword appeared in positive reviews to calculate the **Likelihood**.
4. Counted how often each keyword appeared in all reviews to calculate the **Marginal**.
5. Applied **Bayes' Theorem** to compute the **Posterior Probability** for each keyword.

## Conclusion

Bayesian Probability allows us to make predictions based on evidence. In this project, we showed that a single keyword can significantly influence the probability that a movie review is positive. This demonstrates how Bayes' Theorem forms the foundation of many text classification and spam filtering systems while using only simple probability calculations.



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
