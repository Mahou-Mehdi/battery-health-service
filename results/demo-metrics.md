# Evaluation results

Data source: **synthetic_demo**.

SOH = measured capacity / nominal capacity. MAE and RMSE below are SOH percentage points.
For example, 0.3 percentage points means an absolute SOH error of 0.003.

| Evaluation | Method | MAE (pp) | RMSE (pp) | Macro battery MAE (pp) |
|---|---|---:|---:|---:|
| temporal_test | persistence | 1.493 | 1.554 | 1.493 |
| temporal_test | linear_trend | 0.250 | 0.314 | 0.250 |
| temporal_test | gradient_boosting | 0.320 | 0.410 | 0.320 |
| unseen_battery_test | persistence | 0.962 | 1.074 | 0.962 |
| unseen_battery_test | linear_trend | 0.238 | 0.298 | 0.238 |
| unseen_battery_test | gradient_boosting | 0.226 | 0.296 | 0.226 |

Temporal test: later forecast origins on batteries represented in training.
Unseen-battery test: all windows from batteries excluded entirely from training.
The held-out metrics are evaluation only; hyperparameters are fixed in code.
Synthetic demo results establish software behavior, not real-world battery accuracy.
