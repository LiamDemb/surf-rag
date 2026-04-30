"""Shared dense-weight grid for oracle curves and router policies."""

# Fixed 11-bin dense weight grid. Graph weight is always ``1 - dense_weight``.
DEFAULT_DENSE_WEIGHT_GRID: tuple[float, ...] = tuple(
    round(i / 10.0, 1) for i in range(11)
)
