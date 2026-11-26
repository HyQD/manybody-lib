"""Exact algebraic checks of the closed-shell doubles/triples metrics.

This test is deliberately independent of Spark and Drudge.  It verifies the
class-function coefficients used by ``common.metric_pinv`` against regular
representations of S2 and S3.
"""

from __future__ import annotations

import itertools

from sympy import eye, Matrix, Rational, zeros


def compose(left, right):
    """Return the permutation ``left o right``."""

    return tuple(left[right[position]] for position in range(len(left)))


def inverse(permutation):
    result = [None] * len(permutation)
    for position, image in enumerate(permutation):
        result[image] = position
    return tuple(result)


def cycles(permutation):
    seen = [False] * len(permutation)
    count = 0
    for start in range(len(permutation)):
        if seen[start]:
            continue
        count += 1
        position = start
        while not seen[position]:
            seen[position] = True
            position = permutation[position]
    return count


def regular_class_matrix(rank, weights):
    permutations = tuple(itertools.permutations(range(rank)))
    position = {permutation: index for index, permutation in enumerate(permutations)}
    matrix = zeros(len(permutations))
    for row, target in enumerate(permutations):
        for action in permutations:
            source = compose(inverse(action), target)
            matrix[row, position[source]] += weights[cycles(action)]
    return Matrix(matrix)


def validate_rank(rank, metric_weights, pinv_weights, expected_eigenvalues):
    metric = regular_class_matrix(rank, metric_weights)
    pinv = regular_class_matrix(rank, pinv_weights)
    dimension = metric.rows

    if rank == 3:
        null = Matrix.ones(dimension, dimension) / dimension
    else:
        null = zeros(dimension)
    physical = eye(dimension) - null

    assert metric * pinv == physical
    assert pinv * metric == physical
    assert metric * null == zeros(dimension)
    assert pinv * null == zeros(dimension)

    eigenvalues = []
    for value, multiplicity, _vectors in metric.eigenvects():
        eigenvalues.extend([value] * multiplicity)
    assert sorted(eigenvalues) == sorted(expected_eigenvalues)


def main() -> None:
    validate_rank(
        2,
        {2: 4, 1: -2},
        {2: Rational(1, 3), 1: Rational(1, 6)},
        (2, 6),
    )
    validate_rank(
        3,
        {3: 8, 2: -4, 1: 2},
        {
            3: Rational(17, 144),
            2: Rational(-1, 144),
            1: Rational(-7, 144),
        },
        (0, 6, 6, 6, 6, 24),
    )
    print("Matthews S2/S3 metric identities: OK")


if __name__ == "__main__":
    main()
