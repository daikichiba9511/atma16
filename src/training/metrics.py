def average_precision_at_k(actual: int, predicted: list[int], k: int = 10) -> float:
    """Calculate average precision at k.

    Args:
        actual: actual yad_no
        predicted: predicted yad_no
        k: top-k. default is 10

    Returns:
        the average precision at k over the input predictions

    References:
    [1]
    https://www.guruguru.science/competitions/22/discussions/a62aed5d-438d-46c7-8e44-85e5e5d41e64/
    """
    if actual not in predicted:
        return 0.0
    return 1.0 / (predicted.index(actual) + 1)


def mean_average_precision_at_k(actual: list[int], predicted: list[list[int]], k: int = 10) -> float:
    """Calculate mean average precision at k.

    Args:
        actual: actual yad_no
        predicted: predicted yad_no
        k: top-k. default is 10

    Returns:
        the mean average precision at k over the input predictions
    """
    return sum(average_precision_at_k(a, p, k) for a, p in zip(actual, predicted)) / len(actual)


def _test_map_at_10():
    y_true = [1, 2, 3]
    y_pred = [[2, 3, 4, 1, 5, 6, 7, 8, 9], [3, 2, 1, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    print("mAP@10: ", mean_average_precision_at_k(y_true, y_pred, k=10))


if __name__ == "__main__":
    _test_map_at_10()
