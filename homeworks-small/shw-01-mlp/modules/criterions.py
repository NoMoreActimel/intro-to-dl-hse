import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.mean(np.square(input - target))

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        b, n = input.shape
        return (2 * input - 2 * target) / (b * n)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size, num_classes = input.shape
        log_p = self.log_softmax.compute_output(input)

        reshaped_target = target.reshape((batch_size, 1)).repeat(num_classes, 1)
        reshaped_classes = np.arange(num_classes).reshape((1, num_classes)).repeat(batch_size, 0)
        indicators = np.array(reshaped_target == reshaped_classes, dtype=int)

        return -np.sum(log_p * indicators) / batch_size

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size, num_classes = input.shape

        reshaped_target = target.reshape((batch_size, 1)).repeat(num_classes, 1)
        reshaped_classes = np.arange(num_classes).reshape((1, num_classes)).repeat(batch_size, 0)
        indicators = np.array(reshaped_target == reshaped_classes, dtype=int)

        return -self.log_softmax.compute_grad_input(input, indicators) / batch_size
