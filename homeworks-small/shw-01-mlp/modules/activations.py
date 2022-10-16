import numpy as np
import scipy
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, np.zeros_like(input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * np.where(input <= 0, 0, 1)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        neg_input_exp = np.exp(-input)
        return grad_output * neg_input_exp / ((1 + neg_input_exp) ** 2)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.softmax(input, axis=1)
        # batch_size, num_classes = input.shape
        # input_exp = np.exp(input)
        # input_exp_sum_by_batch = input_exp.sum(1)
        # return input_exp / input_exp_sum_by_batch.reshape((batch_size, 1)).repeat(num_classes, 1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # https://themaverickmeerkat.com/2019-10-23-Softmax/
        b, n = input.shape
        softmax_output = scipy.special.softmax(input, axis=1)

        softmax_tensor = np.einsum('ij,jk->ijk', softmax_output, np.eye(n, n))  # (b, n, n)
        softmax_tensor_squares = np.einsum('ij,ik->ijk', softmax_output, softmax_output)  # (b, n, n)
        softmax_grad_tensor = softmax_tensor - softmax_tensor_squares
        grad_input = np.einsum('ijk,ik->ij', softmax_grad_tensor, grad_output)  # (b, n)
        return grad_input
        # batch_size, num_classes = input.shape
        # input_exp = np.exp(input)
        # input_exp_sum_by_batch = input_exp.sum(1).reshape((batch_size, 1)).repeat(num_classes, 1)
        # softmax_output = input_exp / input_exp_sum_by_batch
        # grad_input = grad_output * (softmax_output - np.square(softmax_output))
        # return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)
        # batch_size, num_classes = input.shape
        # input_exp = np.exp(input)
        # input_exp_sum_by_batch = input_exp.sum(1)
        # softmax_output = input_exp / input_exp_sum_by_batch.reshape((batch_size, 1)).repeat(num_classes, 1)
        # return np.log(softmax_output)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # https://themaverickmeerkat.com/2019-10-23-Softmax/
        b, n = input.shape
        softmax_output = scipy.special.softmax(input, axis=1)

        softmax_rows_tensor = np.einsum('ij,ik->ijk', softmax_output, np.ones((b, n)))  # (b, n, n)
        ones_on_diag_tensor = np.einsum('ij,jk->ijk', np.ones((b, n)), np.eye(n, n))  # (b, n, n)
        softmax_grad_tensor = ones_on_diag_tensor - softmax_rows_tensor
        grad_input = np.einsum('ijk,ik->ij', softmax_grad_tensor, grad_output)  # (b, n)
        return grad_input

        # softmax_tensor = np.einsum('ij,jk->ijk', softmax_output, np.eye(n, n))  # (b, n, n)
        # softmax_squares_tensor = np.einsum('ij,ik->ijk', softmax_output, softmax_output)  # (b, n, n)
        # softmax_inverse_tensor = np.einsum('ij,jk->ijk', np.power(softmax_output, -1), np.eye(n, n))  # (b, n, n)
        # softmax_grad_tensor = softmax_inverse_tensor * (softmax_tensor - softmax_squares_tensor)
        # grad_input = np.einsum('ijk,ik->ij', softmax_grad_tensor, grad_output)  # (b, n)
        # return grad_input
