#!/usr/bin/env python3

import functools
import os

import numpy as np

from mpi import (  # isort:skip
    init_env,
    mpi_frame,
    split_first_dim,
    split_last_dim,
)

__all__ = [
    "Linear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelFFN",
]


class Linear:
    """
    The linear layer is defined as y = x * w + b.
    """

    def __init__(self, weights, bias=None):
        self.weight = weights
        self.bias = bias
        self.context = {}
        self.grad = {}

    def forward(self, x):
        self.context["x"] = x
        y = np.dot(x, self.weight)
        if self.bias:
            y += self.bias
        return y

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad):
        # y = x * w + b
        input_data = self.context.pop("x")

        # TODO implement this function
        if self.bias:
            self.grad["bias"] = ?

        self.grad["weight"] = ?

        prev_layer_grad = ?
        return prev_layer_grad

    def step_grad(self, learning_rate: float):
        grad_weight = self.grad.pop("weight")
        self.weight -= learning_rate * grad_weight
        if self.bias:
            grad_bias = self.grad.pop("bias")
            self.bias -= learning_rate * grad_bias


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism.
    """
    def __init__(self, weights, bias=None, gather_output: bool = True):
        """
        Args:
            gather_output(bool): whether to gather the output from all processes or not.
        """
        # TODO: implement this function
        pass

    def forward(self, x):
        # TODO: implement this function
        pass

    def backward(self, grad):
        # TODO: implement this function
        pass


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism. The linear layer is defined as y = x * w + b.
    """
    def __init__(self, weights, bias=None, input_is_parallel: bool = False):
        """
        Args:
            input_is_parallel(bool): whether the input x is already parallelized along the last dim
        """
        # TODO: implement this function
        pass

    def forward(self, x):
        # TODO: implement this function
        pass

    def backward(self, grad):
        # TODO: implement this function
        pass


class ParallelFFN:

    """
    This would save communication cost if complex activation function is used,
    for example, relu(x1 + x2) != relu(x1) + relu(x2)
    """

    def __init__(self, w1, w2):
        # w1: weights of the 1st layer
        # w2: weights of the 2nd layer
        # TODO: implement this function
        pass

    def forward(self, x):
        # TODO: implement this function
        pass

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad):
        # TODO: implement this function
        pass


# ----------------------------- test functions ----------------------------- #

def linear_step():
    np.random.seed(42)

    batch_size = 16
    input_dim, output_dim = 128, 32
    lr = 1

    w = np.random.rand(input_dim, output_dim)
    data = np.random.rand(batch_size, input_dim)
    linear = Linear(weights=w)
    output = linear(data)
    grad_output = np.random.rand(*output.shape)
    data_grad = linear.backward(grad_output)
    linear.step_grad(lr)

    np.save("linear_weight.npy", linear.weight)
    np.save("linear_output.npy", output)
    np.save("linear_data_grad.npy", data_grad)


def col_linear(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    np.random.seed(42)

    batch_size = 16
    input_dim, output_dim = 128, 32
    lr = 1

    w = np.random.rand(input_dim, output_dim)
    data = np.random.rand(batch_size, input_dim)

    w_parallel = split_last_dim(w)
    print(f"rank {rank} w_parallel shape: {w_parallel.shape}")
    linear = ColumnParallelLinear(weights=w_parallel, gather_output=True)

    output = linear(data)
    grad_output = np.random.rand(*output.shape)
    data_grad = linear.backward(grad_output)
    linear.step_grad(lr)

    np.save(f"linear_weight_rank{rank}.npy", linear.weight)
    np.save(f"linear_output_rank{rank}.npy", output)
    np.save(f"linear_data_grad_rank{rank}.npy", data_grad)


def row_linear(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    np.random.seed(42)

    batch_size = 16
    input_dim, output_dim = 128, 32
    lr = 1

    w = np.random.rand(input_dim, output_dim)
    data = np.random.rand(batch_size, input_dim)

    w_parallel = split_first_dim(w)
    print(f"rank {rank} w_parallel shape: {w_parallel.shape}")
    linear = RowParallelLinear(weights=w_parallel)

    output = linear(data)
    grad_output = np.random.rand(*output.shape)
    data_grad = linear.backward(grad_output)
    linear.step_grad(lr)

    np.save(f"linear_weight_rank{rank}.npy", linear.weight)
    np.save(f"linear_output_rank{rank}.npy", output)
    np.save(f"linear_data_grad_rank{rank}.npy", data_grad)


def col_parallel_step():
    world_size = 4
    mpi_frame(col_linear, world_size=world_size)


def row_parallel_step():
    world_size = 4
    mpi_frame(row_linear, world_size=world_size)


def check(prefix: str):
    print(f"\nchecking {prefix}...")
    out = np.load(f"{prefix}.npy")
    for r in range(8):
        load_name = f"{prefix}_rank{r}.npy"
        if os.path.exists(load_name):
            data = np.load(load_name)
            assert np.allclose(out, data)

    print(f"checking {prefix} passed.")


def check_weights(para_type: str = "col"):
    assert para_type in ["col", "row"]
    axis = -1 if para_type == "col" else 0
    print(f"\nchecking {para_type} weights...")
    out = np.load("linear_weight.npy")

    weights = []
    for r in range(8):
        load_name = f"linear_weight_rank{r}.npy"
        if os.path.exists(load_name):
            data = np.load(load_name)
            weights.append(data)
    w = np.concatenate(weights, axis=axis)
    assert np.allclose(out, w, atol=1e-8)

    print(f"checking {para_type} weights passed.")


check_output = functools.partial(check, prefix="linear_output")
check_grad = functools.partial(check, prefix="linear_data_grad")


def test_col_linear():
    linear_step()
    col_parallel_step()

    check_output()
    check_grad()
    check_weights(para_type="col")


def test_row_linear():
    linear_step()
    row_parallel_step()

    check_output()
    check_grad()
    check_weights(para_type="row")


def parallel_ffn(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    np.random.seed(42)
    batch, input_dim, hidden_dim, output_dim = 16, 128, 64, 32
    w1, w2 = np.random.rand(input_dim, hidden_dim), np.random.rand(hidden_dim, output_dim)
    data = np.random.rand(batch, input_dim)
    grad = np.random.rand(batch, output_dim)

    model = ParallelFFN(w1, w2)
    output = model(data)
    grad_out = model.backward(grad)
    np.save(f"ffn_output_rank{rank}.npy", output)
    np.save(f"ffn_data_grad_rank{rank}.npy", grad_out)


def test_parallel_ffn():
    np.random.seed(42)
    batch, input_dim, hidden_dim, output_dim = 16, 128, 64, 32
    w1, w2 = np.random.rand(input_dim, hidden_dim), np.random.rand(hidden_dim, output_dim)
    data = np.random.rand(batch, input_dim)
    grad = np.random.rand(batch, output_dim)

    l1, l2 = Linear(w1), Linear(w2)
    output = l2(l1(data))
    grad_out = l1.backward(l2.backward(grad))
    np.save("ffn_output.npy", output)
    np.save("ffn_data_grad.npy", grad_out)

    world_size = 4
    mpi_frame(parallel_ffn, world_size=world_size)
    check_output(prefix="ffn_output")
    check_grad(prefix="ffn_data_grad")


if __name__ == "__main__":
    print("Test linear layer with column parallelism...")
    test_col_linear()

    print("\n\nTest linear layer with row parallelism...")
    test_row_linear()

    print("\n\nTest parallel FFN layer...")
    test_parallel_ffn()
