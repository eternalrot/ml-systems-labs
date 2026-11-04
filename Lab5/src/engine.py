"""Scalar automatic differentiation engine (reverse mode).

Every Value is a node of a computation graph. A node stores
  data      the forward value
  grad      d(output)/d(this node), filled by backward()
  _prev     the parent nodes it was computed from
  _backward the local derivative rule (closure) that pushes this node's grad to its parents
The graph is built during the forward pass, simply by using the operators.
"""
import math


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"

    # ---- required operations: add, mul, pow, relu -------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():            # d(a^k)/da = k * a^(k-1)
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():            # 1 if input > 0 else 0
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---- extra operations needed by sigmoid + BCE -------------------------
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        # numerically stable form
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():            # s * (1 - s)
            self.grad += out.data * (1.0 - out.data) * out.grad
        out._backward = _backward
        return out

    # ---- sugar built from the primitives ----------------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    # ---- backward pass -----------------------------------------------------
    def backward(self):
        # 1) topological order (iterative DFS: deep graphs must not hit the recursion limit)
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        # 2) output gradient = 1
        self.grad = 1.0
        # 3) reverse order, each local rule ACCUMULATES (+=) into the parents
        for node in reversed(topo):
            node._backward()


def zero_grad(root):
    """Reset .grad of every node reachable from root (call before a new backward on the same graph)."""
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        n.grad = 0.0
        stack.extend(n._prev)
