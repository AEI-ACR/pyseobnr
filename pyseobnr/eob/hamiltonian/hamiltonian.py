#!/usr/bin/env python3
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import grad, jit, vmap


class Hamiltonian:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, q, p, chi_1, chi_2, **kwargs):
        """Evaluate the Hamiltonian"""
        pass

    @abstractmethod
    def grad(self):
        """Return the gradient of the Hamiltonian with resepct to dynamical vars"""
        pass

    @abstractmethod
    def hessian(self):
        """Return the Hessian of the Hamiltonian with resepct to dynamical vars. Needed for IC"""
        pass
