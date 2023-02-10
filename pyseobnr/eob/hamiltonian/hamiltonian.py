#!/usr/bin/env python3
"""
Abstract Hamiltonian classes
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import grad, jit, vmap


class Hamiltonian:
    """Aligned-spin Hamiltonian class"""
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


class Hamiltonian_v5PHM:
    """Precessing-spin Hamiltonian class"""
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, q,p,chi1_v,chi2_v,m_1,m_2,chi_1, chi_2, chiL_1, chiL_2, **kwargs):
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
