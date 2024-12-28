"""Utilities for the assignment."""

import autograd.numpy as np


class MultivaraiateNormal:
    """A multivaraite normal distribution."""

    @staticmethod
    def pdf(x: np.ndarray, mu: np.ndarray, cor: np.ndarray) -> np.ndarray:
        """The probability density function."""
        return np.exp(MultivaraiateNormal.logpdf(x, mu, cor))

    @staticmethod
    def logpdf(x: np.ndarray, mu: np.ndarray, cor: np.ndarray) -> np.ndarray:
        """The logarithm of the pdf."""
        return MultivaraiateNormal._logpdf_std(x.T - mu.T, cor)

    @staticmethod
    def _logpdf_std(x: np.ndarray, cor: np.ndarray) -> np.ndarray:
        """The logarithm of the pdf of a Gaussian with zero mean."""
        k = x.shape[-1]

        return -0.5 * (
            k * np.log(2 * np.pi)
            + np.log(np.linalg.det(cor))
            + np.einsum(
                "...i,ij,j...->...",
                x,
                np.linalg.inv(cor),
                x.T,
            )
        )
