"""
Generic controller-plant verification framework.

This package provides abstract interfaces for separating controller and plant
logic in cyber-physical system verification.
"""

from .controller import Controller
from .plant import Plant
from .maestro import simulate, verify

__all__ = ['Controller', 'Plant', 'simulate', 'verify']
