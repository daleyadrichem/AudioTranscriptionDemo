"""
Base classes and interfaces for AI use cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class UseCase(ABC):
    """Abstract base class for AI use cases.

    Subclasses should implement the :meth:`run` method to start any
    UI or command-line loop.

    Attributes
    ----------
    name : str
        Human-readable name of the use case.
    """

    name: str = "Unnamed Use Case"

    @abstractmethod
    def run(self) -> None:
        """Run the use case.

        This method is typically responsible for starting a UI loop,
        command-line interaction, or other application flow.
        """
        raise NotImplementedError
