"""
Constraints module for CISE.

Provides penalty functions that define constraints for energy-based reweighting.
"""

from cise.constraints.base import BaseConstraint, ConstraintType
from cise.constraints.smoothness import SmoothnessConstraint
from cise.constraints.l1 import L1Constraint
from cise.constraints.lowrank import LowRankConstraint
from cise.constraints.hierarchy import HierarchyConstraint

CONSTRAINTS = {
    "smoothness": SmoothnessConstraint,
    "l1": L1Constraint,
    "lowrank": LowRankConstraint,
    "hierarchy": HierarchyConstraint,
}

# Constraints applicable to each sample type
VECTOR_CONSTRAINTS = ["smoothness", "l1", "hierarchy"]
MATRIX_CONSTRAINTS = ["l1", "lowrank", "hierarchy"]


def get_constraint(name: str) -> type:
    """Get constraint class by name.

    Args:
        name: Constraint name.

    Returns:
        Constraint class.

    Raises:
        ValueError: If constraint name is unknown.
    """
    if name not in CONSTRAINTS:
        raise ValueError(f"Unknown constraint: {name}. Available: {list(CONSTRAINTS.keys())}")
    return CONSTRAINTS[name]


def get_constraints_for_type(sample_type: str) -> list:
    """Get list of applicable constraints for a sample type.

    Args:
        sample_type: Either 'vector' or 'matrix'.

    Returns:
        List of constraint names.
    """
    if sample_type == "vector":
        return VECTOR_CONSTRAINTS
    elif sample_type == "matrix":
        return MATRIX_CONSTRAINTS
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")


__all__ = [
    "BaseConstraint",
    "ConstraintType",
    "SmoothnessConstraint",
    "L1Constraint",
    "LowRankConstraint",
    "HierarchyConstraint",
    "get_constraint",
    "get_constraints_for_type",
    "CONSTRAINTS",
    "VECTOR_CONSTRAINTS",
    "MATRIX_CONSTRAINTS",
]
