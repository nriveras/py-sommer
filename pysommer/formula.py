"""Lightweight formula and variance-structure helpers for pysommer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ISM:
    """Identity covariance structure marker for a random effect variable."""

    var: str


@dataclass(frozen=True)
class DSM:
    """Diagonal-by-level structure marker (e.g., environment-specific effects)."""

    var: str


@dataclass(frozen=True)
class USM:
    """Unstructured marker (reserved for future expansion)."""

    var: str


@dataclass(frozen=True)
class VSMCall:
    """Container for vsm-like declarations.

    Example:
        vsm(dsm("Env"), ism("id"), Gu=A)
    """

    structures: tuple[Any, ...]
    Gu: np.ndarray | None = None


def ism(var: str) -> ISM:
    return ISM(var=var)


def dsm(var: str) -> DSM:
    return DSM(var=var)


def usm(var: str) -> USM:
    return USM(var=var)


def vsm(*structures: Any, Gu: np.ndarray | None = None) -> VSMCall:
    return VSMCall(structures=tuple(structures), Gu=Gu)


def _fetch_col(data: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Column '{key}' was not found in data")
    col = np.asarray(data[key])
    if col.ndim != 1:
        raise ValueError(f"Column '{key}' must be 1D")
    return col


def _encode_categorical(x: np.ndarray, drop_first: bool) -> tuple[np.ndarray, np.ndarray]:
    levels, inv = np.unique(x, return_inverse=True)
    mat = np.eye(levels.size, dtype=float)[inv]
    if drop_first and mat.shape[1] > 1:
        return mat[:, 1:], levels
    return mat, levels


def parse_fixed_formula(fixed: str, data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse a minimal fixed formula into y and X.

    Supported patterns:
    - "y ~ 1"
    - "y ~ x1 + x2"
    - "y ~ x1 + cat" (categorical terms are one-hot with drop-first)
    - "y ~ x1:x2" (numeric interaction)
    """
    if "~" not in fixed:
        raise ValueError("fixed must have form 'response ~ terms'")

    lhs, rhs = [p.strip() for p in fixed.split("~", 1)]
    y = _fetch_col(data, lhs).astype(float).reshape(-1, 1)
    n = y.shape[0]

    rhs_terms = [t.strip() for t in rhs.split("+") if t.strip()]
    if len(rhs_terms) == 0:
        rhs_terms = ["1"]

    x_cols: list[np.ndarray] = []
    names: list[str] = []
    has_intercept = any(t == "1" for t in rhs_terms)

    if has_intercept:
        x_cols.append(np.ones((n, 1), dtype=float))
        names.append("Intercept")

    for term in rhs_terms:
        if term == "1":
            continue
        if ":" in term:
            a, b = [s.strip() for s in term.split(":", 1)]
            va = _fetch_col(data, a).astype(float)
            vb = _fetch_col(data, b).astype(float)
            x_cols.append((va * vb)[:, None])
            names.append(f"{a}:{b}")
            continue

        col = _fetch_col(data, term)
        if np.issubdtype(col.dtype, np.number):
            x_cols.append(col.astype(float)[:, None])
            names.append(term)
        else:
            dummies, levels = _encode_categorical(col, drop_first=True)
            if dummies.shape[1] == 0:
                continue
            x_cols.append(dummies)
            names.extend([f"{term}[{lev}]" for lev in levels[1:]])

    if len(x_cols) == 0:
        x = np.ones((n, 1), dtype=float)
        names = ["Intercept"]
    else:
        x = np.hstack(x_cols)

    return y, x, names


def build_random_from_vsm(
    random: Sequence[VSMCall],
    data: Mapping[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Build Z and K terms from a restricted vsm-like interface.

    Implemented patterns:
    - vsm(ism(group), Gu=K)
    - vsm(dsm(env), ism(group), Gu=K)
    """
    z_terms: list[np.ndarray] = []
    k_terms: list[np.ndarray] = []
    names: list[str] = []

    for call in random:
        structures = call.structures
        if len(structures) == 1 and isinstance(structures[0], ISM):
            g = _fetch_col(data, structures[0].var)
            z, levels = _encode_categorical(g, drop_first=False)
            k = call.Gu if call.Gu is not None else np.eye(levels.size, dtype=float)
            k = np.asarray(k, dtype=float)
            if k.shape != (levels.size, levels.size):
                raise ValueError(
                    f"Gu for {structures[0].var} must be square with size {levels.size}"
                )
            z_terms.append(z)
            k_terms.append(k)
            names.append(f"ism({structures[0].var})")
            continue

        if (
            len(structures) == 2
            and isinstance(structures[0], DSM)
            and isinstance(structures[1], ISM)
        ):
            env = _fetch_col(data, structures[0].var)
            grp = _fetch_col(data, structures[1].var)

            z_base, grp_levels = _encode_categorical(grp, drop_first=False)
            env_levels, env_inv = np.unique(env, return_inverse=True)

            k = call.Gu if call.Gu is not None else np.eye(grp_levels.size, dtype=float)
            k = np.asarray(k, dtype=float)
            if k.shape != (grp_levels.size, grp_levels.size):
                raise ValueError(
                    f"Gu for {structures[1].var} must be square with size {grp_levels.size}"
                )

            for idx, lev in enumerate(env_levels):
                mask = (env_inv == idx).astype(float)[:, None]
                z_terms.append(mask * z_base)
                k_terms.append(k)
                names.append(f"dsm({structures[0].var}={lev})xism({structures[1].var})")
            continue

        # Usm placeholder for future extension.
        if any(isinstance(s, USM) for s in structures):
            raise NotImplementedError(
                "usm() parsing is not implemented yet in formula interface"
            )

        raise NotImplementedError(
            "Unsupported random structure. Supported: vsm(ism(...)) and vsm(dsm(...), ism(...))"
        )

    return z_terms, k_terms, names
