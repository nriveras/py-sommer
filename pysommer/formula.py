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
    Cu: np.ndarray | None = None


def ism(var: str) -> ISM:
    """Declare an identity-structured random effect for a variable."""
    return ISM(var=var)


def dsm(var: str) -> DSM:
    """Declare a diagonal-by-level random-effect structure for a variable."""
    return DSM(var=var)


def usm(var: str) -> USM:
    """Declare an unstructured random-effect marker for a variable."""
    return USM(var=var)


def vsm(*structures: Any, Gu: np.ndarray | None = None, Cu: np.ndarray | None = None) -> VSMCall:
    """Build a lightweight sommer-style variance-structure declaration."""
    return VSMCall(structures=tuple(structures), Gu=Gu, Cu=Cu)


def _fetch_col(data: Mapping[str, Any], key: str) -> np.ndarray:
    """Fetch a 1D column from formula-mode data by key."""
    if key not in data:
        raise KeyError(f"Column '{key}' was not found in data")
    col = np.asarray(data[key])
    if col.ndim != 1:
        raise ValueError(f"Column '{key}' must be 1D")
    return col


def _encode_categorical(x: np.ndarray, drop_first: bool) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode a categorical vector and return encoded levels."""
    levels, inv = np.unique(x, return_inverse=True)
    mat = np.eye(levels.size, dtype=float)[inv]
    if drop_first and mat.shape[1] > 1:
        return mat[:, 1:], levels
    return mat, levels


def _normalize_level(value: Any) -> Any:
    """Convert NumPy scalar values into plain Python scalars for dictionary lookup."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_level_index(levels: np.ndarray) -> dict[Any, int]:
    """Build a lookup from encoded level value to column position."""
    return {_normalize_level(level): idx for idx, level in enumerate(levels.tolist())}


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

    lhs_terms = [t.strip() for t in lhs.split("+") if t.strip()]
    if len(lhs_terms) == 0:
        raise ValueError("fixed must include at least one response on the left-hand side")

    y_cols = [_fetch_col(data, name).astype(float).reshape(-1, 1) for name in lhs_terms]
    n = y_cols[0].shape[0]
    for col in y_cols:
        if col.shape[0] != n:
            raise ValueError("All response columns must have the same number of rows")
    y = np.hstack(y_cols)

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
    return_metadata: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]] | tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    list[dict[str, Any]],
]:
    """Build Z and K terms from a restricted vsm-like interface.

    Implemented patterns:
    - vsm(ism(group), Gu=K)
    - vsm(dsm(env), ism(group), Gu=K)
    - vsm(usm(env), ism(group), Gu=K, Cu=C)
    """
    z_terms: list[np.ndarray] = []
    k_terms: list[np.ndarray] = []
    names: list[str] = []
    metadata: list[dict[str, Any]] = []

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
            name = f"ism({structures[0].var})"
            names.append(name)
            metadata.append(
                {
                    "kind": "ism",
                    "var": structures[0].var,
                    "levels": levels.copy(),
                    "name": name,
                }
            )
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

            term_names: list[str] = []
            for idx, lev in enumerate(env_levels):
                mask = (env_inv == idx).astype(float)[:, None]
                z_terms.append(mask * z_base)
                k_terms.append(k)
                name = f"dsm({structures[0].var}={lev})xism({structures[1].var})"
                names.append(name)
                term_names.append(name)
            metadata.append(
                {
                    "kind": "dsm_ism",
                    "env_var": structures[0].var,
                    "group_var": structures[1].var,
                    "env_levels": env_levels.copy(),
                    "group_levels": grp_levels.copy(),
                    "term_names": term_names,
                }
            )
            continue

        if (
            len(structures) == 2
            and isinstance(structures[0], USM)
            and isinstance(structures[1], ISM)
        ):
            env = _fetch_col(data, structures[0].var)
            grp = _fetch_col(data, structures[1].var)

            z_base, grp_levels = _encode_categorical(grp, drop_first=False)
            env_levels, env_inv = np.unique(env, return_inverse=True)

            gu = call.Gu if call.Gu is not None else np.eye(grp_levels.size, dtype=float)
            gu = np.asarray(gu, dtype=float)
            if gu.shape != (grp_levels.size, grp_levels.size):
                raise ValueError(
                    f"Gu for {structures[1].var} must be square with size {grp_levels.size}"
                )

            cu = call.Cu if call.Cu is not None else np.eye(env_levels.size, dtype=float)
            cu = np.asarray(cu, dtype=float)
            if cu.shape != (env_levels.size, env_levels.size):
                raise ValueError(
                    f"Cu for {structures[0].var} must be square with size {env_levels.size}"
                )

            z_blocks = []
            for idx in range(env_levels.size):
                mask = (env_inv == idx).astype(float)[:, None]
                z_blocks.append(mask * z_base)

            z_combined = np.hstack(z_blocks)
            k_combined = np.kron(cu, gu)

            z_terms.append(z_combined)
            k_terms.append(k_combined)
            name = f"usm({structures[0].var})xism({structures[1].var})"
            names.append(name)
            metadata.append(
                {
                    "kind": "usm_ism",
                    "env_var": structures[0].var,
                    "group_var": structures[1].var,
                    "env_levels": env_levels.copy(),
                    "group_levels": grp_levels.copy(),
                    "name": name,
                }
            )
            continue

        raise NotImplementedError(
            "Unsupported random structure. Supported: vsm(ism(...)), vsm(dsm(...), ism(...)), vsm(usm(...), ism(...))"
        )

    if return_metadata:
        return z_terms, k_terms, names, metadata
    return z_terms, k_terms, names


def build_prediction_random_terms(
    random_metadata: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Build aligned random-effect design matrices for new data.

    New rows that reference unseen factor levels are zeroed for the affected
    random-effect terms so predictions fall back to the fixed-effect component.
    """
    z_terms: list[np.ndarray] = []
    status: list[dict[str, Any]] = []

    for meta in random_metadata:
        kind = str(meta["kind"])

        if kind == "ism":
            g = _fetch_col(data, str(meta["var"]))
            levels = np.asarray(meta["levels"])
            level_index = _build_level_index(levels)

            z = np.zeros((g.size, levels.size), dtype=float)
            matched = np.zeros(g.size, dtype=bool)
            for row, value in enumerate(g):
                idx = level_index.get(_normalize_level(value))
                if idx is not None:
                    z[row, idx] = 1.0
                    matched[row] = True

            z_terms.append(z)
            status.append(
                {
                    "name": str(meta["name"]),
                    "matched_rows": int(np.count_nonzero(matched)),
                    "zeroed_rows": int(g.size - np.count_nonzero(matched)),
                }
            )
            continue

        if kind == "dsm_ism":
            env = _fetch_col(data, str(meta["env_var"]))
            grp = _fetch_col(data, str(meta["group_var"]))
            env_levels = np.asarray(meta["env_levels"])
            grp_levels = np.asarray(meta["group_levels"])
            env_index = _build_level_index(env_levels)
            grp_index = _build_level_index(grp_levels)
            term_names = [str(name) for name in meta["term_names"]]

            z_by_env = [np.zeros((env.size, grp_levels.size), dtype=float) for _ in term_names]

            for row, (env_value, grp_value) in enumerate(zip(env, grp)):
                env_idx = env_index.get(_normalize_level(env_value))
                grp_idx = grp_index.get(_normalize_level(grp_value))
                if env_idx is not None and grp_idx is not None:
                    z_by_env[env_idx][row, grp_idx] = 1.0

            z_terms.extend(z_by_env)
            for term_name, z_term in zip(term_names, z_by_env):
                matched_rows = int(np.count_nonzero(np.any(z_term != 0.0, axis=1)))
                status.append(
                    {
                        "name": term_name,
                        "matched_rows": matched_rows,
                        "zeroed_rows": int(env.size - matched_rows),
                    }
                )
            continue

        if kind == "usm_ism":
            env = _fetch_col(data, str(meta["env_var"]))
            grp = _fetch_col(data, str(meta["group_var"]))
            env_levels = np.asarray(meta["env_levels"])
            grp_levels = np.asarray(meta["group_levels"])
            env_index = _build_level_index(env_levels)
            grp_index = _build_level_index(grp_levels)

            z = np.zeros((env.size, env_levels.size * grp_levels.size), dtype=float)
            matched = np.zeros(env.size, dtype=bool)

            for row, (env_value, grp_value) in enumerate(zip(env, grp)):
                env_idx = env_index.get(_normalize_level(env_value))
                grp_idx = grp_index.get(_normalize_level(grp_value))
                if env_idx is not None and grp_idx is not None:
                    col = env_idx * grp_levels.size + grp_idx
                    z[row, col] = 1.0
                    matched[row] = True

            z_terms.append(z)
            status.append(
                {
                    "name": str(meta["name"]),
                    "matched_rows": int(np.count_nonzero(matched)),
                    "zeroed_rows": int(env.size - np.count_nonzero(matched)),
                }
            )
            continue

        raise NotImplementedError(f"Unsupported prediction metadata kind: {kind}")

    return z_terms, status
