"""Copairs backend selection and lazy module loading for benchmark metrics.

:func:`_get_legacy_modules` / :func:`_get_modern_modules` import the optional
copairs API lazily (inside the function bodies), so importing this module never
requires copairs installed. Callers reach these through a module reference so
tests can monkeypatch the loaders.
"""

from typing import Literal

# Canonical modes:
# - stable: old copairs API (paper-compatible behavior)
# - experimental: new copairs API
# Backward-compatible aliases:
# - legacy -> stable
# - modern -> experimental
CopairsMode = Literal["stable", "experimental", "legacy", "modern"]

EXPERIMENTAL_COPAIRS_ERROR = (
    "Experimental copairs mode requires the new copairs API. Install a recent copairs release."
)

STABLE_COPAIRS_ERROR = (
    "Stable copairs mode requires the old copairs API. "
    "Install the reference version from the paper environment:\n"
    "pip install git+https://github.com/cytomining/copairs@"
    "880f22a551bd897896d148a0b07baa99d981c6a9"
)


def _is_multiprocessing_permission_error(exc: Exception) -> bool:
    """Detect sandbox/runtime failures from old copairs multiprocessing helpers."""
    if isinstance(exc, PermissionError):
        return True

    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, PermissionError):
        return True

    message = str(exc).lower()
    return "semlock" in message and "permission denied" in message


def _normalize_copairs_mode(copairs_mode: CopairsMode) -> Literal["stable", "experimental"]:
    """Normalize copairs mode and keep backward compatibility."""
    if copairs_mode in ("stable", "legacy"):
        return "stable"
    if copairs_mode in ("experimental", "modern"):
        return "experimental"
    raise ValueError(
        f"Unsupported copairs_mode='{copairs_mode}'. "
        "Use one of: stable, experimental (aliases: legacy, modern)."
    )


def _get_modern_modules():
    """Import modern copairs modules required for experimental mode."""
    try:
        from copairs.map import mean_average_precision  # type: ignore
        from copairs.map.average_precision import (
            average_precision as average_precision_single,  # type: ignore
        )
        from copairs.map.multilabel import (
            average_precision as average_precision_multilabel,  # type: ignore
        )
    except Exception as exc:  # pragma: no cover - depends on installed copairs version
        raise RuntimeError(EXPERIMENTAL_COPAIRS_ERROR) from exc

    return {
        "mean_average_precision": mean_average_precision,
        "average_precision_single": average_precision_single,
        "average_precision_multilabel": average_precision_multilabel,
    }


def _get_legacy_modules():
    """Import old copairs modules required for stable mode."""
    try:
        import copairs.compute_np as backend  # type: ignore
        from copairs.compute import cosine_indexed  # type: ignore
        from copairs.map import (
            aggregate,  # type: ignore
            build_rank_list_multi,  # type: ignore
            build_rank_lists,  # type: ignore
            results_to_dframe,  # type: ignore
        )
        from copairs.matching import Matcher, MatcherMultilabel, dict_to_dframe
    except Exception as exc:  # pragma: no cover - depends on installed copairs version
        raise RuntimeError(STABLE_COPAIRS_ERROR) from exc

    return {
        "backend": backend,
        "cosine_indexed": cosine_indexed,
        "aggregate": aggregate,
        "build_rank_list_multi": build_rank_list_multi,
        "build_rank_lists": build_rank_lists,
        "results_to_dframe": results_to_dframe,
        "Matcher": Matcher,
        "MatcherMultilabel": MatcherMultilabel,
        "dict_to_dframe": dict_to_dframe,
    }
