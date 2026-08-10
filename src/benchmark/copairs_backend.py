"""Lazy loading of the pinned copairs API used by the benchmark metrics.

:func:`_get_legacy_modules` imports copairs inside the function body, so
importing this module never requires copairs installed. Callers reach it through
a module reference so tests can monkeypatch the loader.
"""

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


def _get_legacy_modules():
    """Import old copairs modules required for stable mode."""
    try:
        import copairs.compute_np as backend
        from copairs.compute import cosine_indexed
        from copairs.map import (
            aggregate,
            build_rank_list_multi,
            build_rank_lists,
            results_to_dframe,
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
