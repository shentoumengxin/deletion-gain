"""Logging for the paths that switch the defense off without failing.

Every failure on the serving path degrades to a miss, so nothing here ever raises and
nothing here is on the fast path of a served request. But "degrades to a miss" is
exactly what makes these failures invisible: a fleet-wide embedder change, or a
``--policy`` sweep left half applied, mismatches every profile against the fence, misses
every lookup, and drives the hit rate to zero while looking identical to a cold cache.
A counter nobody polls is not evidence. A log line is.

:func:`warn_once` keeps that cheap. A misconfiguration repeats on *every* request, so
warning per request would put an I/O call in a hot loop and bury the first line under a
million copies of itself. One line per distinct cause -- the key names the cause, not the
request -- says everything the operator needs, because the second occurrence carries no
information the first did not.

The keys are deliberately built from configuration values (fingerprints, embedder names,
span policies) and never from request content, so the set stays bounded no matter how
much traffic arrives.
"""

from __future__ import annotations

import logging
import threading

#: Named for the package rather than this module, so a deployment silences or routes
#: the whole defense with one ``logging.getLogger("sentry.cache.defense")`` call.
logger = logging.getLogger("sentry.cache.defense")

_lock = threading.Lock()
_seen: set[str] = set()


def warn_once(key: str, message: str, *args: object) -> bool:
    """Emit ``message`` at WARNING the first time this ``key`` is seen.

    Returns whether the line was emitted, which is what a test asserts on -- the
    second call for a key is a no-op by design and a test that could not tell the two
    apart would not be testing the rate limit.
    """
    with _lock:
        if key in _seen:
            return False
        _seen.add(key)
    logger.warning(message, *args)
    return True


def reset_warnings() -> None:
    """Forget which keys have been warned about. For tests only.

    The suppression set is process-global on purpose -- a misconfiguration is a property
    of the deployment, not of one object -- which means one test's warning would
    otherwise silence the next test's.
    """
    with _lock:
        _seen.clear()
