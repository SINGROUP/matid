"""Utility functions to protect main process against segfaults"""
import signal


def sig_handler(signum, frame):
    raise RuntimeError("SIGSEGV encountered.")


def segfault_protect(function, *args, **kwargs):
    """Used to run a function in a separate process to catch RuntimeErrors such
    as segfaults.
    """
    signal.signal(signal.SIGSEGV, sig_handler)
    result = function(*args)
    return result
