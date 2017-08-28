"""Utility functions to protect main process against segfaults"""
import concurrent.futures


def segfault_protect(function, *args, **kwargs):
    """Used to run a function in a separate process to catch RuntimeErrors such
    as segfaults.
    """
    if segfault_protect.first_time_called:
        segfault_protect.first_time_called = False
    if segfault_protect.executor is None:
        # Create process pool with 1 member
        segfault_protect.executor = concurrent.futures.ProcessPoolExecutor(1)

    result = None
    try:
        # Submit function call to process pool and wait for an answer.
        futures = [segfault_protect.executor.submit(function, *args, **kwargs)]
        concurrent.futures.wait(futures)
        result = futures[0].result()
    except RuntimeError:
        # Need to reset executor
        segfault_protect.executor.shutdown()
        segfault_protect.executor = None
        raise RuntimeError("Encountered RuntimeError when calling function with 'segfault_protect'.")
    return result

segfault_protect.executor = None
segfault_protect.first_time_called = True
