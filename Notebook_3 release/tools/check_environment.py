"""Fail before data staging when a notebook retains incompatible NumPy binaries."""
from importlib import metadata

def check_numpy():
    try:
        import numpy as np
        installed = metadata.version('numpy')
        if np.__version__ != installed:
            raise RuntimeError(f'Loaded NumPy {np.__version__}, installed NumPy {installed}')
        # This imports the Python utilities that call the compiled BLAS API.
        import numpy.testing
        numpy.testing.assert_array_equal(np.array([1]), np.array([1]))
    except (ImportError, AttributeError, RuntimeError) as error:
        raise RuntimeError(
            'NumPy installation / loaded kernel is inconsistent. '
            'Select Runtime > Restart session (keep this VM), then rerun setup. '
            'Do not disconnect/delete runtime or delete dataset/cache. '
            'If it persists after restarting, run repair_numpy_cell.txt and restart again. '
            f'Original error: {error}'
        ) from error
    print(f'Environment OK: NumPy {installed}; numpy.testing / compiled API verified', flush=True)
