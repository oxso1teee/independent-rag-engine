"""Returning freed heap memory to the operating system.

Dropping a model only makes it collectable, and the C allocator usually keeps the freed pages for
itself: measured with the real E5 model, ``release()`` plus ``gc`` gave back about a third of the
resident set, and ``malloc_trim`` most of the rest (``eval/scripts/release_memory.py``). Where the
platform has no ``malloc_trim`` this does nothing and says so.
"""

from __future__ import annotations

import ctypes
import gc


def trim_heap() -> bool:
    """Collect garbage and ask glibc to return free pages; ``True`` when the call was available."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        return False
    return True
