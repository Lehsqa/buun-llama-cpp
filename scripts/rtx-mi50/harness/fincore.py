#!/usr/bin/env python3
"""fincore.py FILE... -- resident MiB / total MiB per file (page cache), via mmap + mincore."""
import ctypes, os, sys

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = os.sysconf("SC_PAGE_SIZE")
ODD = bytes(i & 1 for i in range(256))

def resident(path):
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed: " + path)
        n = (size + PAGE - 1) // PAGE
        vec = (ctypes.c_ubyte * n)()
        if libc.mincore(addr, size, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed: " + path)
        libc.munmap(addr, size)
        return bytes(vec).translate(ODD).count(1) * PAGE, size
    finally:
        os.close(fd)

tot_r = tot_s = 0
for p in sys.argv[1:]:
    r, s = resident(p)
    tot_r += r; tot_s += s
    print("%10.1f / %10.1f MiB  %s" % (r / 2**20, s / 2**20, os.path.basename(p)))
print("%10.1f / %10.1f MiB  TOTAL" % (tot_r / 2**20, tot_s / 2**20))
