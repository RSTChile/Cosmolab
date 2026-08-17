#!/usr/bin/env python3
"""Prueba mínima: sdrplay_api_GetDevices vía libsdrplay_api (servicio en Pi)."""
import ctypes
import os
import sys

LIB = "/usr/local/lib/libsdrplay_api.so.3"
MAX_DEV = 16
SER_LEN = 64


class DeviceT(ctypes.Structure):
    _fields_ = [
        ("SerNo", ctypes.c_char * SER_LEN),
        ("hwVer", ctypes.c_ubyte),
        ("tuner", ctypes.c_uint),
        ("rspDuoMode", ctypes.c_uint),
        ("valid", ctypes.c_ubyte),
        ("rspDuoSampleFreq", ctypes.c_double),
        ("dev", ctypes.c_void_p),
    ]


def main() -> int:
    os.environ.setdefault("LD_LIBRARY_PATH", "/usr/local/lib")
    try:
        lib = ctypes.CDLL(LIB)
    except OSError as e:
        print(f"ERROR: no carga {LIB}: {e}")
        return 1

    api_ver = ctypes.c_float()
    err_names = {
        0: "Success", 1: "Fail", 14: "ServiceNotResponding", 24: "InvalidServiceVersion",
    }

    if hasattr(lib, "sdrplay_api_ApiVersion"):
        lib.sdrplay_api_ApiVersion.argtypes = [ctypes.POINTER(ctypes.c_float)]
        lib.sdrplay_api_ApiVersion.restype = ctypes.c_int
        r = lib.sdrplay_api_ApiVersion(ctypes.byref(api_ver))
        print(f"ApiVersion: {api_ver.value:.3f} (err={r} {err_names.get(r, r)})")

    if hasattr(lib, "sdrplay_api_Open"):
        lib.sdrplay_api_Open.argtypes = []
        lib.sdrplay_api_Open.restype = ctypes.c_int
        ro = lib.sdrplay_api_Open()
        print(f"Open: err={ro} ({err_names.get(ro, ro)})")
        if ro not in (0, 9):  # 9 = AlreadyInitialised
            return 2

    lib.sdrplay_api_GetDevices.argtypes = [
        ctypes.POINTER(DeviceT), ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
    ]
    lib.sdrplay_api_GetDevices.restype = ctypes.c_int

    devs = (DeviceT * MAX_DEV)()
    n = ctypes.c_uint(0)
    rc = lib.sdrplay_api_GetDevices(devs, ctypes.byref(n), MAX_DEV)
    print(f"GetDevices: err={rc} ({err_names.get(rc, 'code '+str(rc))}) count={n.value}")

    if rc != 0 or n.value == 0:
        print("FALLO — revisa: sudo systemctl status sdrplay; lsusb | grep 1df7")
        return 2

    for i in range(n.value):
        d = devs[i]
        sn = d.SerNo.split(b"\0", 1)[0].decode(errors="replace")
        print(f"  [{i}] SerNo={sn!r} hwVer={d.hwVer} valid={d.valid}")
    print("OK — RSP accesible por API")
    return 0


if __name__ == "__main__":
    sys.exit(main())