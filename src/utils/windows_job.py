"""
Windows Process & System Utilities.
1. Windows Job Object: Ensures child worker processes are killed on parent termination
   (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE).
2. Windows Named Mutex: Enforces single-instance desktop application per user session.
"""
from __future__ import annotations

import ctypes
from ctypes import Structure, c_size_t, c_ulonglong, c_void_p, wintypes
from typing import Optional

kernel32 = ctypes.windll.kernel32

# Job Object constants
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9


class IO_COUNTERS(Structure):
    _fields_ = [
        ("ReadOperationCount", c_ulonglong),
        ("WriteOperationCount", c_ulonglong),
        ("OtherOperationCount", c_ulonglong),
        ("ReadTransferCount", c_ulonglong),
        ("WriteTransferCount", c_ulonglong),
        ("OtherTransferCount", c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", c_size_t),
        ("MaximumWorkingSetSize", c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", c_size_t),
        ("JobMemoryLimit", c_size_t),
        ("PeakProcessMemoryLimit", c_size_t),
        ("PeakJobMemoryLimit", c_size_t),
    ]


class WindowsJobObject:
    """
    Manages a Windows Job Object. All child processes assigned to this job
    will be automatically terminated if the parent process dies or crashes.
    """

    def __init__(self, job_name: Optional[str] = None):
        self.handle = kernel32.CreateJobObjectW(None, job_name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        success = kernel32.SetInformationJobObject(
            self.handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not success:
            kernel32.CloseHandle(self.handle)
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_process(self, process_handle: int) -> bool:
        """Assigns an active process handle to the job object."""
        return bool(kernel32.AssignProcessToJobObject(self.handle, c_void_p(process_handle)))

    def close(self) -> None:
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> WindowsJobObject:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SingleInstanceMutex:
    """
    Enforces a single running instance of the application per Windows user session.
    """

    ERROR_ALREADY_EXISTS = 183

    def __init__(self, mutex_name: str = "Local\\HalotecContainerOCRDesktopMutex"):
        self.mutex_name = mutex_name
        self.handle = kernel32.CreateMutexW(None, wintypes.BOOL(True), mutex_name)
        self.already_running = kernel32.GetLastError() == self.ERROR_ALREADY_EXISTS

    def is_already_running(self) -> bool:
        return self.already_running

    def release(self) -> None:
        if self.handle:
            try:
                kernel32.ReleaseMutex(self.handle)
            except Exception:
                pass
            try:
                kernel32.CloseHandle(self.handle)
            except Exception:
                pass
            self.handle = None
