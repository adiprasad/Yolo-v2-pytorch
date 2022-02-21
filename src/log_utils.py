from contextlib import contextmanager
import ctypes
import io
import os, sys
import platform
import tempfile

libc = ctypes.CDLL(None)

OUT_POINTER_DICT = {
    "Linux": {
        'stdout': 'stdout',
        'stderr': 'stderr'
    },
    "Darwin": {
        'stdout': '__stdoutp',
        'stderr': '__stderrp'
    }
}

platform_name = platform.system()

c_stdout = ctypes.c_void_p.in_dll(libc, OUT_POINTER_DICT[platform_name]['stdout'])
c_stderr = ctypes.c_void_p.in_dll(libc, OUT_POINTER_DICT[platform_name]['stderr'])


@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stdout(to_stdout_fd, to_stderr_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)

        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.close()

        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_stdout_fd, original_stdout_fd)
        # Make original_stderr_fd point to the same file as to_fd
        os.dup2(to_stderr_fd, original_stderr_fd)


        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))
        # Create a new sys.stderr that points to the redirected fd
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    # Save a copy of the original stderr fd in saved_stdout_fd
    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno(), tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd, saved_stderr_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)