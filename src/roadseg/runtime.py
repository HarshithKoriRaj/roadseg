"""Process-level contracts shared by `roadseg.train` and `roadseg.eval`.

Kept free of torch so that the SLURM-facing behaviour (exit codes, ending the
process) can be tested in a process that starts in milliseconds.

Exit codes -- a contract with scripts/sbatch_train_array.sh, which decides from
them whether to requeue, score, or fail the task:

    0   EXIT_OK         done (or the run was already complete)
    1   EXIT_CRASHED    uncaught exception; the traceback is on stderr
    2   EXIT_REFUSED    declined to act on what is on disk: resume with a different
                        configuration, run directory locked by another process, code
                        changed since the checkpoint, test split already scored
                        (also argparse's usage error)
    3   EXIT_PREFLIGHT  the GPU is not usable for this run: `--device cuda` without
                        CUDA, or the preflight self-test failed (or an earlier task's
                        did: a PREFLIGHT_FAILED.* file is in the runs directory)
    75  EXIT_REQUEUE    interrupted by a signal, checkpoint saved, run me again

Only 75 is ever requeued. 1, 2 and 3 would fail the same way again.
"""
from __future__ import annotations

import errno
import fcntl
import multiprocessing
import os
import socket
import sys
import time
import traceback
from pathlib import Path

EXIT_OK = 0
EXIT_CRASHED = 1
EXIT_REFUSED = 2
EXIT_PREFLIGHT = 3
EXIT_REQUEUE = 75                     # EX_TEMPFAIL


class Refused(SystemExit):
    """Raised with a message when the process declines to act (exit code EXIT_REFUSED)."""


class PreflightFailed(SystemExit):
    """Raised with a message when the GPU cannot run this job (exit code EXIT_PREFLIGHT)."""


def tmp_name(path) -> Path:
    """`.<name>.<hostname>.<pid>.tmp` beside `path`: the temp file THIS process writes
    before renaming it to `path`. Unique per process, so two writers can never share an
    inode, and a leftover names the (dead) process it belonged to."""
    path = Path(path)
    return path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")


_HELD: set[str] = set()                # lock files this process holds (see RunLock.acquire)


class RunLock:
    """Exclusive lock on a run directory: `<run>/.lock`, held for the life of the process.

    POSIX record lock (`fcntl.lockf`), non-blocking. It works on NFS 4.1 (/scratch),
    belongs to the process -- forked DataLoader workers do not share it -- and the
    kernel drops it when the process ends in any way, SIGKILL included, so there is
    no stale lock to clean up. The file itself is never removed (unlinking a lock
    file lets two processes each lock a different inode of the same name).
    """

    def __init__(self, run) -> None:
        self.run = Path(run)
        self.path = self.run / ".lock"
        self.fd: int | None = None

    def acquire(self) -> "RunLock":
        """Take the lock or raise Refused (exit code 2) naming the run and the holder."""
        self.run.mkdir(parents=True, exist_ok=True)
        key = os.path.realpath(self.path)
        # POSIX locks never conflict inside one process, and closing ANY descriptor of the
        # file drops them all -- so a second holder in this process must be refused here.
        if key in _HELD:
            raise Refused(f"run directory {self.run} is locked by this very process")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            try:
                holder = os.pread(fd, 300, 0).decode(errors="replace").strip()
            except OSError:
                holder = ""
            os.close(fd)
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise Refused(f"run directory {self.run} is locked by another process "
                          f"({holder or 'holder unknown'}). Two processes writing one run "
                          f"would corrupt it; not starting.") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{socket.gethostname()} pid {os.getpid()} since "
                      f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n".encode(), 0)
        self.fd = fd
        _HELD.add(key)
        return self

    def release(self) -> None:
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        _HELD.discard(os.path.realpath(self.path))
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "RunLock":
        return self if self.fd is not None else self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def clear_stale_temps(run) -> list[str]:
    """Remove every `*.tmp` in `run`. ONLY call while holding the run's lock: then no
    live process is writing there, and each temp file is the debris of a killed one
    (never read by anything, but 300 MB apiece for a checkpoint)."""
    removed = []
    for p in Path(run).iterdir():
        if p.name.endswith(".tmp") and not p.is_dir():
            p.unlink(missing_ok=True)
            removed.append(p.name)
    return sorted(removed)


def kill_children(timeout: float = 10.0) -> int:
    """SIGKILL every live `multiprocessing` child of this process and reap it.

    DataLoader workers ignore SIGTERM (a job-wide preemption signal must not kill
    them mid-batch), and multiprocessing's exit handler ends daemonic children
    with SIGTERM and then joins them without a timeout: a process that still has
    such a child when the interpreter shuts down never exits. Returns how many
    were killed.
    """
    children = multiprocessing.active_children()
    for p in children:
        try:
            p.kill()
        except Exception:
            pass
    for p in children:
        try:
            p.join(timeout)
        except Exception:
            pass
    return len(children)


def exit_code_of(exc: BaseException) -> int:
    """The exit code `run_cli` gives a process whose main raised `exc` (message printed)."""
    if isinstance(exc, (Refused, PreflightFailed)):
        print(f"{'refused' if isinstance(exc, Refused) else 'preflight failed'}: {exc}",
              file=sys.stderr)
        return EXIT_REFUSED if isinstance(exc, Refused) else EXIT_PREFLIGHT
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None:
            return EXIT_OK
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)                       # SystemExit("message"): a failure
        return EXIT_CRASHED
    if isinstance(exc, KeyboardInterrupt):
        print("interrupted", file=sys.stderr)
        return 130
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    return EXIT_CRASHED


def run_cli(main, argv=None):
    """Run `main(argv)` as THE process and end the process -- whatever happened.

    Normal return, SystemExit and an uncaught exception all end in `os._exit` with
    the contract's code after the traceback is printed, the streams are flushed and
    every child process is killed. Nothing after `main` can therefore block the
    exit: not multiprocessing's exit handler, not a DataLoader destructor, not a
    non-daemon thread.
    """
    try:
        code = main(argv)
        code = EXIT_OK if code is None else int(code)
    except BaseException as exc:                           # noqa: BLE001 - this IS the top level
        try:
            code = exit_code_of(exc)
        except BaseException:                              # e.g. stderr is a closed pipe
            code = EXIT_CRASHED
    try:
        kill_children()
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(code)
