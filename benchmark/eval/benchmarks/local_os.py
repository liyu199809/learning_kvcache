"""Docker-free sandbox for LifelongAgentBench OS-interaction tasks.

Drop-in replacement for the reference ``OSInteractionContainer`` on hosts
without a Docker daemon (or without CAP_SYS_ADMIN).  Each task gets a private
copy of a prepared Ubuntu 22.04 rootfs (the ``local-os/default`` image
contents).  A holder process owns a user namespace with an identity uid/gid map
(0-65535, so uid 0 is real root and useradd/chown work) plus a mount namespace
in which the host ``/proc`` is bind-mounted into the rootfs.  Every command
enters those namespaces with ``nsenter`` and runs under ``chroot`` -- the
analogue of ``docker exec``.  ``execute_independent`` is inherited unchanged;
only the container lifecycle and the exec primitive are replaced.

Known gap versus Docker: there is no private pid namespace (a fresh procfs
cannot be mounted under masked container /proc), so ``ps`` also lists
processes outside the sandbox.  ``/proc/<pid>``, ``/proc/self`` and ``<(...)``
work, and background jobs outlive the command that started them.

Enable with ``LIFELONG_OS_BACKEND=local``.  ``LIFELONG_OS_ROOTFS`` points at the
prepared rootfs (default ``/var/tmp/lifelong_os/rootfs``) and
``LIFELONG_OS_WORKDIR`` at the per-task copies (default
``/var/tmp/lifelong_os/instances``).
"""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

_ROOTFS = Path(os.environ.get("LIFELONG_OS_ROOTFS", "/var/tmp/lifelong_os/rootfs"))
_WORKDIR = Path(os.environ.get("LIFELONG_OS_WORKDIR", "/var/tmp/lifelong_os/instances"))

# Environment seen by `docker exec` in a container started with tty=True from
# ubuntu:22.04 (image ENV PATH, plus HOSTNAME/TERM/HOME set by the runtime).
_BASE_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "TERM": "xterm",
    "HOME": "/root",
}


def _processes_rooted_at(root: Path) -> list[int]:
    """PIDs whose filesystem root is ``root`` (covers daemons that called setsid)."""

    pids = []
    target = str(root)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.readlink(entry / "root") == target:
                pids.append(int(entry.name))
        except OSError:
            continue
    return pids


def make_local_container_class(base_cls, result_cls):
    """Subclass the reference OSInteractionContainer with a chroot sandbox."""

    class LocalOSContainer(base_cls):
        def __init__(self, command_execution_timeout: int, image: str = "local-os/default"):  # noqa: ARG002
            if not (_ROOTFS / "usr" / "bin" / "bash").is_file():
                raise FileNotFoundError(f"prepared rootfs not found at {_ROOTFS}")
            self.timeout_sec = command_execution_timeout
            _WORKDIR.mkdir(parents=True, exist_ok=True)
            self._instance = Path(tempfile.mkdtemp(prefix="os_", dir=_WORKDIR))
            self._root = self._instance / "rootfs"
            subprocess.run(["cp", "-a", str(_ROOTFS), str(self._root)], check=True)
            self._env = dict(_BASE_ENV, HOSTNAME=secrets.token_hex(6))
            self._log_dir = self._instance / "exec"
            self._log_dir.mkdir()
            self._counter = 0
            self._lock = threading.Lock()
            self._holder = None
            try:
                self._start_holder()
            except Exception:
                self.terminate()
                raise

        def _start_holder(self) -> None:
            """Create the user+mount namespaces that every command enters."""

            gate = self._instance / "gate"
            ready = self._instance / "ready"
            os.mkfifo(gate)
            self._holder = subprocess.Popen(
                ["unshare", "-U", "-m", "--propagation", "private", "bash", "-c",
                 f'read _ < "{gate}"; mount --rbind /proc "{self._root}/proc" '
                 f'&& touch "{ready}" && exec sleep infinity'],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            pid = self._holder.pid
            own_ns = os.readlink("/proc/self/ns/user")
            for _ in range(200):
                try:
                    if os.readlink(f"/proc/{pid}/ns/user") != own_ns:
                        break
                except OSError:
                    pass
                if self._holder.poll() is not None:
                    raise RuntimeError("namespace holder exited before unshare")
                threading.Event().wait(0.01)
            else:
                raise RuntimeError("namespace holder did not enter a new user namespace")
            # Identity map: uid/gid 0 is host root, as inside a Docker container.
            Path(f"/proc/{pid}/uid_map").write_text("0 0 65536\n")
            Path(f"/proc/{pid}/gid_map").write_text("0 0 65536\n")
            Path(gate).write_text("go\n")
            for _ in range(1000):
                if ready.exists():
                    return
                if self._holder.poll() is not None:
                    break
                threading.Event().wait(0.01)
            raise RuntimeError("namespace holder failed to mount /proc")

        def terminate(self) -> None:
            if self._holder is not None and self._holder.poll() is None:
                self._holder.kill()
                self._holder.wait(timeout=10)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                pids = _processes_rooted_at(self._root)
                if not pids:
                    break
                for pid in pids:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
                if sig == signal.SIGTERM:
                    threading.Event().wait(0.5)
            shutil.rmtree(self._instance, ignore_errors=True)

        def _execute_with_timeout(self, cmd: list[str]):
            with self._lock:
                self._counter += 1
                log_path = self._log_dir / f"{self._counter}.out"
            # Output goes to a file rather than a pipe: like `docker exec`, the
            # call returns when the command exits even if it left background
            # children holding the stream open.
            with open(log_path, "wb") as log:
                process = subprocess.Popen(
                    ["nsenter", "-t", str(self._holder.pid), "-U", "-m", "--",
                     "chroot", str(self._root), "/usr/bin/env", "-i",
                     *(f"{k}={v}" for k, v in self._env.items()), *cmd],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd="/",
                )
            try:
                exit_code = process.wait(timeout=self.timeout_sec)
            except subprocess.TimeoutExpired:
                # Same contract as the reference: report a timeout and leave the
                # command running; terminate() reaps it with the sandbox.
                return result_cls(exit_code=None, output=None, timeout_flag=True)
            return result_cls(
                exit_code=exit_code,
                output=log_path.read_bytes().decode("utf-8"),
                timeout_flag=False,
            )

    return LocalOSContainer
