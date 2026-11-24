"""Docker-free MySQL backend for LifelongAgentBench DB tasks.

Drop-in replacement for the reference ``DBBenchContainer`` on hosts without a
Docker daemon.  Each instance runs its own native ``mysqld`` (separate datadir,
port and socket) copied from a template datadir that is initialized once, so
tasks stay isolated exactly as with one ``mysql:8.0`` container per task.
``execute`` is inherited unchanged from the reference class.

Enable with ``LIFELONG_DB_BACKEND=local``.  ``LIFELONG_MYSQL_ROOT`` sets the
working directory (default ``/var/tmp/lifelong_mysql``) and ``MYSQLD_BIN`` the
server binary (default ``mysqld`` on PATH).
"""

from __future__ import annotations

import fcntl
import os
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import mysql.connector

_ROOT = Path(os.environ.get("LIFELONG_MYSQL_ROOT", "/var/tmp/lifelong_mysql"))
_MYSQLD = os.environ.get("MYSQLD_BIN") or shutil.which("mysqld") or "/usr/sbin/mysqld"
_TEMPLATE = _ROOT / "template"
_template_lock = threading.Lock()

# Server options shared by template initialization and every instance.  They
# mirror the official mysql:8.0 image config (skip-name-resolve, no X plugin)
# and shrink the redo log so per-task datadir copies stay small.
_COMMON_ARGS = [
    "--no-defaults",
    "--user=root",
    "--skip-name-resolve",
    "--mysqlx=OFF",
    "--innodb-redo-log-capacity=33554432",
    # Each instance's native AIO reserves kernel aio contexts; dozens of
    # concurrent instances exhaust fs.aio-max-nr (65536) and mysqld then
    # fails with "io_setup() failed with EAGAIN". Tiny test databases don't
    # need native AIO.
    "--innodb-use-native-aio=0",
]


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _ensure_template(password: str) -> Path:
    """Initialize the template datadir once (thread- and process-safe)."""

    with _template_lock:
        _ROOT.mkdir(parents=True, exist_ok=True)
        with open(_ROOT / "template.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if (_TEMPLATE / ".ready").is_file():
                return _TEMPLATE
            shutil.rmtree(_TEMPLATE, ignore_errors=True)
            _TEMPLATE.mkdir(parents=True)
            init_sql = _ROOT / "template_init.sql"
            # Same accounts as the docker image entrypoint with MYSQL_ROOT_PASSWORD.
            init_sql.write_text(
                f"ALTER USER 'root'@'localhost' IDENTIFIED BY '{password}';\n"
                f"CREATE USER 'root'@'%' IDENTIFIED BY '{password}';\n"
                "GRANT ALL ON *.* TO 'root'@'%' WITH GRANT OPTION;\n"
                "FLUSH PRIVILEGES;\n"
            )
            subprocess.run(
                [_MYSQLD, *_COMMON_ARGS, "--initialize-insecure",
                 f"--datadir={_TEMPLATE / 'data'}", f"--init-file={init_sql}",
                 f"--log-error={_ROOT / 'template_init.err'}"],
                check=True,
            )
            (_TEMPLATE / ".ready").write_text("ok\n")
            return _TEMPLATE


def make_local_container_class(base_cls):
    """Subclass the reference DBBenchContainer with a native mysqld lifecycle."""

    class LocalMySQLContainer(base_cls):
        def __init__(self, image: str = "mysql"):  # noqa: ARG002 - image is ignored
            self.deleted = False
            self.image = image
            self._process = None
            self._instance_dir = None
            template = _ensure_template(self.password)

            port = base_cls.port + random.randint(0, 10000)
            while not _port_free(port):
                port += random.randint(1, 20)
            self.port = port

            self._instance_dir = Path(tempfile.mkdtemp(prefix=f"mysql_{port}_", dir=_ROOT))
            datadir = self._instance_dir / "data"
            shutil.copytree(template / "data", datadir, symlinks=True)
            self._process = subprocess.Popen(
                [_MYSQLD, *_COMMON_ARGS,
                 f"--datadir={datadir}",
                 f"--port={port}",
                 "--bind-address=127.0.0.1",
                 f"--socket={self._instance_dir / 'mysqld.sock'}",
                 f"--pid-file={self._instance_dir / 'mysqld.pid'}",
                 f"--secure-file-priv={self._instance_dir}",
                 f"--log-error={self._instance_dir / 'error.log'}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            deadline = time.monotonic() + 120
            while True:
                if self._process.poll() is not None:
                    log = (self._instance_dir / "error.log").read_text(errors="replace")[-2000:]
                    raise RuntimeError(f"local mysqld exited early on port {port}:\n{log}")
                try:
                    self.conn = mysql.connector.connect(
                        host="127.0.0.1",
                        user="root",
                        password=self.password,
                        port=self.port,
                        pool_reset_session=True,
                    )
                    break
                except mysql.connector.errors.Error:
                    if time.monotonic() > deadline:
                        self.delete()
                        raise
                    time.sleep(0.5)

        def delete(self) -> None:
            self.deleted = True
            try:
                self.conn.close()
            except Exception:  # noqa
                pass
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            if self._instance_dir is not None:
                shutil.rmtree(self._instance_dir, ignore_errors=True)

    return LocalMySQLContainer
