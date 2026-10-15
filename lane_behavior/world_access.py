"""Bounded simulator-port reservations shared with the collection supervisor.

A reservation prevents *new* collection dispatch. It does not steal the world
lock, interrupt an existing worker, or establish that its actors were removed.
Capture must still acquire world_owner(port, root) and inspect the empty world.
No reservations, processes or simulator connections are created on import.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
import uuid

import psutil

from lane_dataset.config import write_json
from lane_dataset.simulator import world_owner

PROTOCOL = 'pid_creation_expiry_v1'
MAX_DURATION_S = 3600.


class PortReservedError(RuntimeError):
    """Dispatch/acquisition must wait; this is not a failed collection job."""


@dataclass(frozen=True)
class Reservation:
    port: int
    owner_pid: int
    owner_create_time: float
    token: str
    issued_at: float
    expires_at: float
    purpose: str = 'bounded behavior qualification'
    protocol: str = PROTOCOL


@dataclass(frozen=True)
class ReservationStatus:
    blocked: bool
    status: str
    reason: str
    reservation: Reservation | None = None


def _port(port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('reservation port must be an integer TCP port')
    return port


def _path(root, port):
    return Path(root)/'reports/worker-reservations'/f'port-{_port(port)}.json'


def _identity(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('reservation owner PID must be positive integer')
    process = psutil.Process(pid)
    created = float(process.create_time())
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        raise psutil.NoSuchProcess(pid)
    if not math.isfinite(created):
        raise ValueError('nonfinite process creation time')
    return created


def reservation_status(root, port):
    """Read-only status. Malformed/unverifiable leases block only their port."""
    path = _path(root, port)
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return ReservationStatus(False, 'absent', '')
    except (OSError, ValueError) as error:
        return ReservationStatus(True, 'malformed', f'cannot read reservation: {type(error).__name__}: {error}')
    now = time.time()
    try:
        lease = Reservation(**document)
        if (lease.protocol != PROTOCOL or lease.port != port or type(lease.port) is not int
                or type(lease.owner_pid) is not int or lease.owner_pid <= 0
                or not isinstance(lease.token, str) or len(lease.token) != 32
                or any(c not in '0123456789abcdef' for c in lease.token)
                or not isinstance(lease.purpose, str) or not lease.purpose.strip()
                or not all(math.isfinite(x) for x in (lease.owner_create_time, lease.issued_at, lease.expires_at))
                or not 0 < lease.expires_at-lease.issued_at <= MAX_DURATION_S
                or lease.issued_at > now+5. or lease.owner_create_time <= 0):
            raise ValueError('invalid reservation fields or lifetime')
    except (TypeError, ValueError, AttributeError) as error:
        return ReservationStatus(True, 'malformed', f'invalid reservation: {error}')
    if lease.expires_at <= now:
        return ReservationStatus(False, 'expired', 'reservation expired', lease)
    try:
        actual = _identity(lease.owner_pid)
    except psutil.NoSuchProcess:
        return ReservationStatus(False, 'dead_owner', 'reservation owner exited', lease)
    except (psutil.AccessDenied, OSError, ValueError) as error:
        return ReservationStatus(True, 'owner_unverifiable', f'cannot verify reservation owner: {error}', lease)
    if abs(actual-lease.owner_create_time) > .001:
        return ReservationStatus(False, 'pid_reused', 'owner PID now identifies a different process', lease)
    return ReservationStatus(True, 'active', 'port reserved for '+lease.purpose, lease)


@contextmanager
def _reservation_lock(root, port):
    lock = world_owner('worker-reservation-'+str(_port(port)), root)
    try:
        lock.__enter__()
    except OSError as error:
        raise PortReservedError('reservation mutex is busy for port '+str(port)) from error
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _duration(value):
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= MAX_DURATION_S:
        raise ValueError('reservation duration must be within (0, 3600] seconds')
    return float(value)


def acquire_reservation(root, port, *, duration_s=1800., owner_pid=None,
                        purpose='bounded behavior qualification'):
    """Reserve future dispatch for a live exact PID identity; no force option.

    owner_pid permits a launcher to bind a known, already running child. The
    capture process must itself match that identity when using the reservation.
    """
    duration_s = _duration(duration_s)
    owner_pid = os.getpid() if owner_pid is None else owner_pid
    created = _identity(owner_pid)
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError('nonempty reservation purpose required')
    with _reservation_lock(root, port):
        current = reservation_status(root, port)
        if current.blocked:
            raise PortReservedError(f'port {port}: {current.status}: {current.reason}')
        now = time.time()
        lease = Reservation(_port(port), owner_pid, created, uuid.uuid4().hex, now, now+duration_s, purpose)
        write_json(_path(root, port), asdict(lease))
        return lease


def _same_owner(a, b):
    return (a is not None and a.port == b.port and a.token == b.token and a.owner_pid == b.owner_pid
        and abs(a.owner_create_time-b.owner_create_time) <= .001)


def renew_reservation(root, lease, *, duration_s=1800.):
    """Renew a still-active exact lease; expiration requires fresh acquisition."""
    duration_s = _duration(duration_s)
    with _reservation_lock(root, lease.port):
        current = reservation_status(root, lease.port)
        if current.status != 'active' or not _same_owner(current.reservation, lease):
            raise PortReservedError('cannot renew expired, changed, or differently owned reservation')
        now = time.time()
        renewed = Reservation(lease.port, lease.owner_pid, lease.owner_create_time, lease.token,
            now, now+duration_s, lease.purpose)
        write_json(_path(root, lease.port), asdict(renewed))
        return renewed


def release_reservation(root, lease):
    """Remove only this token/owner identity; another owner's lease is untouched."""
    with _reservation_lock(root, lease.port):
        current = reservation_status(root, lease.port)
        if current.status == 'absent':
            return False
        if not _same_owner(current.reservation, lease):
            raise PortReservedError('cannot release another or malformed reservation')
        try:
            actual = _identity(lease.owner_pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
            raise PortReservedError('reservation owner identity is no longer live') from error
        if abs(actual-lease.owner_create_time) > .001:
            raise PortReservedError('reservation owner PID was reused')
        _path(root, lease.port).unlink()
        return True


@contextmanager
def collection_dispatch(root, port):
    """Atomically check a lease and start one collection process under its mutex."""
    with _reservation_lock(root, port):
        current = reservation_status(root, port)
        if current.blocked:
            raise PortReservedError(f'port {port}: {current.status}: {current.reason}')
        yield


def protocol_identity():
    pid = os.getpid()
    return dict(protocol=PROTOCOL, pid=pid, create_time=_identity(pid))


def _active_capture_pids(root, port):
    found = []
    expected = Path(root).resolve()
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info.get('cmdline') or []
        if ('reports.automatic_corpus' not in command or '--job' not in command
                or '--root' not in command or '--port' not in command or '--reprocess' in command):
            continue
        try:
            if (int(command[command.index('--port')+1]) == port
                    and Path(command[command.index('--root')+1]).resolve() == expected):
                found.append(int(process.info['pid']))
        except (ValueError, IndexError, TypeError):
            continue
    return found


def _require_collector_protocol(root, port):
    status_path = Path(root)/'reports/automatic/status.json'
    if not status_path.exists():
        return
    try:
        status = json.loads(status_path.read_text(encoding='utf-8'))
        active = status.get('status') == 'running' and port in {int(s['port']) for s in status.get('configured_workers', [])}
        if not active:
            return
        pid = status.get('pid')
        try:
            created = _identity(pid)
        except psutil.NoSuchProcess:
            return
        state = json.loads((Path(root)/'reports/automatic/state.json').read_text(encoding='utf-8'))
        identity = state.get('worker_reservation_protocol', {})
        if (identity.get('protocol') != PROTOCOL or identity.get('pid') != pid
                or abs(identity.get('create_time', -1.)-created) > .001):
            raise RuntimeError('running collector has not advertised matching reservation support')
    except (OSError, ValueError, TypeError, psutil.AccessDenied) as error:
        raise RuntimeError('cannot verify collector reservation support: '+str(error)) from error


def require_reserved_or_idle(root, port, owner_pid=None, paused_supervisor_pid=None):
    """Authorize the caller's lease or existing idle/explicit-paused permission.

    This gate is necessary but not world ownership: acquire world_owner(port,
    root), call this gate again, and reject existing vehicle/walker/sensor actors
    before changing the world. Active capture jobs must drain first, including
    processes that have not acquired their world lock yet.
    """
    current = reservation_status(root, port)
    caller = os.getpid()
    if owner_pid is not None and owner_pid != caller:
        raise RuntimeError('reservation owner must be the current capture process; no owner override')
    if current.blocked:
        lease = current.reservation
        if current.status != 'active' or lease is None or lease.owner_pid != caller:
            raise RuntimeError(f'port {port} unavailable: {current.status}: {current.reason}')
        if abs(_identity(caller)-lease.owner_create_time) > .001:
            raise RuntimeError('capture process does not match reservation identity')
        _require_collector_protocol(root, port)
        workers = _active_capture_pids(root, port)
        if workers:
            raise RuntimeError('reserved port is still draining collection processes: '+','.join(map(str, workers)))
        return dict(access='reserved', owner_pid=caller, expires_at=lease.expires_at, token=lease.token,
            world_lock_and_empty_actor_check_required=True)
    from .qualification_probe import refuse_active_collection_port
    refuse_active_collection_port(root, port, paused_supervisor_pid)
    return dict(access='idle_or_explicitly_paused', world_lock_and_empty_actor_check_required=True)
