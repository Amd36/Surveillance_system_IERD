"""Bounded, asynchronous weapon notifications for Full System only."""
import logging
import queue
import threading
import time

from gsm_control import GSMConnection

logger = logging.getLogger(__name__)


class WeaponAlertWorker:
    def __init__(self, config, connection_factory=GSMConnection, clock=time.monotonic):
        self.config = config.get('gsm', {}).get('alerts', {})
        self.phone = config.get('gsm', {}).get('default_phone', '+8801719473555')
        self.enabled = bool(self.config.get('enabled', True))
        self.connection = connection_factory('full', config, None)
        self.clock = clock
        self.queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.mutex = threading.Lock()
        self.worker = None
        self.armed = True
        self.clear_since = None
        self.last_scan = None
        self.scan_max_age = float(config.get('weapon_detection', {}).get('result_max_age', 4))
        self.last_attempt = float('-inf')
        self.pending = False
        self.state = 'starting' if self.enabled else 'disabled'
        self.message = 'Preparing GSM alerts' if self.enabled else 'GSM alerts disabled'
        self.last_classes = []
        self.last_result = None

    def _set(self, state, message):
        with self.mutex:
            changed = (self.state, self.message) != (state, message)
            self.state, self.message = state, message
        if changed and state == 'error':
            logger.warning('GSM alert: %s', message)

    def start(self):
        if self.enabled:
            self.worker = threading.Thread(target=self._run, name='WeaponAlerts', daemon=True)
            self.worker.start()

    def observe(self, classes):
        if not self.enabled or self.stop_event.is_set():
            return
        now = self.clock()
        with self.mutex:
            if self.last_scan is not None and now - self.last_scan > self.scan_max_age:
                self.clear_since = None
            self.last_scan = now
            if not classes:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= float(self.config.get('clear_duration', 5)):
                    self.armed = True
                return
            self.clear_since = None
            if not self.armed or self.pending or now - self.last_attempt < float(self.config.get('cooldown', 60)):
                return
            self.armed = False
            self.pending = True
            self.last_attempt = now
            self.last_classes = list(dict.fromkeys(classes))
            self.queue.put_nowait(self.last_classes)

    def _run(self):
        try:
            try:
                self.connection.start()
                self._set('ready', 'AT OK; waiting for weapon detection')
            except Exception as error:
                self._set('error', str(error))
            while not self.stop_event.is_set():
                try:
                    classes = self.queue.get(timeout=.1)
                except queue.Empty:
                    if self.connection.failure_event.is_set():
                        self._set('error', self.connection.error or 'GSM connection lost')
                    continue
                try:
                    if not self.stop_event.is_set():
                        self._send(classes)
                finally:
                    with self.mutex:
                        self.pending = False
                    self.queue.task_done()
        finally:
            try:
                self.connection.stop()
            except Exception as error:
                self._set('error', f'GSM cleanup failed: {error}')

    def _send(self, classes):
        self._set('sending', 'Weapon detected: ' + ', '.join(classes))
        results, errors = [], []
        if self.config.get('sms_enabled', True) and not self.stop_event.is_set():
            try:
                self.connection.action('sms', self.phone, self.config.get('message',
                    'ALERT: A weapon has been detected by the surveillance system.'), wait=True)
                results.append('SMS submitted to network')
            except Exception as error:
                errors.append(f'SMS failed: {error}')
        if self.config.get('call_enabled', False) and not self.stop_event.is_set():
            try:
                self.connection.action('call', self.phone, wait=True)
                results.append('Call dialed')
                self.stop_event.wait(float(self.config.get('call_duration', 20)))
            except Exception as error:
                errors.append(f'Call failed: {error}')
            finally:
                if self.connection.call_active:
                    try:
                        self.connection.action('hangup', wait=True)
                    except Exception as error:
                        errors.append(f'Hangup failed: {error}')
        message = '; '.join(results + errors) or 'SMS and calling are disabled'
        with self.mutex:
            self.last_result = message
        self._set('error' if errors else 'sent', message)

    def status(self):
        with self.mutex:
            return {'state': self.state, 'message': self.message, 'pending': self.pending,
                    'last_classes': list(self.last_classes), 'last_result': self.last_result,
                    'call_enabled': bool(self.config.get('call_enabled', False))}

    def stop(self):
        self.stop_event.set()
        if self.worker:
            # Wait for any bounded modem operation and release UART before switching modes.
            timeout = max(60, float(self.connection.config.get('sms_timeout', 30)) +
                          10 * float(self.connection.config.get('command_timeout', 3)) + 10)
            self.worker.join(timeout=timeout)
            if self.worker.is_alive():
                raise RuntimeError('GSM alert worker did not stop; UART cannot be reassigned')
