"""SIM808 UART test mode. No SIM is required for module readiness."""
import re
import threading
import time


class GSMConnection:
    def __init__(self, mode, config, lock_controller, progress=None, serial_factory=None):
        self.mode = mode
        self.config = config.get('gsm', {})
        self.lock_controller = lock_controller
        self.progress = progress or (lambda *_: None)
        self.serial_factory = serial_factory
        self.port = None
        self.stop_event = threading.Event()
        self.failure_event = threading.Event()
        self.ready_event = threading.Event()
        self.error = None
        self.recorder = None
        self._mutex = threading.RLock()
        self.worker = None
        self.message = 'Waiting for AT test'
        self.call_active = False

    def _command(self, command, timeout=None, expect=b'OK', clear=True):
        if clear:
            self.port.reset_input_buffer()
        self.port.write(command)
        deadline = time.monotonic() + float(timeout or self.config.get('command_timeout', 3))
        data = b''
        while time.monotonic() < deadline:
            data += self.port.read(256)
            lines = [line.strip() for line in data.splitlines()]
            if any(line == b'ERROR' or line.startswith((b'+CME ERROR:', b'+CMS ERROR:')) for line in lines):
                raise RuntimeError('SIM808: ' + data.decode('ascii', 'replace').strip())
            if (expect == b'>' and b'>' in data) or (expect != b'>' and expect in lines):
                return data
        raise RuntimeError('SIM808 response timed out; check power, TX/RX wiring and baud rate')

    def start(self):
        try:
            if (self.config.get('tx_pin', 14), self.config.get('rx_pin', 15)) != (14, 15):
                raise ValueError('GSM Mode requires BCM TX14/RX15; configure OS UART routing separately')
            factory = self.serial_factory
            if factory is None:
                import serial
                factory = serial.Serial
            device = self.config.get('serial_device', '/dev/ttyAMA0')
            self.progress('starting', f'Testing SIM808 AT on {device}')
            self.port = factory(port=device, baudrate=int(self.config.get('baud_rate', 9600)),
                                timeout=0.1, write_timeout=1, exclusive=True)
            # A repeated AT also accommodates modules using automatic baud detection.
            for attempt in range(3):
                try:
                    self._command(b'AT\r')
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
            self._command(b'ATE0\r')
            self.message = 'AT OK · Module active. SIM required for SMS and calls.'
            self.ready_event.set()
            self._on_ready()
            self.worker = threading.Thread(target=self._monitor, name='GSMMonitor', daemon=True)
            self.worker.start()
        except Exception as error:
            self.error = f'GSM startup failed: {error}'
            self.stop()
            raise RuntimeError(self.error) from error

    def _monitor(self):
        while not self.stop_event.wait(float(self.config.get('health_interval', 10))):
            try:
                with self._mutex:
                    if self.stop_event.is_set():
                        return
                    self._command(b'AT\r')
                    if self.call_active:
                        response = self._command(b'AT+CLCC\r')
                        self.call_active = b'+CLCC:' in response
            except Exception as error:
                self.error = f'GSM connection lost: {error}'
                self.failure_event.set()
                return

    def _require_network(self):
        response = self._command(b'AT+CPIN?\r')
        if b'+CPIN: READY' not in response:
            raise RuntimeError('SIM is missing, locked or not ready')
        response = self._command(b'AT+CREG?\r')
        if not re.search(rb'\+CREG:\s*\d+,\s*[15](?:\s|,|$)', response):
            raise RuntimeError('SIM is not registered on the GSM network')

    def action(self, action, phone=None, message=None, wait=False):
        if action not in {'sms', 'call', 'hangup'}:
            raise ValueError('Select SMS, call or hangup')
        if action != 'hangup' and (not isinstance(phone, str) or not re.fullmatch(r'\+?[0-9]{7,15}', phone)):
            raise ValueError('Enter a phone number with 7–15 digits and optional leading +')
        if action == 'sms' and (not isinstance(message, str) or not message.strip() or
                                len(message) > 160 or any(ord(c) < 32 or ord(c) > 126 for c in message)):
            raise ValueError('SMS must contain 1–160 printable English ASCII characters')
        if not self._mutex.acquire(blocking=wait):
            raise RuntimeError('GSM is busy; try again')
        try:
            if self.stop_event.is_set() or self.failure_event.is_set() or not self.ready_event.is_set():
                raise RuntimeError('GSM module is not active')
            if action == 'hangup':
                self._command(b'ATH\r')
                self.call_active = False
                self.message = 'Call ended'
            else:
                if self.call_active:
                    raise RuntimeError('End the current call first')
                self._require_network()
                if action == 'call':
                    self._command(f'ATD{phone};\r'.encode('ascii'))
                    self.call_active = True
                    self.message = f'Dial request accepted: {phone}'
                else:
                    self._command(b'AT+CMGF=1\r')
                    self._command(b'AT+CSCS="IRA"\r')
                    self._command(f'AT+CMGS="{phone}"\r'.encode('ascii'), expect=b'>')
                    try:
                        response = self._command(message.encode('ascii') + b'\x1a',
                            timeout=self.config.get('sms_timeout', 30), clear=False)
                        if b'+CMGS:' not in response:
                            raise RuntimeError('SIM808 did not confirm SMS submission')
                    except Exception:
                        self.port.write(b'\x1b')
                        raise
                    self.message = f'SMS submitted to network: {phone}'
            return self.status()['gsm']
        except Exception as error:
            self.message = str(error)
            raise RuntimeError(str(error)) from error
        finally:
            self._mutex.release()

    def status(self):
        return {'connected': False, 'fps': 0, 'detections': {'faces': [], 'weapons': []},
                'face_inference_ms': None, 'weapon_inference_ms': None,
                'gsm': {'connected': self.ready_event.is_set() and not self.stop_event.is_set() and not self.failure_event.is_set(),
                        'default_phone': self.config.get('default_phone', '+8801719473555'),
                        'device': self.config.get('serial_device', '/dev/ttyAMA0'),
                        'baud_rate': self.config.get('baud_rate', 9600),
                        'message': self.message, 'call_active': self.call_active}}

    def stop(self):
        self.stop_event.set()
        with self._mutex:
            if self.port is not None:
                try:
                    if self.call_active:
                        self._command(b'ATH\r')
                finally:
                    self.port.close()
                    self.port = None
                    self.call_active = False
        if self.worker and self.worker is not threading.current_thread():
            self.worker.join(timeout=5)


    def _on_ready(self):
        """Shared connections do not change the surveillance access policy."""


class GSMRuntime(GSMConnection):
    """Standalone module test mode."""

    def _on_ready(self):
        self.lock_controller.set_mode('gsm')
