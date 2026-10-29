import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsm_control import GSMRuntime


class Port:
    def __init__(self, missing=False, dead=False):
        self.commands = []
        self.data = b''
        self.closed = False
        self.missing = missing
        self.dead = dead

    def reset_input_buffer(self):
        self.data = b''

    def write(self, command):
        self.commands.append(command)
        if self.dead:
            self.data = b'NOT OK\r\n'
        elif command == b'AT+CPIN?\r':
            self.data = b'+CME ERROR: SIM not inserted\r\n' if self.missing else b'+CPIN: READY\r\nOK\r\n'
        elif command == b'AT+CREG?\r':
            self.data = b'+CREG: 0,1\r\nOK\r\n'
        elif command.startswith(b'AT+CMGS='):
            self.data = b'> '
        elif command.endswith(b'\x1a'):
            self.data = b'+CMGS: 42\r\nOK\r\n'
        else:
            self.data = b'OK\r\n'

    def read(self, size):
        data, self.data = self.data, b''
        return data

    def close(self):
        self.closed = True


class GSMTests(unittest.TestCase):
    def runtime(self, **kwargs):
        port = Port(**kwargs)
        runtime = GSMRuntime('gsm', {'gsm': {'command_timeout': .01}}, Mock(), serial_factory=lambda **_: port)
        self.addCleanup(runtime.stop)
        return runtime, port

    def test_ready_without_sim(self):
        runtime, port = self.runtime(missing=True)
        runtime.start()
        self.assertTrue(runtime.status()['gsm']['connected'])
        with self.assertRaisesRegex(RuntimeError, 'SIM not inserted'):
            runtime.action('sms', '+8801719473555', 'Test')
        self.assertFalse(runtime.failure_event.is_set())

    def test_sms_submission_and_call_cleanup(self):
        runtime, port = self.runtime()
        runtime.start()
        runtime.action('sms', '+8801719473555', 'Test')
        self.assertIn(b'Test\x1a', port.commands)
        runtime.action('call', '+8801719473555')
        self.assertTrue(runtime.call_active)
        runtime.stop()
        self.assertIn(b'ATH\r', port.commands)
        self.assertTrue(port.closed)

    def test_exact_ok_required_and_port_closed(self):
        runtime, port = self.runtime(dead=True)
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            runtime.start()
        self.assertTrue(port.closed)

    def test_injection_and_non_english_rejected(self):
        runtime, port = self.runtime()
        runtime.start()
        for action, phone, message in [('call', '1234567;ATH', ''), ('sms', '1234567', 'বাংলা'), ('sms', '1234567', 'x\x1a')]:
            with self.assertRaises(ValueError):
                runtime.action(action, phone, message)
        self.assertFalse(any(c.startswith(b'ATD') for c in port.commands))

    def test_open_failure(self):
        runtime = GSMRuntime('gsm', {}, Mock(), serial_factory=Mock(side_effect=PermissionError('denied')))
        with self.assertRaisesRegex(RuntimeError, 'denied'):
            runtime.start()

    def test_connection_loss(self):
        runtime, port = self.runtime()
        runtime.start()
        port.dead = True
        runtime.config['health_interval'] = .001
        # Execute monitor synchronously with a short health interval.
        runtime.stop_event.set()
        runtime.worker.join(1)
        runtime.stop_event.clear()
        runtime._monitor()
        self.assertTrue(runtime.failure_event.is_set())
        self.assertFalse(runtime.status()['gsm']['connected'])



class ModeIntegrationTests(unittest.TestCase):
    def test_gsm_transition_and_cleanup(self):
        from unittest.mock import patch
        import surveillance_runtime as runtime_module
        port = Port()
        def create(mode, config, lock, progress):
            return GSMRuntime(mode, config, lock, progress, serial_factory=lambda **_: port)
        with patch.object(runtime_module, 'AccessPolicyController') as lock, \
             patch.object(runtime_module, 'GSMRuntime', side_effect=create):
            controller = runtime_module.ModeController(initial_mode='gsm')
            self.addCleanup(controller.shutdown)
            controller._transition_thread.join(2)
            status = controller.status()
            self.assertEqual(status['state'], 'ready')
            self.assertEqual(status['active'], 'gsm')
            self.assertTrue(status['gsm']['connected'])
            self.assertFalse(status['camera']['connected'])
            controller.gsm_action('call', '+8801719473555', None)
            controller.shutdown()
            self.assertTrue(port.closed)
            self.assertIn(b'ATH\r', port.commands)

    def test_start_failure_reaches_mode_error(self):
        from unittest.mock import patch
        import surveillance_runtime as runtime_module
        def create(mode, config, lock, progress):
            return GSMRuntime(mode, config, lock, progress,
                              serial_factory=Mock(side_effect=FileNotFoundError('UART missing')))
        with patch.object(runtime_module, 'AccessPolicyController'), \
             patch.object(runtime_module, 'GSMRuntime', side_effect=create):
            controller = runtime_module.ModeController(initial_mode='gsm')
            self.addCleanup(controller.shutdown)
            controller._transition_thread.join(2)
            self.assertEqual(controller.status()['state'], 'error')
            self.assertIn('UART missing', controller.status()['error'])


if __name__ == '__main__':
    unittest.main()
