import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsm_alerts import WeaponAlertWorker
from gsm_control import GSMConnection


class AlertTests(unittest.TestCase):
    def make(self, **settings):
        self.now = 0
        self.connection = Mock()
        self.connection.config = {}
        self.connection.failure_event = threading.Event()
        self.connection.call_active = False
        self.alert = WeaponAlertWorker({'gsm': {'default_phone': '+8801719473555', 'alerts': settings}},
            connection_factory=lambda *_: self.connection, clock=lambda: self.now)
        self.addCleanup(self.alert.stop)
        return self.alert

    def test_every_class_queues_once_per_episode(self):
        for name in ['pistol', 'rifle', 'knife', 'future-class']:
            alert = self.make()
            alert.observe([name])
            alert.observe([name])
            self.now = 120
            alert.observe([name])
            self.assertEqual(alert.queue.qsize(), 1)
            self.assertEqual(alert.queue.get_nowait(), [name])

    def test_rearm_requires_clear_scans_and_cooldown(self):
        alert = self.make()
        alert.observe(['rifle'])
        alert.queue.get_nowait()
        alert.pending = False
        for t in range(1, 7):
            self.now = t
            alert.observe([])
        self.now = 7
        alert.observe(['knife'])
        self.assertTrue(alert.queue.empty())
        self.now = 60
        alert.observe(['knife'])
        self.assertEqual(alert.queue.get_nowait(), ['knife'])

    def test_stale_scans_do_not_rearm(self):
        alert = self.make()
        alert.observe(['rifle'])
        self.now = 1
        alert.observe([])
        self.now = 100
        alert.observe([])
        self.assertFalse(alert.armed)

    def test_sms_default_and_call_disabled(self):
        alert = self.make(message='Configured alert')
        alert._send(['rifle'])
        self.connection.action.assert_called_once_with('sms', '+8801719473555', 'Configured alert', wait=True)
        self.assertEqual(alert.status()['state'], 'sent')

    def test_call_attempted_after_sms_failure_and_hangs_up(self):
        alert = self.make(call_enabled=True, call_duration=0)
        def action(kind, *args, **kwargs):
            if kind == 'sms':
                raise RuntimeError('SIM missing')
            if kind == 'call':
                self.connection.call_active = True
            if kind == 'hangup':
                self.connection.call_active = False
        self.connection.action.side_effect = action
        alert._send(['knife'])
        self.assertEqual([c.args[0] for c in self.connection.action.call_args_list], ['sms', 'call', 'hangup'])
        self.assertEqual(alert.status()['state'], 'error')
        self.assertIn('SIM missing', alert.status()['message'])

    def test_observation_does_not_block_on_sms_and_stop_closes(self):
        alert = self.make()
        entered, release = threading.Event(), threading.Event()
        def action(*args, **kwargs):
            entered.set()
            release.wait(2)
        self.connection.action.side_effect = action
        alert.start()
        alert.observe(['pistol'])
        self.assertTrue(entered.wait(1))
        alert.observe(['rifle'])
        self.assertTrue(alert.status()['pending'])
        release.set()
        alert.stop()
        self.assertFalse(alert.worker.is_alive())
        self.connection.stop.assert_called_once()

    def test_gsm_start_failure_does_not_escape_worker(self):
        alert = self.make()
        self.connection.start.side_effect = RuntimeError('UART missing')
        alert.start()
        alert.stop()
        self.assertEqual(alert.status()['state'], 'error')
        self.assertIn('UART missing', alert.status()['message'])

    def test_disabled_does_not_open_uart(self):
        alert = self.make(enabled=False)
        alert.start()
        alert.observe(['rifle'])
        self.connection.start.assert_not_called()
        self.assertTrue(alert.queue.empty())

    def test_shared_connection_does_not_change_policy(self):
        from test_gsm_control import Port
        lock = Mock()
        connection = GSMConnection('full', {}, lock, serial_factory=lambda **_: Port())
        try:
            connection.start()
            lock.set_mode.assert_not_called()
        finally:
            connection.stop()

    def test_all_classes_engage_lock_hold(self):
        from surveillance_runtime import AccessPolicyController
        with patch('lock_control.configure'), patch('lock_control.status', return_value={'available': True}), \
             patch('lock_control.lock_on', return_value=True) as lock:
            policy = AccessPolicyController({})
            try:
                for name in ['pistol', 'rifle', 'knife', 'future-class']:
                    policy.observe_weapons([name])
                    self.assertEqual(policy.policy_state, 'weapon_hold')
                    self.assertIn(name, policy.reason)
                self.assertEqual(lock.call_count, 4)
            finally:
                policy.shutdown()

if __name__ == '__main__':
    unittest.main()
