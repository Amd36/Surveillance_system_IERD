import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_recording import RawRecorder


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.camera = MagicMock()
        self.recorder = RawRecorder(self.camera, Path(self.directory.name))
        self.validation_patch = patch.object(RawRecorder, '_validate_video')
        self.validation_patch.start()
        self.addCleanup(self.validation_patch.stop)
        self.encoder_patch = patch('picamera2.encoders.H264Encoder')
        self.output_patch = patch('picamera2.outputs.FfmpegOutput')
        self.encoder_patch.start()
        output = self.output_patch.start().return_value
        output.ffmpeg = None
        self.addCleanup(self.encoder_patch.stop)
        self.addCleanup(self.output_patch.stop)
        self.addCleanup(self.recorder.close)
        self.addCleanup(self.directory.cleanup)
        def finish(*args):
            self.recorder.path.write_bytes(b'fake encoded video')
        self.camera.stop_encoder.side_effect = finish

    def test_start_runs_on_persistent_worker(self):
        import threading
        launched = []
        self.camera.start_encoder.side_effect = lambda *args, **kwargs: launched.append(threading.current_thread())
        request = threading.Thread(target=lambda: self.recorder.action('start'))
        request.start()
        request.join(timeout=5)
        self.assertFalse(request.is_alive())
        self.assertEqual(launched, [self.recorder.worker])
        self.assertTrue(self.recorder.worker.is_alive())
        self.recorder.action('stop')
        self.recorder.close()
        self.assertFalse(self.recorder.worker.is_alive())

    def test_stop_requires_explicit_save(self):
        self.recorder.action('start')
        self.recorder.action('stop')
        self.assertEqual(self.recorder.status()['state'], 'review')
        self.assertFalse(list(Path(self.directory.name).glob('raw_*.mp4')))
        self.recorder.action('save')
        self.assertEqual(len(list(Path(self.directory.name).glob('raw_*.mp4'))), 1)
        self.assertFalse(list(Path(self.directory.name).glob('.raw-take-*')))

    def test_retake_discards_old_take_and_starts_new_one(self):
        self.recorder.action('start')
        old = self.recorder.path
        self.recorder.action('stop')
        self.recorder.action('retake')
        self.assertFalse(old.exists())
        self.assertEqual(self.recorder.status()['state'], 'recording')
        self.assertNotEqual(old, self.recorder.path)
        self.recorder.discard()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_cannot_save_while_recording(self):
        self.recorder.action('start')
        with self.assertRaises(RuntimeError):
            self.recorder.action('save')
        self.recorder.discard()

    def test_writer_failure_discards_take(self):
        self.recorder.action('start')
        self.recorder.failure.set()
        self.assertEqual(self.recorder.status()['state'], 'error')
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_output_failure_is_logged_and_shown(self):
        self.recorder.action('start')
        with self.assertLogs('raw_recording', level='ERROR') as logs:
            self.recorder._output_failed(BrokenPipeError('test writer failure'))
        self.assertIn('BrokenPipeError', logs.output[0])
        status = self.recorder.status()
        self.assertEqual(status['state'], 'error')
        self.assertIn('BrokenPipeError: test writer failure', status['message'])
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_invalid_video_cannot_be_saved(self):
        self.recorder.action('start')
        self.recorder._validate_video.side_effect = RuntimeError('invalid video')
        with self.assertRaises(RuntimeError):
            self.recorder.action('stop')
        self.assertEqual(self.recorder.status()['state'], 'error')
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_encoder_start_failure_cleans_up(self):
        self.camera.start_encoder.side_effect = RuntimeError('encoder unavailable')
        with self.assertRaises(RuntimeError):
            self.recorder.action('start')
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
