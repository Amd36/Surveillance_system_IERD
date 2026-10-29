import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_config import create_camera, load_system_config
from libcamera import Transform


class CameraConfigTests(unittest.TestCase):
    @patch('picamera2.Picamera2')
    def test_shared_config_from_another_directory(self, constructor):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                settings = load_system_config()['camera']
                camera = create_camera()
            finally:
                os.chdir(previous)
        options = camera.create_preview_configuration.call_args.kwargs
        self.assertEqual(options['main']['size'], (settings['width'], settings['height']))
        self.assertEqual(options['main']['format'], settings['format'])
        self.assertEqual(options['transform'], Transform(hflip=True, vflip=True))
        camera.configure.assert_called_once_with(camera.create_preview_configuration.return_value)
        camera.start.assert_not_called()

    @patch('picamera2.Picamera2')
    def test_original_orientation_and_custom_size(self, constructor):
        camera = create_camera({'camera': {
            'width': 640, 'height': 480, 'hflip': False, 'vflip': False,
        }})
        options = camera.create_preview_configuration.call_args.kwargs
        self.assertEqual(options['main']['size'], (640, 480))
        self.assertEqual(options['transform'], Transform())

    @patch('picamera2.Picamera2')
    def test_quoted_boolean_is_rejected(self, constructor):
        with self.assertRaises(ValueError):
            create_camera({'camera': {'hflip': 'false'}})
        constructor.assert_not_called()

    @patch('picamera2.Picamera2')
    def test_configuration_failure_closes_camera(self, constructor):
        constructor.return_value.configure.side_effect = RuntimeError('configuration failed')
        with self.assertRaisesRegex(RuntimeError, 'configuration failed'):
            create_camera({'camera': {}})
        constructor.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
