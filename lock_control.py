"""GPIO output driver for the door lock and lock-state indicator LED."""

import threading
import time


try:
    import RPi.GPIO as GPIO
    GPIO_IMPORTED = True
except Exception:
    GPIO = None
    GPIO_IMPORTED = False


LOCK_PIN = 23
INDICATOR_LED_PIN = 24
GPIO_AVAILABLE = False

_lock_active_low = True
_led_on_when_locked = True
_configured = False
_last_error = None
_commanded_state = "unknown"
_gpio_mutex = threading.Lock()


def configure(
    lock_pin: int,
    indicator_led_pin: int,
    lock_active_low: bool = True,
    indicator_led_on_when_locked: bool = True,
) -> bool:
    """Configure BCM GPIO pins and start in the fail-secure locked state."""
    global LOCK_PIN, INDICATOR_LED_PIN, GPIO_AVAILABLE
    global _lock_active_low, _led_on_when_locked, _configured, _last_error
    global _commanded_state

    with _gpio_mutex:
        LOCK_PIN = int(lock_pin)
        INDICATOR_LED_PIN = int(indicator_led_pin)
        if LOCK_PIN == INDICATOR_LED_PIN:
            _last_error = "Lock and indicator LED must use different GPIO pins"
            GPIO_AVAILABLE = False
            _configured = False
            _commanded_state = "unknown"
            return False
        _lock_active_low = bool(lock_active_low)
        _led_on_when_locked = bool(indicator_led_on_when_locked)

        if not GPIO_IMPORTED:
            _last_error = "RPi.GPIO is not installed"
            GPIO_AVAILABLE = False
            _commanded_state = "unknown"
            return False

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(LOCK_PIN, GPIO.OUT, initial=_lock_level(locked=True))
            GPIO.setup(
                INDICATOR_LED_PIN,
                GPIO.OUT,
                initial=_led_level(locked=True),
            )
            _configured = True
            GPIO_AVAILABLE = True
            _last_error = None
            _commanded_state = "locked"
            return True
        except Exception as error:
            _last_error = str(error)
            _configured = False
            GPIO_AVAILABLE = False
            _commanded_state = "unknown"
            print(f"[lock_control] GPIO initialization failed: {error}")
            return False


def _lock_level(locked: bool):
    active = GPIO.LOW if _lock_active_low else GPIO.HIGH
    inactive = GPIO.HIGH if _lock_active_low else GPIO.LOW
    return active if locked else inactive


def _led_level(locked: bool):
    illuminated = locked if _led_on_when_locked else not locked
    return GPIO.HIGH if illuminated else GPIO.LOW


def _set_outputs(locked: bool) -> bool:
    global GPIO_AVAILABLE, _configured, _last_error, _commanded_state
    with _gpio_mutex:
        if not _configured or not GPIO_AVAILABLE:
            return False
        try:
            GPIO.output(LOCK_PIN, _lock_level(locked))
            GPIO.output(INDICATOR_LED_PIN, _led_level(locked))
            _commanded_state = "locked" if locked else "unlocked"
            return True
        except Exception as error:
            _last_error = str(error)
            GPIO_AVAILABLE = False
            _configured = False
            _commanded_state = "unknown"
            print(f"[lock_control] GPIO output failed: {error}")
            return False


def lock_on() -> bool:
    """Engage the lock and update its indicator LED."""
    return _set_outputs(locked=True)


def lock_off() -> bool:
    """Release the lock and update its indicator LED."""
    return _set_outputs(locked=False)


def status() -> dict:
    with _gpio_mutex:
        return {
            "available": GPIO_AVAILABLE,
            "configured": _configured,
            "lock_pin": LOCK_PIN,
            "indicator_led_pin": INDICATOR_LED_PIN,
            "state": _commanded_state,
            "error": _last_error,
        }


def cleanup() -> None:
    """Release configured GPIO channels during an explicit hardware shutdown."""
    global GPIO_AVAILABLE, _configured, _commanded_state
    with _gpio_mutex:
        if _configured and GPIO_IMPORTED:
            GPIO.cleanup((LOCK_PIN, INDICATOR_LED_PIN))
        GPIO_AVAILABLE = False
        _configured = False
        _commanded_state = "unknown"


def blink(delay: float = 1.0) -> None:
    """Manual lock-and-indicator output test."""
    if not configure(LOCK_PIN, INDICATOR_LED_PIN):
        raise RuntimeError(status()["error"])
    try:
        while True:
            lock_on()
            time.sleep(delay)
            lock_off()
            time.sleep(delay)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    blink()
