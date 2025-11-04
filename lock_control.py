import time

# Try to import RPi.GPIO but tolerate environments where it's unavailable
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO = None
    GPIO_AVAILABLE = False

LOCK_PIN = 23

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LOCK_PIN, GPIO.OUT)


def lock_on(pin: int = LOCK_PIN):
    """Engage the lock (GPIO LOW on this hardware). If GPIO not available, noop with a log."""
    if GPIO_AVAILABLE:
        GPIO.output(pin, GPIO.LOW)
    else:
        print(f"[lock_control] GPIO not available — lock_on({pin}) (noop)")


def lock_off(pin: int = LOCK_PIN):
    """Disengage the lock (GPIO HIGH on this hardware). If GPIO not available, noop with a log."""
    if GPIO_AVAILABLE:
        GPIO.output(pin, GPIO.HIGH)
    else:
        print(f"[lock_control] GPIO not available — lock_off({pin}) (noop)")


def cleanup():
    """Cleanup GPIO if available."""
    if GPIO_AVAILABLE:
        GPIO.cleanup()
    else:
        print("[lock_control] GPIO not available — cleanup (noop)")


def blink(delay: float = 1.0):
    """Simple blink loop for manual testing of the lock pin."""
    try:
        while True:
            lock_on(LOCK_PIN)
            time.sleep(delay)
            lock_off(LOCK_PIN)
            time.sleep(delay)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    blink()
