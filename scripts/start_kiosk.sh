#!/bin/bash
set -u

HEALTH_URL="http://127.0.0.1:5000/status"
APP_URL="http://127.0.0.1:5000"

until curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null; do
    sleep 2
done

exec chromium \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --no-first-run \
    --start-maximized \
    "$APP_URL"
