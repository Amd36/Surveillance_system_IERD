#!/bin/bash
set -Eeuo pipefail

APP_USER="ierd"
APP_GROUP="ierd"
APP_HOME="/home/ierd"
REPO_DIR="$APP_HOME/Surveillance_system_IERD"

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    echo "Run this installer with sudo." >&2
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    echo "Repository not found: $REPO_DIR" >&2
    exit 1
fi

chmod +x \
    "$REPO_DIR/scripts/update_repo.sh" \
    "$REPO_DIR/scripts/start_kiosk.sh" \
    "$REPO_DIR/scripts/install_kiosk.sh"

install -m 0644 "$REPO_DIR/deploy/systemd/surveillance-web.service" /etc/systemd/system/surveillance-web.service
install -m 0644 "$REPO_DIR/deploy/systemd/surveillance-update.service" /etc/systemd/system/surveillance-update.service
install -m 0644 "$REPO_DIR/deploy/systemd/surveillance-update.timer" /etc/systemd/system/surveillance-update.timer

install -d -m 0755 -o "$APP_USER" -g "$APP_GROUP" "$APP_HOME/.config/autostart"
install -m 0644 -o "$APP_USER" -g "$APP_GROUP" \
    "$REPO_DIR/deploy/autostart/surveillance-kiosk.desktop" \
    "$APP_HOME/.config/autostart/surveillance-kiosk.desktop"

systemctl daemon-reload
systemctl enable --now surveillance-web.service
systemctl enable --now surveillance-update.timer

# Run one update check immediately.
systemctl start surveillance-update.service || true

echo "Installed."
echo "Web service:    systemctl status surveillance-web.service"
echo "Update timer:   systemctl list-timers surveillance-update.timer"
echo "Update logs:    journalctl -u surveillance-update.service"
