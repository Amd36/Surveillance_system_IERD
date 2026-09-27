#!/bin/bash
set -Eeuo pipefail

APP_USER="ierd"
APP_GROUP="ierd"
APP_HOME="/home/ierd"
REPO_DIR="$APP_HOME/Surveillance_system_IERD"
REMOTE="origin"
BRANCH="main"
WEB_SERVICE="surveillance-web.service"
UPDATE_TIMER="surveillance-update.timer"
HEALTH_URL="http://127.0.0.1:5000/status"
HEALTH_TIMEOUT=30
LOCK_FILE="/run/lock/surveillance-update.lock"
VENV="$REPO_DIR/venv"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another update is already running; exiting."
    exit 0
fi

as_app() {
    runuser -u "$APP_USER" -- env HOME="$APP_HOME" "$@"
}

git_app() {
    as_app git -C "$REPO_DIR" "$@"
}

wait_for_health() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

sync_deployment_files() {
    local changed=0

    if [ -f "$REPO_DIR/deploy/systemd/surveillance-web.service" ]; then
        install -m 0644 "$REPO_DIR/deploy/systemd/surveillance-web.service" /etc/systemd/system/surveillance-web.service
        changed=1
    fi

    if [ -f "$REPO_DIR/deploy/systemd/surveillance-update.service" ]; then
        install -m 0644 "$REPO_DIR/deploy/systemd/surveillance-update.service" /etc/systemd/system/surveillance-update.service
        changed=1
    fi

    if [ -f "$REPO_DIR/deploy/systemd/surveillance-update.timer" ]; then
        install -m 0644 "$REPO_DIR/deploy/systemd/surveillance-update.timer" /etc/systemd/system/surveillance-update.timer
        changed=1
    fi

    if [ -f "$REPO_DIR/deploy/autostart/surveillance-kiosk.desktop" ]; then
        install -d -m 0755 -o "$APP_USER" -g "$APP_GROUP" "$APP_HOME/.config/autostart"
        install -m 0644 -o "$APP_USER" -g "$APP_GROUP" \
            "$REPO_DIR/deploy/autostart/surveillance-kiosk.desktop" \
            "$APP_HOME/.config/autostart/surveillance-kiosk.desktop"
    fi

    if [ "$changed" -eq 1 ]; then
        systemctl daemon-reload
    fi
}

rollback() {
    local old_commit="$1"
    local requirements_changed="$2"

    echo "Deployment failed. Rolling back to $old_commit..." >&2
    systemctl stop "$WEB_SERVICE" || true

    git_app reset --hard "$old_commit"

    if [ "$requirements_changed" -eq 1 ] && [ -x "$VENV/bin/pip" ] && [ -f "$REPO_DIR/requirements.txt" ]; then
        as_app "$VENV/bin/pip" install -r "$REPO_DIR/requirements.txt" || true
    fi

    sync_deployment_files
    systemctl restart "$WEB_SERVICE"

    if wait_for_health; then
        echo "Rollback successful."
    else
        echo "Rollback completed, but the web app is still unhealthy." >&2
    fi
}

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Git repository not found: $REPO_DIR" >&2
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "Virtual environment not found: $VENV" >&2
    exit 1
fi

CURRENT_BRANCH="$(git_app rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo "Expected branch '$BRANCH', but repository is on '$CURRENT_BRANCH'." >&2
    exit 1
fi

if [ -n "$(git_app status --porcelain --untracked-files=no)" ]; then
    echo "Tracked local changes detected; refusing automatic update." >&2
    exit 1
fi

OLD_COMMIT="$(git_app rev-parse HEAD)"

echo "Checking $REMOTE/$BRANCH for updates..."
if ! git_app fetch "$REMOTE" "$BRANCH"; then
    echo "Git fetch failed; keeping currently installed version." >&2
    exit 0
fi

NEW_COMMIT="$(git_app rev-parse "$REMOTE/$BRANCH")"

if [ "$OLD_COMMIT" = "$NEW_COMMIT" ]; then
    echo "Already up to date."
    exit 0
fi

if ! git_app merge-base --is-ancestor "$OLD_COMMIT" "$NEW_COMMIT"; then
    echo "Remote branch is not a fast-forward from the installed commit; refusing update." >&2
    exit 1
fi

REQUIREMENTS_CHANGED=0
if git_app diff --name-only "$OLD_COMMIT" "$NEW_COMMIT" -- requirements.txt | grep -qx 'requirements.txt'; then
    REQUIREMENTS_CHANGED=1
fi

if ! git_app merge --ff-only "$NEW_COMMIT"; then
    echo "Fast-forward update failed." >&2
    exit 1
fi

if [ "$REQUIREMENTS_CHANGED" -eq 1 ]; then
    echo "requirements.txt changed; updating virtual environment..."
    if [ ! -f "$REPO_DIR/requirements.txt" ] || ! as_app "$VENV/bin/pip" install -r "$REPO_DIR/requirements.txt"; then
        rollback "$OLD_COMMIT" "$REQUIREMENTS_CHANGED"
        exit 1
    fi
fi

sync_deployment_files

systemctl restart "$WEB_SERVICE"

if wait_for_health; then
    echo "Deployment successful: $OLD_COMMIT -> $NEW_COMMIT"
    # Reloaded timer settings take effect immediately after an update.
    systemctl try-restart "$UPDATE_TIMER" >/dev/null 2>&1 || true
    exit 0
fi

rollback "$OLD_COMMIT" "$REQUIREMENTS_CHANGED"
exit 1
