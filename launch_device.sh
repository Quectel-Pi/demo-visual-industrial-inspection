#!/usr/bin/env bash
set -euo pipefail

DESKTOP_USER=$(ps -C gnome-shell -o user= | head -n 1 | tr -d ' ')
if [[ -z "${DESKTOP_USER}" ]]; then
  echo "No GNOME desktop session found. Open http://127.0.0.1:8765/ in your device browser."
  exit 1
fi
DESKTOP_UID=$(id -u "${DESKTOP_USER}")
DESKTOP_HOME=$(getent passwd "${DESKTOP_USER}" | cut -d: -f6)
RUNTIME="/run/user/${DESKTOP_UID}"
AUTH=$(ps -C Xwayland -o args= | awk '{for (field_index=1;field_index<=NF;field_index++) if ($field_index=="-auth") print $(field_index+1)}' | head -n 1)
DISPLAY_NAME=$(ps -C Xwayland -o args= | awk '{for (field_index=1;field_index<=NF;field_index++) if ($field_index ~ /^:[0-9]+$/) {print $field_index; break}}' | head -n 1)
PROFILE="${DESKTOP_HOME}/.local/share/board-inspection-firefox"
install -d -o "${DESKTOP_USER}" -g "$(id -gn "${DESKTOP_USER}")" "${PROFILE}"

if pgrep -f '[f]irefox.*board-inspection-firefox' >/dev/null; then
  echo "Board inspection browser is already running"
  exit 0
fi

COMMAND=(env \
  DISPLAY="${DISPLAY_NAME:-:0}" XAUTHORITY="${AUTH}" \
  XDG_RUNTIME_DIR="${RUNTIME}" DBUS_SESSION_BUS_ADDRESS="unix:path=${RUNTIME}/bus" \
  MOZ_ENABLE_WAYLAND=1 firefox --no-remote --profile "${PROFILE}" \
  --kiosk http://127.0.0.1:8765/)
if [[ "${EUID}" -ne "${DESKTOP_UID}" ]]; then
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this launcher as the active desktop user (${DESKTOP_USER})." >&2
    exit 1
  fi
  COMMAND=(runuser -u "${DESKTOP_USER}" -- "${COMMAND[@]}")
fi
nohup "${COMMAND[@]}" > /tmp/board-inspection-browser.log 2>&1 < /dev/null &
echo "Device GUI launched for ${DESKTOP_USER}, browser log: /tmp/board-inspection-browser.log"