#!/bin/sh
# Runs after the payload is unpacked, on both dpkg and rpm.
set -e

# The model folders are the Python backend's virtualenv scratch space, so the
# whole state tree has to belong to the service user.
if getent passwd nereid >/dev/null 2>&1; then
    chown -R nereid:nereid /var/lib/nereid-server 2>/dev/null || \
        chown -R nereid /var/lib/nereid-server 2>/dev/null || true
fi

if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload >/dev/null 2>&1 || true
    # Deliberately not enabled or started: nereid.yaml ships with an empty
    # `models` list and the server exits on an empty list, so an autostart here
    # would only produce a failed unit. `systemctl enable --now nereid-server`
    # once the config names a model.
    #
    # On upgrade, restart only what was already running.
    if systemctl is-active --quiet nereid-server 2>/dev/null; then
        systemctl restart nereid-server >/dev/null 2>&1 || true
    fi
fi

exit 0
