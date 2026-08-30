#!/bin/sh
# The `nereid` user and /var/lib/nereid-server are left in place on purpose:
# they own the installed models and their virtualenvs, which are the operator's
# data, not ours to delete.
set -e

if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload >/dev/null 2>&1 || true
fi

exit 0
