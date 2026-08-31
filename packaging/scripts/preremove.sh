#!/bin/sh
# dpkg passes "remove"/"upgrade"/"deconfigure"; rpm passes a count of the
# versions that will remain (0 = this is a real uninstall, 1 = an upgrade).
# Stop the unit only for an actual removal, so an upgrade doesn't drop traffic
# between the two halves of the transaction.
set -e

case "${1:-}" in
    remove|purge|0)
        if command -v systemctl >/dev/null 2>&1; then
            systemctl stop nereid-server >/dev/null 2>&1 || true
            systemctl disable nereid-server >/dev/null 2>&1 || true
        fi
        ;;
esac

exit 0
