#!/bin/sh
# Runs before the payload is unpacked, on both dpkg and rpm.
#
# The `nereid` user has to exist *before* postinstall chowns the state
# directory, and creating it here (rather than letting the package metadata
# declare an owner) keeps the two package formats behaving identically —
# neither dpkg nor rpm can map an unknown user name while unpacking.
set -e

# nologin lives in different places across distros, and a minimal image may
# carry none of them — pick the first that is actually there rather than
# hardcoding a path useradd will warn about.
nologin=""
for candidate in /usr/sbin/nologin /sbin/nologin /usr/bin/false /bin/false; do
    if [ -x "$candidate" ]; then
        nologin="$candidate"
        break
    fi
done

if ! getent passwd nereid >/dev/null 2>&1; then
    if command -v useradd >/dev/null 2>&1; then
        useradd --system --no-create-home --home-dir /var/lib/nereid-server \
                ${nologin:+--shell "$nologin"} --comment "nereid-server" nereid
    elif command -v adduser >/dev/null 2>&1; then
        adduser --system --no-create-home --home /var/lib/nereid-server nereid
    fi
fi

exit 0
