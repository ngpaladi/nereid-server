#!/bin/sh
#
# Post-install check for the nereid-server .deb / .rpm, run inside a clean
# distro container by .github/workflows/packages.yml:
#
#   docker run --rm -v "$PWD:/w" debian:bookworm-slim \
#       /w/packaging/smoke-test.sh /w/dist/pkg/nereid-server_1.2.3_amd64.deb
#
# It answers the three questions a bundled-libtorch package actually fails on:
# does it install with only its declared dependencies, does every shared object
# resolve out of $ORIGIN/lib, and does the installed binary get far enough to
# validate its config.
set -eu

pkg="${1:?usage: smoke-test.sh <package>}"

case "$pkg" in
    *.deb)
        apt-get update -qq
        # Resolves and installs the declared dependencies too, which a bare
        # `dpkg -i` would not — that's half of what is being tested here.
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg"
        ;;
    *.rpm)
        dnf install -y -q "$pkg"
        ;;
    *)
        echo "unrecognised package type: $pkg" >&2
        exit 1
        ;;
esac
echo "OK: installed with its declared dependencies only"

# 1. The bundle has to be self-contained. `build.sh --link bundled` copies
#    libtorch beside the binary and patches the rpath to $ORIGIN/lib.
#
#    Note the real path rather than the /usr/bin/nereid-server symlink: given a
#    symlink, some distros' ldd expands $ORIGIN to the *link's* directory, so
#    it goes looking in /usr/bin/lib and reports libtorch as missing on a
#    package that runs perfectly. (EL9 does this; Fedora does not, which is
#    exactly the kind of split that makes it a bad thing to assert on.) That
#    the symlink resolves $ORIGIN correctly is proven properly below, by
#    running `nereid-server` through it — at exec time the loader derives
#    $ORIGIN from the fully resolved /proc/self/exe.
deps="$(ldd /usr/lib/nereid-server/grpc-test 2>&1 || true)"
# Guard against a vacuous pass: if ldd can't read the file as a dynamic
# executable there are no "not found" lines to grep for, and the check below
# would succeed on a binary that is broken in a much worse way.
echo "$deps" | grep -q '=>' || {
    echo "ldd did not report a dynamic executable:" >&2
    echo "$deps" >&2
    exit 1
}
missing="$(echo "$deps" | grep 'not found' || true)"
if [ -n "$missing" ]; then
    echo "unresolved shared libraries after install:" >&2
    echo "$missing" >&2
    exit 1
fi
echo "OK: all shared libraries resolve out of the bundle"

# 2. The packaged files landed where the unit expects them.
#
# Nothing under /usr/share/doc is asserted here: the slim base images used for
# these tests ship a dpkg `path-exclude /usr/share/doc/*`, so the docs are
# dropped at unpack time by the container, not by the package.
for f in /usr/lib/systemd/system/nereid-server.service \
         /var/lib/nereid-server/nereid.yaml; do
    [ -f "$f" ] || { echo "missing packaged file: $f" >&2; exit 1; }
done
[ -d /var/lib/nereid-server/ml-backends ] || {
    echo "missing model directory: /var/lib/nereid-server/ml-backends" >&2; exit 1; }
getent passwd nereid >/dev/null || { echo "preinstall did not create the nereid user" >&2; exit 1; }
grep -q '^WorkingDirectory=/var/lib/nereid-server$' \
    /usr/lib/systemd/system/nereid-server.service || {
    echo "the unit's WorkingDirectory no longer matches the config location" >&2; exit 1; }
echo "OK: unit, config, service user and state directory are in place"

# 3. The shipped conffile must parse. It carries an empty `models` list on
#    purpose, so reaching *that* validation error is the success condition — it
#    means YAML parsing and schema decoding both got through.
out="$(cd /var/lib/nereid-server && nereid-server 2>&1 || true)"
echo "$out"
echo "$out" | grep -q "config must contain at least one model" || {
    echo "the shipped nereid.yaml did not parse as expected" >&2; exit 1; }
echo "OK: the shipped nereid.yaml parses"

# 4. With a configured model, the server must reach backend resolution — the
#    same checkpoint the container image's smoke test uses. Pointing
#    ml_backends_path at a missing directory gets there without shipping a
#    model (and its virtualenv) into CI.
#
#    This and the check above invoke `nereid-server`, i.e. the symlink, on
#    purpose: getting this far means the loader resolved $ORIGIN/lib through it
#    and mapped all ~450 MB of libtorch, which is the claim ldd could not
#    settle.
smoke="$(mktemp -d)"
cat > "$smoke/nereid.yaml" <<'YAML'
server:
  bind_addr: "[::]:50051"
  ml_backends_path: "/nonexistent"
models:
  - name: "smoke"
    device: "cpu"
    queue_capacity: 1
YAML
out="$(cd "$smoke" && nereid-server 2>&1 || true)"
echo "$out"
echo "$out" | grep -q "failed to resolve server.ml_backends_path" || {
    echo "server did not reach config validation as expected" >&2; exit 1; }
echo "OK: server starts and validates its config"

# 5. Removal must run the prerm/postrm scriptlets cleanly and leave the
#    operator's data behind — an uninstall that deleted the models and their
#    virtualenvs would be a nasty surprise. /var/lib/nereid-server is a
#    package-owned directory, so dpkg and rpm will reap it when it is *empty*;
#    what has to hold is that a directory with models in it is left alone,
#    which is what this stand-in model folder checks.
mkdir -p /var/lib/nereid-server/ml-backends/keepme
echo "a model lives here" > /var/lib/nereid-server/ml-backends/keepme/model.txt

case "$pkg" in
    *.deb) DEBIAN_FRONTEND=noninteractive apt-get remove -y -qq nereid-server ;;
    *.rpm) dnf remove -y -q nereid-server ;;
esac
[ ! -e /usr/bin/nereid-server ] || { echo "removal left /usr/bin/nereid-server behind" >&2; exit 1; }
[ ! -e /usr/lib/nereid-server/grpc-test ] || { echo "removal left the bundle behind" >&2; exit 1; }
[ -f /var/lib/nereid-server/ml-backends/keepme/model.txt ] || {
    echo "removal deleted an installed model under /var/lib/nereid-server" >&2; exit 1; }
echo "OK: removal is clean and leaves installed models intact"
