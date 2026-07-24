#!/usr/bin/env bash
#
# check_backend_docs.sh — keep the docs' backend list in step with the code.
#
# The backends table in docs/backends.md is the canonical list the rest of the
# site links to (see docs/backends.md "Contributing"). The one thing that can
# still drift is whether that table even mentions every backend the code
# registers. This checks exactly that: it reads the registered `name` of each
# backend from src/backends/*/mod.rs and fails if any is missing from the table.
#
# So registering a backend without adding its row is a CI failure, not a silent
# omission. It does NOT try to validate the other columns — those are prose the
# author maintains; this only guarantees coverage.

set -euo pipefail
cd "$(dirname "$0")/.."

TABLE="docs/backends.md"
missing=0

# The registered backend name from each `BackendRegistration { name: "..." }`.
# Only mod.rs holds registrations, so this won't pick up a tensor's `name:` in
# imp.rs. Sorted-unique for stable output.
names="$(grep -hoE 'name:[[:space:]]*"[a-z0-9_]+"' src/backends/*/mod.rs \
         | grep -oE '"[a-z0-9_]+"' | tr -d '"' | sort -u)"

if [ -z "$names" ]; then
  echo "check_backend_docs: found no registered backends under src/backends/ — is the layout unchanged?" >&2
  exit 1
fi

for name in $names; do
  # The Feature column lists the backend as a code span, e.g. `torch`.
  if grep -qF "\`$name\`" "$TABLE"; then
    echo "  ok    $name"
  else
    echo "  MISSING  $name — registered in src/backends/, but absent from $TABLE" >&2
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "check_backend_docs: add the missing backend(s) to the table in $TABLE." >&2
  exit 1
fi

echo "check_backend_docs: all registered backends appear in $TABLE."
