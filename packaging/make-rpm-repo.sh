#!/usr/bin/env bash
#
# Assemble the dnf/yum repository that GitHub Pages serves at
# <site>/rpm/, from the .rpm assets already attached to this repo's releases.
#
#   packaging/make-rpm-repo.sh site/rpm
#
# What ends up in the output directory is *only metadata* — repodata/, the
# .repo file and the public key, a few hundred KB in total. The packages
# themselves stay on GitHub Releases and are never copied here: one .rpm is
# ~200 MB (bundled libtorch), and GitHub Pages caps a site at 1 GB, so half a
# dozen releases would blow the whole site budget. createrepo_c's --baseurl
# writes an xml:base into primary.xml, so dnf reads the metadata from Pages and
# then fetches the package itself from
# https://github.com/<slug>/releases/download/<tag>/<file>.rpm. The RPMs are
# laid out under <tag>/ below precisely so the href createrepo_c derives
# (relative to the repo root) comes out as "<tag>/<file>.rpm" and lands on that
# URL when joined to the baseurl.
#
# Only *signed* packages are listed: the published .repo sets gpgcheck=1, so an
# unsigned .rpm in the metadata would be an entry dnf refuses to install. See
# the signing step in .github/workflows/packages.yml.
#
# Environment:
#   REPO_SLUG            owner/repo to read releases from (default: this repo)
#   SITE_BASEURL         public URL of the output dir, no trailing slash
#   KEEP_RELEASES        how many releases to keep listed (default: 3)
#   INCLUDE_PRERELEASES  set to 1 to list prereleases too (default: no)
#   CACHE_DIR            where downloaded .rpm files are kept between runs
#   GPG_PRIVATE_KEY      armoured private key; when set, repomd.xml is signed
#   GPG_PASSPHRASE       passphrase for that key, if it has one
#   PKG_BASEURL          where the packages themselves live (default: this
#                        repo's release-download root). Overridden only by the
#                        local test harness, which serves them over file://.
#
# Needs: gh (authenticated via GH_TOKEN), createrepo_c, rpm, and gpg when
# signing.
set -euo pipefail

out_dir="${1:?usage: make-rpm-repo.sh <output-dir>}"

REPO_SLUG="${REPO_SLUG:-ngpaladi/nereid-server}"
# Where this metadata will be reachable from, which goes into the .repo file's
# baseurl and gpgkey. Read out of mkdocs.yml rather than repeated here: that
# file already has to carry the site URL (the theme's site-name link needs it),
# and two copies of it would only ever disagree.
if [ -z "${SITE_BASEURL:-}" ] && [ -f mkdocs.yml ]; then
    site_url="$(sed -n 's/^site_url:[[:space:]]*//p' mkdocs.yml | head -1)"
    [ -z "$site_url" ] || SITE_BASEURL="${site_url%/}/rpm"
fi
SITE_BASEURL="${SITE_BASEURL:-https://www.noahpaladino.com/nereid-server/rpm}"
KEEP_RELEASES="${KEEP_RELEASES:-3}"
INCLUDE_PRERELEASES="${INCLUDE_PRERELEASES:-0}"
CACHE_DIR="${CACHE_DIR:-$PWD/.cache/rpm-repo}"
PKG_BASEURL="${PKG_BASEURL:-https://github.com/$REPO_SLUG/releases/download/}"
# The public key is committed so that a build without access to the signing
# secret — a pull request from a fork — still produces the same gpgcheck=1
# .repo file that main does, rather than a differently-configured one that
# nobody is really testing. Exporting from the private key is the fallback for
# the first release, before the public half has been committed.
COMMITTED_PUBKEY="${COMMITTED_PUBKEY:-packaging/RPM-GPG-KEY-nereid}"

SITE_BASEURL="${SITE_BASEURL%/}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

log() { printf '%s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1. Which releases to list.
#
# Drafts have no public download URL, and prereleases are excluded by default:
# dnf installs the highest version it can see, so an -rc listed here would
# quietly become what `dnf install nereid-server` hands a new user.
# ---------------------------------------------------------------------------
log "==> reading releases from $REPO_SLUG"
tags="$(gh api "repos/$REPO_SLUG/releases" --paginate \
    --jq '.[] | select(.draft | not) | select(.prerelease | not) | .tag_name' \
    2>/dev/null || true)"
if [ "$INCLUDE_PRERELEASES" = "1" ]; then
    tags="$(gh api "repos/$REPO_SLUG/releases" --paginate \
        --jq '.[] | select(.draft | not) | .tag_name' 2>/dev/null || true)"
fi

# ---------------------------------------------------------------------------
# 2. Fetch each release's .rpm assets into <tag>/ under the cache, then link
#    them into the work tree. The cache is what keeps a docs-only push from
#    re-downloading ~600 MB it already has; it is keyed on the tag, and a
#    release's assets are immutable once published, so a cache hit is always
#    the right file.
# ---------------------------------------------------------------------------
found=0
listed_releases=0
for tag in $tags; do
    # Counted per release that actually contributes a package, not per release
    # examined: a release whose packaging run failed, or one cut before the
    # packages workflow existed, should not use up one of the slots and push a
    # good older version out of the repository.
    [ "$listed_releases" -lt "$KEEP_RELEASES" ] || break

    tag_cache="$CACHE_DIR/$tag"
    if [ ! -d "$tag_cache" ]; then
        mkdir -p "$tag_cache.tmp"
        if ! gh release download "$tag" --repo "$REPO_SLUG" \
                --pattern '*.rpm' --dir "$tag_cache.tmp" 2>/dev/null; then
            log "    $tag: no .rpm asset, skipping"
            rm -rf "$tag_cache.tmp"
            continue
        fi
        # Move into place only once complete, so an interrupted download can
        # never be mistaken for a cache hit on the next run.
        mv "$tag_cache.tmp" "$tag_cache"
    fi

    from_this_tag=0
    for rpm in "$tag_cache"/*.rpm; do
        [ -e "$rpm" ] || continue
        # gpgcheck=1 in the published .repo means an unsigned package is an
        # entry dnf will download in full and then refuse. Better to leave it
        # out of the metadata than to advertise it.
        # Which header tag carries the signature depends on the rpm doing the
        # reading (4.16-6.0 write RSAHEADER; SIGPGP/SIGGPG is the older pair),
        # so ask about all four and take any of them as a yes.
        flags="$(rpm -qp --nosignature --queryformat \
            '%|RSAHEADER?{1}:{0}|%|DSAHEADER?{1}:{0}|%|SIGPGP?{1}:{0}|%|SIGGPG?{1}:{0}|' \
            "$rpm" 2>/dev/null || echo 0000)"
        case "$flags" in
            *1*) ;;
            *)
                log "    $tag: $(basename "$rpm") is unsigned — not listing it"
                continue
                ;;
        esac
        mkdir -p "$work/pkgs/$tag"
        cp -l "$rpm" "$work/pkgs/$tag/" 2>/dev/null || cp "$rpm" "$work/pkgs/$tag/"
        found=$((found + 1))
        from_this_tag=$((from_this_tag + 1))
        log "    $tag: $(basename "$rpm")"
    done
    [ "$from_this_tag" -eq 0 ] || listed_releases=$((listed_releases + 1))
done

mkdir -p "$work/pkgs"
if [ "$found" -eq 0 ]; then
    log "==> no signed .rpm found in any release"
    log "    writing valid but empty metadata; the next signed release fills it in"
else
    log "==> listing $found package(s) from $listed_releases release(s)"
fi

# ---------------------------------------------------------------------------
# 3. Generate the metadata. --baseurl is the whole trick: it becomes the
#    xml:base on every <location>, so the href createrepo_c derives from the
#    on-disk layout ("<tag>/<file>.rpm") resolves against the release-download
#    root rather than against wherever this metadata is hosted.
# ---------------------------------------------------------------------------
log "==> createrepo_c"
createrepo_c --quiet --checksum sha256 \
    --baseurl "$PKG_BASEURL" \
    "$work/pkgs"

# ---------------------------------------------------------------------------
# 4. Sign repomd.xml, and publish the public key.
#
# Two independent checks come out of one key: gpgcheck verifies each package's
# own signature (added at build time by the packages workflow), repo_gpgcheck
# verifies this detached signature over repomd.xml — which is what stops
# anyone who can write to the Pages site from swapping the metadata to point at
# a package of their choosing.
# ---------------------------------------------------------------------------
repo_gpgcheck=0
have_pubkey=0

if [ -f "$COMMITTED_PUBKEY" ]; then
    cp "$COMMITTED_PUBKEY" "$work/RPM-GPG-KEY-nereid"
    have_pubkey=1
fi

if [ -n "${GPG_PRIVATE_KEY:-}" ]; then
    log "==> signing repomd.xml"
    GNUPGHOME="$work/gnupg"
    mkdir -p "$GNUPGHOME"
    chmod 700 "$GNUPGHOME"
    export GNUPGHOME
    printf '%s\n' "$GPG_PRIVATE_KEY" | gpg --batch --quiet --import
    key_id="$(gpg --list-secret-keys --with-colons | awk -F: '/^fpr:/ {print $10; exit}')"
    [ -n "$key_id" ] || { log "!!! GPG_PRIVATE_KEY imported no secret key"; exit 1; }

    pass_args=(--batch --pinentry-mode loopback)
    if [ -n "${GPG_PASSPHRASE:-}" ]; then
        printf '%s' "$GPG_PASSPHRASE" > "$work/gpg-pass"
        chmod 600 "$work/gpg-pass"
        pass_args+=(--passphrase-file "$work/gpg-pass")
    fi

    gpg "${pass_args[@]}" --quiet --yes --armor --detach-sign \
        --local-user "$key_id" \
        --output "$work/pkgs/repodata/repomd.xml.asc" \
        "$work/pkgs/repodata/repomd.xml"
    repo_gpgcheck=1

    if [ "$have_pubkey" -eq 0 ]; then
        log "    $COMMITTED_PUBKEY is not committed — exporting the public key from the secret"
        gpg --batch --quiet --armor --export "$key_id" > "$work/RPM-GPG-KEY-nereid"
        have_pubkey=1
    fi
else
    log "==> no GPG_PRIVATE_KEY — repomd.xml will not be signed (repo_gpgcheck=0)"
fi

# Without a public key to verify against there is nothing gpgcheck=1 could do
# but fail, so say so honestly in the .repo rather than shipping a setting that
# cannot work. On main and on a release the key is always present; this branch
# is for a fork's pull request.
gpgcheck="$have_pubkey"

# ---------------------------------------------------------------------------
# 5. The .repo file, and a small landing page so the directory URL is not a 404.
# ---------------------------------------------------------------------------
{
    printf '[nereid-server]\n'
    printf 'name=nereid-server\n'
    printf 'baseurl=%s/\n' "$SITE_BASEURL"
    printf 'enabled=1\n'
    printf 'gpgcheck=%s\n' "$gpgcheck"
    printf 'repo_gpgcheck=%s\n' "$repo_gpgcheck"
    if [ "$have_pubkey" -eq 1 ]; then
        printf 'gpgkey=%s/RPM-GPG-KEY-nereid\n' "$SITE_BASEURL"
    fi
    printf 'metadata_expire=6h\n'
} > "$work/nereid-server.repo"

cat > "$work/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nereid-server RPM repository</title>
<body style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;max-width:44rem;margin:3rem auto;padding:0 1rem;line-height:1.6">
<h1>nereid-server RPM repository</h1>
<p>A dnf/yum repository for <a href="https://github.com/$REPO_SLUG">$REPO_SLUG</a>.</p>
<pre style="white-space:pre-wrap">sudo dnf config-manager --add-repo $SITE_BASEURL/nereid-server.repo
sudo dnf install nereid-server</pre>
<p>Full instructions: <a href="../install/">Installing nereid-server</a>.</p>
<p>Packages are served from GitHub Releases; this location holds only the
repository metadata.</p>
</body>
HTML

# ---------------------------------------------------------------------------
# 6. Publish. Written into a staging dir and swapped in, so a failure above
#    leaves any existing output untouched rather than half-replaced.
# ---------------------------------------------------------------------------
stage="${out_dir}.staging"
rm -rf "$stage"
mkdir -p "$stage"
cp -r "$work/pkgs/repodata" "$stage/repodata"
cp "$work/nereid-server.repo" "$work/index.html" "$stage/"
if [ "$have_pubkey" -eq 1 ]; then
    cp "$work/RPM-GPG-KEY-nereid" "$stage/"
fi

rm -rf "$out_dir"
mv "$stage" "$out_dir"

log "==> wrote $out_dir"
ls -l "$out_dir" "$out_dir/repodata" >&2
