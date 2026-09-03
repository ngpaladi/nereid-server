#!/bin/sh
#
# Verify the generated dnf repository by installing from it, inside a clean
# distro container:
#
#   docker run --rm -v "$PWD:/w" -w /w almalinux:9 /w/packaging/test-rpm-repo.sh /w/site/rpm
#
# The metadata is read over file:// — this runs before the site is deployed,
# so there is no published copy to point at yet — but the packages themselves
# are fetched over the network from the xml:base baked into primary.xml, i.e.
# from the real GitHub release URLs. So this exercises everything that is
# actually being generated: that dnf parses the metadata, that the xml:base
# resolves to a downloadable package, and that both signature checks pass.
#
# What it deliberately does not re-test is the package's own contents — that is
# packaging/smoke-test.sh's job, run against the freshly built .rpm in the
# packages workflow.
set -eu

repo_dir="${1:?usage: test-rpm-repo.sh <repo-dir>}"

# An empty repo is the correct state before the first signed release. Nothing
# to install, so nothing to assert beyond "dnf accepts the metadata", which the
# makecache below still covers.
[ -f "$repo_dir/repodata/repomd.xml" ] || {
    echo "no repomd.xml in $repo_dir" >&2; exit 1; }

# primary.xml carries the count on its root element; a repo with none says
# packages="0". createrepo_c gzips it and prefixes the checksum, hence the glob.
primary="$(ls "$repo_dir"/repodata/*primary.xml.gz 2>/dev/null | head -1)"
[ -n "$primary" ] || { echo "no primary.xml.gz in $repo_dir/repodata" >&2; exit 1; }
if gzip -dc "$primary" | head -c 2048 | grep -q 'packages="0"'; then
    have_pkgs=0
else
    have_pkgs=1
fi

# Reuse the generated .repo verbatim except for baseurl/gpgkey, which have to
# point at the not-yet-published copy on disk. Everything else — gpgcheck,
# repo_gpgcheck — is exactly what a user will get.
sed -e "s#^baseurl=.*#baseurl=file://$repo_dir/#" \
    -e "s#^gpgkey=.*#gpgkey=file://$repo_dir/RPM-GPG-KEY-nereid#" \
    "$repo_dir/nereid-server.repo" > /etc/yum.repos.d/nereid-server.repo
echo "--- /etc/yum.repos.d/nereid-server.repo ---"
cat /etc/yum.repos.d/nereid-server.repo
echo "---"

if [ -f "$repo_dir/RPM-GPG-KEY-nereid" ]; then
    rpm --import "$repo_dir/RPM-GPG-KEY-nereid"
    echo "OK: imported the repository signing key"
fi

# -y on every dnf call below, makecache included: dnf keeps its own key store
# separate from the rpm keyring, so the gpgkey= is offered for import the first
# time the repo is used. Left to prompt, it reads "N" off a non-tty and then
# reports the failure as "Bad GPG signature" — which looks like a signing bug
# and is not one.
dnf -q -y --disablerepo='*' --enablerepo=nereid-server makecache
echo "OK: dnf accepted the repository metadata"

if [ "$have_pkgs" -eq 0 ]; then
    echo "SKIP: repository lists no packages yet — nothing to install"
    exit 0
fi

# Assert the package is actually visible: a repo dnf could not read leaves
# `list` printing "no matching packages" and exiting 0, so without this the
# install below would be the only thing standing between a broken repo and a
# green build.
dnf -y --disablerepo='*' --enablerepo=nereid-server list --available nereid-server \
    | grep -q '^nereid-server' || {
    echo "the repository metadata does not offer nereid-server" >&2
    exit 1
}

# The real test: resolve, fetch from the release URL in the xml:base, verify the
# package signature against the imported key, install.
dnf install -y --disablerepo='*' --enablerepo=nereid-server nereid-server
echo "OK: installed nereid-server from the repository"

# `dnf install` would have failed on a bad signature already; asserting the
# installed package carries one keeps that from passing vacuously if gpgcheck
# were ever turned off in the generated .repo.
flags="$(rpm -q --queryformat \
    '%|RSAHEADER?{1}:{0}|%|DSAHEADER?{1}:{0}|%|SIGPGP?{1}:{0}|%|SIGGPG?{1}:{0}|' \
    nereid-server)"
case "$flags" in
    *1*) ;;
    *) echo "the installed package carries no signature" >&2; exit 1 ;;
esac
echo "OK: installed package is signed — $(rpm -q --queryformat '%{RSAHEADER:pgpsig}' nereid-server)"

out="$(cd /var/lib/nereid-server && nereid-server 2>&1 || true)"
echo "$out" | grep -q "config must contain at least one model" || {
    echo "the installed server did not start as expected:" >&2
    echo "$out" >&2
    exit 1
}
echo "OK: the installed server runs"

dnf remove -y -q nereid-server
echo "OK: repository install/remove round-trip is clean"
