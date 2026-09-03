# Installing from the dnf repository

`x86_64` RPMs are published to a dnf/yum repository hosted alongside these docs, so
RHEL-family machines can install and upgrade nereid-server the same way they get anything
else:

```bash
sudo dnf config-manager --add-repo https://www.noahpaladino.com/nereid-server/rpm/nereid-server.repo
sudo dnf install nereid-server
```

On dnf5 (Fedora 41+) the first command is `sudo dnf config-manager addrepo --from-repofile=...`;
on older systems `dnf config-manager` lives in the `dnf-plugins-core` package. Failing either,
the repo file is a plain text file and can simply be dropped in place:

```bash
sudo curl -fsSL -o /etc/yum.repos.d/nereid-server.repo \
    https://www.noahpaladino.com/nereid-server/rpm/nereid-server.repo
```

You will be asked once to accept the signing key, whose fingerprint is shown in the prompt.
From then on `dnf upgrade` picks up new releases like any other package.

## Requirements

The package is built on Enterprise Linux 9, so it needs **glibc 2.34 or newer** —
RHEL/Rocky/AlmaLinux 9+ or Fedora 38+ — on `x86_64`. There is no `aarch64` build. See
[Building](building.md) if you need either.

## What the repository is, exactly

The metadata lives on this docs site; the packages themselves stay on
[GitHub Releases](https://github.com/ngpaladi/nereid-server/releases) and dnf is pointed at
them by an `xml:base` in the repository's `primary.xml`. That split is not incidental — one
package is ~200 MB, because it carries its own libtorch, and GitHub Pages caps a site at
1 GB. So the repository is metadata only, and `dnf install` downloads from `github.com`.

Two consequences worth knowing:

- **The repository lists the three most recent releases.** Older versions stay downloadable
  from the releases page, but `dnf` will not offer them.
- **Only signed packages are listed.** The `.repo` file sets `gpgcheck=1` and
  `repo_gpgcheck=1`, so both the packages and the repository metadata are verified against
  the published key. Anything unsigned is left out of the metadata rather than offered and
  then refused.

## Verifying by hand

```bash
# the key the repository is signed with
curl -fsSL https://www.noahpaladino.com/nereid-server/rpm/RPM-GPG-KEY-nereid | gpg --show-keys

# what dnf sees
dnf --disablerepo='*' --enablerepo=nereid-server list --available nereid-server

# a downloaded package, against the imported key
sudo rpm --import https://www.noahpaladino.com/nereid-server/rpm/RPM-GPG-KEY-nereid
rpm -K nereid-server-*.x86_64.rpm    # "digests signatures OK"
```

## Debian and Ubuntu

There is no apt repository. Every release attaches a `.deb` next to the `.rpm`, installable
directly:

```bash
sudo apt install ./nereid-server_<version>-1_amd64.deb
```

It needs Debian 12+ or Ubuntu 22.04+, the same glibc 2.34 floor.

## After installing

Both packages lay out the same files and install the systemd unit **disabled**, since the
shipped config has no models in it:

| Path | Contents |
| --- | --- |
| `/usr/bin/nereid-server` | symlink to the binary |
| `/usr/lib/nereid-server/` | the bundle: binary + `lib/*.so` |
| `/usr/lib/systemd/system/nereid-server.service` | the unit, installed disabled |
| `/var/lib/nereid-server/nereid.yaml` | your config — an upgrade never clobbers it |
| `/var/lib/nereid-server/ml-backends/` | where model folders go |
| `/usr/share/doc/nereid-server/` | `nereid.yaml.example`, README, LICENSE |

The unit runs as the `nereid` system user with `WorkingDirectory=/var/lib/nereid-server`,
which is how the server finds its config: it reads `nereid.yaml` from the working directory
and resolves `server.ml_backends_path` relative to it. So configure it, then start it:

```bash
sudo cp -r mymodel /var/lib/nereid-server/ml-backends/
sudo -e /var/lib/nereid-server/nereid.yaml     # add an entry under `models:`
sudo chown -R nereid:nereid /var/lib/nereid-server
sudo systemctl enable --now nereid-server
```

See the [model contract](model-contract.md) for what belongs in a model folder. Uninstalling
leaves `/var/lib/nereid-server` and the models in it alone.
