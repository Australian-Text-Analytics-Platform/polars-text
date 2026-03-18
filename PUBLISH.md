# Publishing polars-text

This repository publishes pre-built wheels for Linux, macOS, and Windows with
GitHub Actions and trusted publishing.

The workflow is defined in `.github/workflows/release.yml` and follows the
standard split used by mature Rust-backed Python packages:

- build wheels separately per platform
- build one source distribution
- upload artifacts once from a dedicated publish job
- publish tagged releases to PyPI
- use TestPyPI only for explicit manual dry runs

It uses `maturin` directly in CI. That is the normal path for PyO3 packages,
and it matches both the official `maturin-action` examples and `maturin
generate-ci` output.

The workflow intentionally tracks the latest `v1` release of
`PyO3/maturin-action` and `pypa/gh-action-pypi-publish`, and it lets
`maturin-action` install its default latest `maturin` release instead of
pinning an exact `maturin-version`.

## Release policy

The workflow treats versions as follows:

- Branch pushes and pull requests: build and validate artifacts only, no upload
- Tagged releases such as `v0.2.0`, `v0.2.0rc1`, or `v0.2.0b1`: publish to PyPI
- Manual `workflow_dispatch` runs can publish the selected ref to TestPyPI

This avoids the common problem of repeatedly uploading the same version from
branch builds.

## One-time setup

Before the first release, configure the package index side correctly.

### 1. Create the project on TestPyPI and PyPI

Reserve the `polars-text` project name on both indexes.

For a brand new package, do this by creating a pending trusted publisher on
PyPI and, if you want manual dry runs, on TestPyPI as well.

### 2. Configure trusted publishing

Add GitHub trusted publishers on both indexes.

Recommended settings:

- Owner: `Australian-Text-Analytics-Platform`
- Repository: `polars-text`
- Workflow: `release.yml`
- Environment: leave this empty unless you later add GitHub environment gates

The workflow currently publishes without GitHub environment gates to keep the
CI configuration simple and validator-clean.

The workflow uses OIDC trusted publishing, so no long-lived API token is
required for the normal release path.

### 3. Optional: add GitHub environments

If you want manual approval gates before publishing, create these environments
in the GitHub repository settings:

- `testpypi`
- `pypi`

Recommended protection rules:

- `testpypi`: optional reviewer protection
- `pypi`: require one or more reviewers before publish

If you enable environments later, add the matching `environment:` entries back
to the publish jobs in `.github/workflows/release.yml` and update the trusted
publisher configuration on PyPI and TestPyPI to match.

### 4. Verify repository permissions

The publish jobs only need:

- `contents: read`
- `id-token: write`

Do not broaden those permissions unless there is a demonstrated need.

## First release checklist

Before publishing `v0.1.0`, complete this once:

1. Create a PyPI account for the maintainer who will administer the project
2. Enable 2FA on that PyPI account
3. Go to `https://pypi.org/manage/account/publishing/`
4. Add a pending trusted publisher for `polars-text`
5. Fill in the pending publisher with these values:

- PyPI project name: `polars-text`
- Owner: `Australian-Text-Analytics-Platform`
- Repository: `polars-text`
- Workflow file name: `release.yml`
- Environment name: leave blank with the current workflow

1. If you want TestPyPI dry runs, repeat the same setup at `https://test.pypi.org/manage/account/publishing/`
2. Make sure `pyproject.toml` and `Cargo.toml` both say `0.1.0`
3. Make sure the workflow file is already present on GitHub default branch before pushing the release tag

You do not need to create or store a PyPI API token for the normal release
path with the current workflow.

## What the workflow does

On every pull request and on pushes to `main` or `dev`, the workflow:

- builds Linux, macOS Intel, macOS Apple Silicon, and Windows wheels
- builds an `sdist`
- runs `twine check --strict` on the built artifacts

On tags:

- any tag matching `v*` publishes to PyPI
- prerelease versions remain prereleases because the package version itself uses
  PEP 440 markers such as `rc`, `b`, or `a`

On manual dispatch:

- choose `publish_target = testpypi` to publish the selected ref to TestPyPI
- leave `publish_target = none` to run a manual build-only check

The publish step happens only after every build job succeeds.

## Why the workflow uses maturin directly

For this project, `maturin` is the build backend and the packaging-specific
tool. Using it directly in CI is more mainstream than wrapping the same build
through `uv build`.

Why:

- `maturin-action` is the standard GitHub Actions path used by PyO3 projects
- it handles wheel targets, manylinux options, and Rust-specific packaging
  flags directly
- official `maturin` documentation and generated CI templates are centered on
  `maturin build`, `maturin sdist`, and `maturin publish`

`uv build` is still a valid generic frontend for PEP 517 builds, but it does
not add much value here because this package already depends on `maturin` for
the actual compiled-extension build logic.

## Development and preview builds

Normal branch pushes do not publish to TestPyPI. This is deliberate.

Why:

- package indexes do not allow overwriting the same version
- a static version in the repository would collide quickly
- branch builds are better handled as GitHub Actions artifacts unless someone
  truly needs an installable preview

If you need a preview package that can be installed with `pip`, the more
typical approach is to publish a prerelease version to PyPI using a tag like
`v0.3.0rc1`. Use TestPyPI when you specifically want a dry run of the publish
pipeline rather than a real public prerelease.

## Dev RC procedure

Use this when you want a public installable preview before an official release.

### RC example

You are preparing `0.3.0`, but you want a candidate build first.

1. Update the version in `pyproject.toml` to `0.3.0rc1`
2. Update the version in `Cargo.toml` to `0.3.0rc1`
3. Commit the change
4. Push the branch
5. Tag the release candidate

```bash
git tag v0.3.0rc1
git push origin v0.3.0rc1
```

1. Wait for the workflow to publish to PyPI as a prerelease
2. Verify installation from PyPI

```bash
python -m pip install --pre polars-text==0.3.0rc1
```

1. If more fixes are needed, increment the rc number and repeat with `rc2`, `rc3`, and so on

Notes:

- do not reuse the same rc version for new content

## Optional TestPyPI dry run

Use this when you want to verify trusted publishing and package upload behavior
without creating a real public release on PyPI.

1. Open the `Package And Release` workflow in GitHub Actions
2. Choose `Run workflow`
3. Select the ref you want to test
4. Set `publish_target` to `testpypi`
5. Run the workflow

After the workflow completes, install from TestPyPI if you want to verify the
artifact end to end.

## Official release procedure

Use this for a normal stable release.

### Stable release example

You are promoting `0.3.0`.

1. Update the version in `pyproject.toml` to `0.3.0`
2. Update the version in `Cargo.toml` to `0.3.0`
3. Commit the version bump
4. Push the branch
5. Create and push the stable tag

```bash
git tag v0.3.0
git push origin v0.3.0
```

1. Wait for the workflow to publish to PyPI
2. Verify installation from PyPI

```bash
python -m pip install polars-text==0.3.0
```

The PyPI publish job intentionally does not use `skip-existing`. Stable release
duplicates should fail loudly.

## First `v0.1.0` release procedure

Use this exact flow for the first public release of `polars-text`.

1. Confirm the first-release checklist above is complete
2. In `pyproject.toml`, set the project version to `0.1.0`
3. In `Cargo.toml`, set the crate version to `0.1.0`
4. Run local validation

```bash
python -m pip install --upgrade maturin twine pytest
maturin build --release --out dist --interpreter 3.14 --compatibility pypi
maturin sdist --out dist
twine check --strict dist/*
pytest -q
```

1. Commit and push the release-ready state to GitHub
2. Create and push the first release tag

```bash
git tag v0.1.0
git push origin v0.1.0
```

1. Open the GitHub Actions run for `Package And Release`
2. Wait for all wheel and sdist jobs to finish
3. Wait for the `Publish To PyPI` job to finish
4. Verify the package appears on PyPI
5. Verify installation from PyPI

```bash
python -m pip install polars-text==0.1.0
```

If the publish job fails with a trusted publishing error on the first attempt,
the usual cause is a mismatch in repository owner, repository name, workflow
file name, or environment name in PyPI's pending publisher settings.

## Hotfix procedure

Use this when a released version needs a patch release quickly.

### Hotfix example

You need to fix `0.3.0` with a patch release `0.3.1`.

1. Create a branch from the release tag or from the release commit

```bash
git checkout -b hotfix/0.3.1 v0.3.0
```

1. Apply the fix
2. Update the version in both `pyproject.toml` and `Cargo.toml` to `0.3.1`
3. Run the local validation commands

```bash
maturin build --release --out dist --interpreter 3.14 --compatibility pypi
python -m pip install --upgrade twine
twine check --strict dist/*
pytest -q
```

1. Merge the hotfix branch using your normal review process
2. Tag and push the hotfix release

```bash
git tag v0.3.1
git push origin v0.3.1
```

1. Back-merge or cherry-pick the hotfix back to the ongoing development branch

## Local validation before tagging

Before any tag, run at least:

```bash
python -m pip install --upgrade maturin twine pytest
maturin build --release --out dist --interpreter 3.14 --compatibility pypi
twine check --strict dist/*
pytest -q
```

If you want a local source distribution check as well:

```bash
maturin sdist --out dist
twine check --strict dist/*
```

## Common failure modes

### Version mismatch

Symptom:

- build or publish fails because the Python and Rust package metadata are out of sync

Fix:

- make `pyproject.toml`, `Cargo.toml`, and your release tag agree before retrying

### Wrong index target

Symptom:

- a package was published to the wrong index

Fix:

- use tagged releases for PyPI
- use manual workflow dispatch with `publish_target = testpypi` for TestPyPI dry runs

### Missing trusted publisher configuration

Symptom:

- publish job fails during authentication

Fix:

- verify the trusted publisher settings on PyPI or TestPyPI
- verify the environment name matches the configured publisher

### Re-running a stable release

Symptom:

- PyPI publish fails because files already exist

Fix:

- do not try to republish the same stable version
- create a new patch release instead

## Suggested release cadence

For most changes:

- feature work happens on normal branches
- RCs are used only when someone needs an installable preview
- stable releases are tag-driven and intentional
- hotfixes are patch releases from the latest stable baseline

This keeps the process close to what mature compiled Python projects already do
in the ecosystem without adding unnecessary version automation.
