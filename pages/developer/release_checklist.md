title: Release Checklist
author: Jack Atkinson
date: Last Updated: February 2026

This guide covers how to prepare and push a release of FTorch.
Follow this checklist to ensure that documentation, tags, archives etc. are all updated
consistently and correctly.

## Release Checklist

1. Ensure that you are working from the `main` branch.
1. Clarify the name of the release.
    1. Releases follow [Semantic Versioning](https://semver.org/) with the form `vX.Y.Z`
1. Create a git branch `vX.Y.Z-release`
1. Update the `CHANGELOG.md`.
    1. Move all content currently listed under `## Unreleased` to be under a header for the current version
        1. The new header should be titled `## vX.Y.Z - yyyy-mm-dd`
        1. The version number should be a hyperlink to the GitHub tag:
           `https://github.com/Cambridge-ICCS/FTorch/releases/tag/vX.Y.Z`
        1. Add a link to the diff before the changes of the form:
           `GitHub diff with A.B.C` pointing to
           `https://github.com/Cambridge-ICCS/FTorch/compare/vA.B.C...vX.Y.Z)`,
           where `A.B.C` was the previous release.
    1. Update the `## Unreleased` section:
        1. Add empty section headers ready for future changes
        1. Update the GitHub diff to compare `vX.Y.Z` to `HEAD`
    1. Add the `CHANGELOG.md` to the git staging index
1. Update the `CMakeLists.txt`
    1. Change `set(PACKAGE_VERSION X.Y.Z)`
    1. Add the `CMakeLists.txt` to the git staging index
1. Update code on the git remote:
    1. Commit the updated version files with message:
       `Update version number for release vX.Y.Z`
    1. Push to remote.
    1. Open a Pull Request for release preparation
    1. Once approved merge via rebase
1. Create a tag for the release
    1. Ensure that you are on the `main` branch.
    1. Run `git pull` to get the latest version of the code, ensuring that the most
       recent commit is `Update version number for release vX.Y.Z`.
    1. Run `git tag -a vX.Y.Z -m "Version X.Y.Z"`. Optionally add `-s` to also sign.
    1. Push tags to the remote with `git push --tags origin main`.
1. Create a GitHub release corresponding to the new tag.
    1. [Release creation for FTorch](https://github.com/Cambridge-ICCS/FTorch/releases/new).
    1. Title: v.X.Y.Z
    1. Add brief release notes, referencing full details in Changelog.
    1. Ensure "Set as the latest release" is checked.
1. Check that [Zenodo archive](https://zenodo.org/records/14968153) is updated accordingly.
    1. Zenodo has been linked with the ICCS GitHub so the archive should appear automatically.
    1. Check author details and descriptions and update manually as appropriate.
1. Party like you just published a release!
    1. Publicise on mailing list and wherever appropriate.
