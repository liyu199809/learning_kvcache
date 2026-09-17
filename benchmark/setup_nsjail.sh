#!/usr/bin/env bash
# Ubuntu 24.04 amd64: download/extract dependencies locally; never apt install.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD="$ROOT/benchmark/runtime/nsjail-build"
NSJAIL_REV=f78475530b46d0186111a9096b30725f816b55fe
KAFEL_REV=76d0f41bf3eb5c4008713d64b9767b461a9129a3
mkdir -p "$BUILD/debs" "$BUILD/deps" "$BUILD/source"
if [[ ! -f "$BUILD/source/Makefile" ]]; then
    curl --fail --location --retry 3 "https://codeload.github.com/google/nsjail/tar.gz/$NSJAIL_REV" -o "$BUILD/nsjail.tar.gz"
    tar -xzf "$BUILD/nsjail.tar.gz" --strip-components=1 -C "$BUILD/source"
fi
if [[ ! -f "$BUILD/source/kafel/Makefile" ]]; then
    mkdir -p "$BUILD/source/kafel"
    curl --fail --location --retry 3 "https://codeload.github.com/google/kafel/tar.gz/$KAFEL_REV" -o "$BUILD/kafel.tar.gz"
    tar -xzf "$BUILD/kafel.tar.gz" --strip-components=1 -C "$BUILD/source/kafel"
fi
if [[ ! -x "$BUILD/deps/usr/bin/protoc" ]]; then
    cd "$BUILD/debs"
    apt-get download libprotobuf-dev=3.21.12-8.2ubuntu0.3 \
        protobuf-compiler=3.21.12-8.2ubuntu0.3 libprotobuf32t64=3.21.12-8.2ubuntu0.3 \
        libprotobuf-lite32t64=3.21.12-8.2ubuntu0.3 libprotoc32t64=3.21.12-8.2ubuntu0.3 \
        bison=2:3.8.2+dfsg-1build2 flex=2.6.4-8.2build1
    for package in "$BUILD"/debs/*.deb; do dpkg-deb -x "$package" "$BUILD/deps"; done
fi
bash "$ROOT/benchmark/build_nsjail.sh"
printf '%s\n' "NsJail source: $NSJAIL_REV" "Kafel source: $KAFEL_REV" > "$BUILD/source_versions.txt"
