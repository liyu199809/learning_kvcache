#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD="$ROOT/benchmark/runtime/nsjail-build"
DEPS="$BUILD/deps"
export PATH="$DEPS/usr/bin:$PATH"
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BISON_PKGDATADIR="$DEPS/usr/share/bison"
cd "$BUILD/source"
nice -n 15 make -j2 \
    PROTOBUF_CFLAGS="-I$DEPS/usr/include" \
    PROTOBUF_LIBS="-L$DEPS/usr/lib/x86_64-linux-gnu -Wl,-Bstatic -lprotobuf -Wl,-Bdynamic -lz"
mkdir -p "$ROOT/benchmark/runtime/bin"
install -m 755 nsjail "$ROOT/benchmark/runtime/bin/nsjail"
ldd "$ROOT/benchmark/runtime/bin/nsjail"
sha256sum "$ROOT/benchmark/runtime/bin/nsjail" "$BUILD"/debs/*.deb
