# Source this file: `. ./env.sh`
# NixOS: pip wheels (numpy, scipy) need libstdc++ and zlib from the nix store.
if command -v nix >/dev/null 2>&1; then
  for _pkg in stdenv.cc.cc.lib zlib; do
    _lib=$(nix build --no-link --print-out-paths "nixpkgs#$_pkg" 2>/dev/null | tail -1)
    [ -n "$_lib" ] && export LD_LIBRARY_PATH="$_lib/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
  unset _pkg _lib
fi
_here=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
if [ ! -d "$_here/.venv" ]; then
  # system python3 carries torch with ROCm (GPU); the venv sees it via --system-site-packages
  uv venv -q --system-site-packages --python /run/current-system/sw/bin/python3 "$_here/.venv" &&
    VIRTUAL_ENV="$_here/.venv" uv pip install -q --index-strategy unsafe-best-match \
      -r "$_here/test/requirements.txt" -r "$_here/model/requirements.txt"
fi
. "$_here/.venv/bin/activate"
export PYTHONPATH="$_here/model${PYTHONPATH:+:$PYTHONPATH}"
unset _here
