"""Obfuscate the data embedded in the published HTML reports.

The reports are static pages, so the browser must end up with the data - this
only keeps it out of View Source, `curl | grep` and search-engine indexes
(gzip -> XOR with a short key -> base64). Anyone with devtools can still dump
the decoded data; it is a speed bump, not access control.

Usage from a generator script:
    payload = pack(text)                       # goes inside <script type="text/plain">
    html = template.replace("__UNPACK_JS__", UNPACK_JS)
and in the page: `var text = await unpack(el.textContent);` (needs an async
function; DecompressionStream needs Chrome 80+, Firefox 113+, Safari 16.4+).
"""

import base64
import csv
import gzip
import io
import re
import shutil
import subprocess
import textwrap

TERSER = ["npx", "--yes", "terser@5", "--compress", "--mangle"]

XOR_KEY = b"similarity-explorer"

UNPACK_JS = (
    "var UNPACK_KEY = %r;\n"
    "async function unpack(b64) {\n"
    "  var bin = atob(b64.replace(/\\s+/g, ''));\n"
    "  var bytes = new Uint8Array(bin.length);\n"
    "  for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i) ^ UNPACK_KEY.charCodeAt(i %% UNPACK_KEY.length);\n"
    "  var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));\n"
    "  return await new Response(stream).text();\n"
    "}\n"
) % XOR_KEY.decode("ascii")


def pack(text: str) -> str:
    """gzip -> XOR -> base64 (wrapped at 76 columns) so the result is inert text inside a <script>."""
    data = gzip.compress(text.encode("utf-8"), mtime=0)
    scrambled = bytes(b ^ XOR_KEY[i % len(XOR_KEY)] for i, b in enumerate(data))
    return "\n".join(textwrap.wrap(base64.b64encode(scrambled).decode("ascii"), 76))


def unpack(payload: str) -> str:
    """Inverse of pack(); used by tests / sanity checks, the pages use UNPACK_JS."""
    scrambled = base64.b64decode("".join(payload.split()))
    data = bytes(b ^ XOR_KEY[i % len(XOR_KEY)] for i, b in enumerate(scrambled))
    return gzip.decompress(data).decode("utf-8")


def drop_csv_columns(csv_text: str, columns: set[str]) -> str:
    """Re-serialise CSV text without the named columns (the on-disk CSV stays complete)."""
    rows = list(csv.reader(io.StringIO(csv_text, newline="")))
    header = rows[0]
    unknown = columns - set(header)
    if unknown:
        raise SystemExit(f"cannot drop unknown CSV columns: {sorted(unknown)}")
    keep = [i for i, name in enumerate(header) if name not in columns]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for row in rows:
        writer.writerow([row[i] for i in keep])
    return out.getvalue()


def splice(template: str, replacements: dict[str, str], template_name: str) -> str:
    """Replace each marker, requiring it to appear exactly once in the template."""
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise SystemExit(f"{template_name} must contain exactly one {marker} marker")
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def minify_js(js: str) -> str:
    """Compress + mangle JS with terser (via npx; needs Node and, on first run, network)."""
    if shutil.which("npx") is None:
        raise SystemExit("npx not found - install Node.js to minify the report JS")
    result = subprocess.run(TERSER, input=js, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit(f"terser failed:\n{result.stderr}")
    return result.stdout.strip()


def minify_inline_scripts(html: str) -> str:
    """Minify every plain inline <script>...</script> (not src= / type= data blocks) in the page."""
    return re.sub(
        r"<script>(.*?)</script>",
        lambda m: "<script>" + minify_js(m.group(1)) + "</script>",
        html,
        flags=re.S,
    )
