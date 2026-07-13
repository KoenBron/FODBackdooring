"""
Plot Server: Scans directories for PNG files and serves them as an HTML gallery.

The gallery has two levels:
  - Collections: the direct subdirectories of --root (e.g. fl_retention_exp,
    other_exp). Switch between them live via the nav bar (?dir=NAME) or set the
    default landing collection with --subdir.
  - Sections:    inside a collection, plots are grouped by their experiment
    subdirectory into collapsible sections, newest first.

Usage:
    python server.py [--root DIR] [--subdir NAME] [--port PORT] [--host HOST]

Examples:
    python server.py                               # scan ./outputs, land on first collection
    python server.py --subdir other_exp            # land on outputs/other_exp
    python server.py --root ./experiments          # scan a different base dir
    python server.py --port 8080                   # custom port

Then visit http://<your-tailscale-hostname>:8000 from your laptop.
"""

import argparse
import html
import http.server
import os
import socketserver
import urllib.parse
from datetime import datetime
from pathlib import Path


def list_collections(base: Path):
    """Direct subdirectories of base that contain at least one PNG (recursively).

    These are the switchable 'collections' (fl_retention_exp, other_exp, ...).
    Hidden dirs and dirs with no plots are skipped so the nav bar stays clean.
    """
    cols = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and any(child.rglob("*.png")):
            cols.append(child.name)
    return cols


def resolve_collection(base: Path, requested, default_subdir):
    """Pick which collection to display and return (active, collections).

    Preference order: the ?dir= request, then the --subdir default, then the
    first collection. Only names that are real subdirectories are honoured, so a
    crafted ?dir= value can never escape base. Returns active=None when base has
    no plot-bearing subdirectories (render base itself, no nav bar).
    """
    cols = list_collections(base)
    for candidate in (requested, default_subdir):
        if candidate and candidate in cols:
            return candidate, cols
    if cols:
        return cols[0], cols
    return None, cols


def find_pngs(root: Path):
    """Recursively find PNGs, grouped two levels deep for ordered presentation:

        {top_subdir: {parent_dir: [png, ...]}}

    - top_subdir  : the FIRST path component under root — i.e. the experiment
                    output subdirectory the gallery sorts and collapses by. PNGs
                    that sit directly in root fall under '(root)'.
    - parent_dir  : the full relative parent directory, so plots nested deeper
                    than one level keep their finer grouping inside the section.

    Top-level sections are presented newest-first (see build_index_html); the
    inner subsections and the files within each are sorted by name, so the
    layout is deterministic across refreshes.
    """
    groups = {}
    for png in sorted(root.rglob("*.png")):
        rel = png.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue  # skip hidden dirs
        rel_dir = png.parent.relative_to(root)
        top = rel_dir.parts[0] if rel_dir.parts else "(root)"
        parent_key = str(rel_dir) if str(rel_dir) != "." else "(root)"
        groups.setdefault(top, {}).setdefault(parent_key, []).append(png)
    return groups


def build_index_html(base: Path, scan_root: Path, active, collections) -> str:
    """Build the HTML page for one collection.

    base        : directory the HTTP file server serves from (plot URLs are
                  relative to it, so any collection's images resolve).
    scan_root   : the collection directory actually scanned for plots.
    active      : name of the active collection (None when scanning base itself).
    collections : all collection names, for the switcher nav bar.
    """
    groups = find_pngs(scan_root)
    total = sum(len(files) for inner in groups.values() for files in inner.values())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>Plot Gallery</title>",
        "<style>",
        "  body { font-family: -apple-system, system-ui, sans-serif; ",
        "         max-width: 1400px; margin: 2em auto; padding: 0 1em; ",
        "         background: #fafafa; color: #222; }",
        "  h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }",
        "  h3 { color: #555; font-size: 0.95em; margin: 1.2em 0 0.3em; ",
        "       font-family: monospace; word-break: break-all; }",
        "  .meta { color: #666; font-size: 0.9em; }",
        "  .nav { background: #333; border-radius: 6px; padding: 0.6em 0.9em; ",
        "         margin: 1em 0; color: #ddd; }",
        "  .nav a { display: inline-block; margin: 0.2em 0.4em; padding: 0.3em 0.7em; ",
        "           border-radius: 4px; color: #ddd; text-decoration: none; ",
        "           background: #4a4a4a; }",
        "  .nav a:hover { background: #5c5c5c; color: #fff; }",
        "  .nav a.active { background: #2d7; color: #063; font-weight: 600; }",
        "  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); ",
        "          gap: 1.5em; margin: 1em 0; }",
        "  .card { background: white; border: 1px solid #ddd; border-radius: 6px; ",
        "          padding: 0.8em; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }",
        "  .card img { width: 100%; height: auto; cursor: zoom-in; ",
        "              border: 1px solid #eee; }",
        "  .card .name { font-family: monospace; font-size: 0.85em; ",
        "                color: #555; margin-top: 0.5em; word-break: break-all; }",
        "  .card .mtime { color: #999; font-size: 0.75em; }",
        "  .toc { background: white; border: 1px solid #ddd; border-radius: 6px; ",
        "         padding: 1em; margin: 1em 0; }",
        "  .toc a { display: inline-block; margin-right: 1em; }",
        "  .controls { margin: 1em 0; }",
        "  .controls button { padding: 0.4em 0.8em; margin-right: 0.5em; cursor: pointer; }",
        "  /* collapsible per-subdirectory sections */",
        "  details.section { background: white; border: 1px solid #ddd; border-radius: 6px; ",
        "                    margin: 1em 0; padding: 0 0.9em 0.4em; }",
        "  details.section > summary { cursor: pointer; list-style: none; user-select: none; ",
        "                    font-size: 1.15em; font-weight: 600; color: #444; ",
        "                    padding: 0.6em 0.2em; }",
        "  details.section > summary::-webkit-details-marker { display: none; }",
        "  details.section > summary::before { content: '\\25B8'; display: inline-block; ",
        "                    margin-right: 0.5em; transition: transform 0.15s; }",
        "  details.section[open] > summary::before { transform: rotate(90deg); }",
        "  details.section > summary:hover { color: #000; }",
        "  .count { color: #999; font-weight: normal; font-size: 0.85em; }",
        "  /* lightbox */",
        "  .lightbox { display: none; position: fixed; inset: 0; ",
        "              background: rgba(0,0,0,0.9); z-index: 999; ",
        "              cursor: zoom-out; }",
        "  .lightbox.active { display: flex; align-items: center; justify-content: center; }",
        "  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; }",
        "</style></head><body>",
        f"<h1>Plot Gallery</h1>",
    ]

    # Collection switcher — the direct subdirectories of the base directory.
    if collections:
        parts.append("<div class='nav'><b>Collection:</b> ")
        for name in collections:
            q = urllib.parse.quote(name, safe="")
            cls = " class='active'" if name == active else ""
            parts.append(f"<a href='/?dir={q}'{cls}>{html.escape(name)}</a>")
        parts.append("</div>")

    parts += [
        f"<p class='meta'>Showing: <code>{html.escape(str(scan_root.resolve()))}</code><br>",
        f"Found <b>{total}</b> PNG file(s) across <b>{len(groups)}</b> "
        f"subdirector{'y' if len(groups)==1 else 'ies'}.<br>",
        f"Page generated: {now}</p>",
        "<div class='controls'>",
        "  <button onclick='location.reload()'>&#8635; Refresh</button>",
        "  <button onclick='setAll(true)'>Expand all</button>",
        "  <button onclick='setAll(false)'>Collapse all</button>",
        "  <label><input type='checkbox' id='autoRefresh'> Auto-refresh every 30s</label>",
        "</div>",
    ]

    if not groups:
        parts.append("<p><em>No PNG files found in this collection.</em></p>")
    else:
        # Newest-first ordering: top-level names are date-stamped, so reverse
        # lexicographic sort puts the most recent runs at the top.
        ordered_tops = sorted(groups, reverse=True)

        # Table of contents — one entry per top-level subdirectory.
        parts.append("<div class='toc'><b>Jump to:</b><br>")
        for top in ordered_tops:
            count = sum(len(f) for f in groups[top].values())
            anchor = urllib.parse.quote(top, safe="")
            parts.append(f"<a href='#{anchor}'>{html.escape(top)} ({count})</a>")
        parts.append("</div>")

        # One collapsible section per top-level subdirectory, newest first,
        # collapsed by default (click a summary or "Expand all" to open).
        for top in ordered_tops:
            inner = groups[top]
            count = sum(len(f) for f in inner.values())
            anchor = urllib.parse.quote(top, safe="")
            parts.append(f"<details class='section' id='{anchor}'>")
            parts.append(
                f"<summary>{html.escape(top)} <span class='count'>({count} plot"
                f"{'' if count == 1 else 's'})</span></summary>"
            )
            for parent_key in sorted(inner):
                files = sorted(inner[parent_key])
                # Show a sub-header only when the plots live deeper than the
                # section's own directory (nested subfolders). Strip the shared
                # top-level prefix so the header reads as the relative subpath.
                if parent_key not in (top, "(root)"):
                    subpath = parent_key[len(top) + 1:] if parent_key.startswith(top + "/") else parent_key
                    parts.append(f"<h3>{html.escape(subpath)}</h3>")
                parts.append("<div class='grid'>")
                for png in files:
                    # URL relative to base (what the file server serves); label
                    # relative to the scanned collection.
                    url = "/" + urllib.parse.quote(png.relative_to(base).as_posix())
                    label = png.relative_to(scan_root).as_posix()
                    mtime = datetime.fromtimestamp(png.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    size_kb = png.stat().st_size / 1024
                    parts.append("<div class='card'>")
                    parts.append(f"  <img src='{url}' alt='{html.escape(label)}' loading='lazy' onclick='openLightbox(this.src)'>")
                    parts.append(f"  <div class='name'>{html.escape(png.name)}</div>")
                    parts.append(f"  <div class='mtime'>{mtime} &middot; {size_kb:.1f} KB</div>")
                    parts.append("</div>")
                parts.append("</div>")
            parts.append("</details>")

    # Lightbox + collapse controls + auto-refresh JS
    parts.append("""
<div class='lightbox' id='lightbox' onclick='this.classList.remove("active")'>
  <img id='lightboxImg' src=''>
</div>
<script>
  function openLightbox(src) {
    document.getElementById('lightboxImg').src = src;
    document.getElementById('lightbox').classList.add('active');
  }
  function setAll(open) {
    document.querySelectorAll('details.section').forEach(d => d.open = open);
  }
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('active');
  });
  // Opening a TOC link should expand (not just scroll to) its section.
  function openTarget() {
    if (!location.hash) return;
    const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (el && el.tagName === 'DETAILS') { el.open = true; el.scrollIntoView(); }
  }
  window.addEventListener('hashchange', openTarget);
  openTarget();
  const cb = document.getElementById('autoRefresh');
  const KEY = 'plotgallery_autorefresh';
  if (localStorage.getItem(KEY) === '1') cb.checked = true;
  let timer = null;
  function applyAutoRefresh() {
    if (cb.checked) {
      localStorage.setItem(KEY, '1');
      timer = setTimeout(() => location.reload(), 30000);
    } else {
      localStorage.setItem(KEY, '0');
      if (timer) clearTimeout(timer);
    }
  }
  cb.addEventListener('change', applyAutoRefresh);
  applyAutoRefresh();
</script>
</body></html>""")
    return "\n".join(parts)


class PlotHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the dynamic index at / and PNG files from disk (under base_dir)."""

    # Set by main() before serving
    base_dir: Path = Path.cwd()
    default_subdir = None

    def __init__(self, *args, **kwargs):
        # Serve static files from the base dir so any collection's PNGs resolve.
        super().__init__(*args, directory=str(self.base_dir), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            qs = urllib.parse.parse_qs(parsed.query)
            requested = qs.get("dir", [None])[0]
            active, collections = resolve_collection(self.base_dir, requested, self.default_subdir)
            scan_root = self.base_dir / active if active else self.base_dir
            content = build_index_html(self.base_dir, scan_root, active, collections).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        # Otherwise, serve files from disk (PNGs etc.)
        super().do_GET()

    def log_message(self, fmt, *args):
        # Slightly tidier log line
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {fmt % args}")


def main():
    ap = argparse.ArgumentParser(description="Serve a gallery of PNG plots over HTTP.")
    ap.add_argument("--root", default="outputs",
                    help="Base directory to scan (default: outputs). Its direct "
                         "subdirectories become switchable collections.")
    ap.add_argument("--subdir", default="fl_retention_exp",
                    help="Collection (direct subdirectory of --root) to show first "
                         "(default: fl_retention_exp; falls back to the first collection "
                         "found if absent). Switch live via the nav bar or ?dir=NAME.")
    ap.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="Interface to bind (default: 0.0.0.0, all interfaces — needed for Tailscale access)")
    args = ap.parse_args()

    base = Path(args.root).resolve()
    if not base.is_dir():
        raise SystemExit(f"Error: {base} is not a directory")

    PlotHandler.base_dir = base
    PlotHandler.default_subdir = args.subdir

    collections = list_collections(base)
    print(f"Serving plots from: {base}")
    if collections:
        print(f"Collections:        {', '.join(collections)}")
    else:
        print("Collections:        (none — scanning base directory directly)")

    # Allow re-binding the port quickly after Ctrl+C
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer((args.host, args.port), PlotHandler) as httpd:
        print(f"Listening on:      http://{args.host}:{args.port}")
        print(f"From your laptop:  http://<desktop-tailscale-name>:{args.port}")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
