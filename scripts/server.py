"""
Plot Server: Scans directories for PNG files and serves them as an HTML gallery.
 
Usage:
    python plot_server.py [--root DIR] [--port PORT] [--host HOST]
 
Examples:
    python plot_server.py                          # serve from current dir on :8000
    python plot_server.py --root ./experiments     # serve from specific dir
    python plot_server.py --port 8080              # custom port
 
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
 
 
def find_pngs(root: Path):
    """Recursively find all PNG files, grouped by their parent directory."""
    groups = {}
    for png in sorted(root.rglob("*.png")):
        if any(part.startswith(".") for part in png.relative_to(root).parts):
            continue  # skip hidden dirs
        rel_dir = png.parent.relative_to(root)
        key = str(rel_dir) if str(rel_dir) != "." else "(root)"
        groups.setdefault(key, []).append(png)
    return groups
 
 
def build_index_html(root: Path) -> str:
    """Build the HTML page listing all PNGs grouped by directory."""
    groups = find_pngs(root)
    total = sum(len(v) for v in groups.values())
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
        "  h2 { margin-top: 2em; color: #444; background: #eee; ",
        "       padding: 0.4em 0.8em; border-radius: 4px; }",
        "  .meta { color: #666; font-size: 0.9em; }",
        "  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); ",
        "          gap: 1.5em; margin-top: 1em; }",
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
        "  /* lightbox */",
        "  .lightbox { display: none; position: fixed; inset: 0; ",
        "              background: rgba(0,0,0,0.9); z-index: 999; ",
        "              cursor: zoom-out; }",
        "  .lightbox.active { display: flex; align-items: center; justify-content: center; }",
        "  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; }",
        "</style></head><body>",
        f"<h1>Plot Gallery</h1>",
        f"<p class='meta'>Root: <code>{html.escape(str(root.resolve()))}</code><br>",
        f"Found <b>{total}</b> PNG file(s) across <b>{len(groups)}</b> director{'y' if len(groups)==1 else 'ies'}.<br>",
        f"Page generated: {now}</p>",
        "<div class='controls'>",
        "  <button onclick='location.reload()'>↻ Refresh</button>",
        "  <label><input type='checkbox' id='autoRefresh'> Auto-refresh every 30s</label>",
        "</div>",
    ]
 
    if not groups:
        parts.append("<p><em>No PNG files found.</em></p>")
    else:
        # Table of contents
        parts.append("<div class='toc'><b>Jump to:</b><br>")
        for key in groups:
            anchor = urllib.parse.quote(key, safe="")
            parts.append(f"<a href='#{anchor}'>{html.escape(key)} ({len(groups[key])})</a>")
        parts.append("</div>")
 
        # Sections per directory
        for key, files in groups.items():
            anchor = urllib.parse.quote(key, safe="")
            parts.append(f"<h2 id='{anchor}'>{html.escape(key)}</h2>")
            parts.append("<div class='grid'>")
            for png in files:
                rel = png.relative_to(root).as_posix()
                url = "/" + urllib.parse.quote(rel)
                mtime = datetime.fromtimestamp(png.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                size_kb = png.stat().st_size / 1024
                parts.append("<div class='card'>")
                parts.append(f"  <img src='{url}' alt='{html.escape(rel)}' loading='lazy' onclick='openLightbox(this.src)'>")
                parts.append(f"  <div class='name'>{html.escape(png.name)}</div>")
                parts.append(f"  <div class='mtime'>{mtime} &middot; {size_kb:.1f} KB</div>")
                parts.append("</div>")
            parts.append("</div>")
 
    # Lightbox + auto-refresh JS
    parts.append("""
<div class='lightbox' id='lightbox' onclick='this.classList.remove("active")'>
  <img id='lightboxImg' src=''>
</div>
<script>
  function openLightbox(src) {
    document.getElementById('lightboxImg').src = src;
    document.getElementById('lightbox').classList.add('active');
  }
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('active');
  });
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
    """Serves the dynamic index at / and PNG files from disk."""
 
    # Set by main() before serving
    root_dir: Path = Path.cwd()
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.root_dir), **kwargs)
 
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = build_index_html(self.root_dir).encode("utf-8")
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
    ap.add_argument("--root", default="outputs/fl_retention_exp", help="Directory to scan recursively (default: current dir)")
    ap.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="Interface to bind (default: 0.0.0.0, all interfaces — needed for Tailscale access)")
    args = ap.parse_args()
 
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Error: {root} is not a directory")
 
    PlotHandler.root_dir = root
 
    # Allow re-binding the port quickly after Ctrl+C
    socketserver.TCPServer.allow_reuse_address = True
 
    with socketserver.TCPServer((args.host, args.port), PlotHandler) as httpd:
        print(f"Serving plots from: {root}")
        print(f"Listening on:      http://{args.host}:{args.port}")
        print(f"From your laptop:  http://<desktop-tailscale-name>:{args.port}")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
 
 
if __name__ == "__main__":
    main()
