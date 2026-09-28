# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""Browse a folder of generated meshes in the browser.

    python3 view_assets.py out/

Serves the folder and renders every .glb/.gltf in it as an orbitable preview grid, picking up
face counts and timings from manifest.csv when batch_gen.py wrote one. Nothing is written to
the folder - the index page is generated in memory.

Needs network access the first time a page loads, for the model-viewer component on the CDN.
"""

import argparse
import functools
import html
import http.server
import os
import socketserver
import threading
import webbrowser

MODEL_EXTENSIONS = ('.glb', '.gltf')
VIEWER_SCRIPT = 'https://cdn.jsdelivr.net/npm/@google/model-viewer@3.1.1/dist/model-viewer.min.js'

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script type="module" src="{script}"></script>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; padding: 24px; background: #15171a; color: #e8eaed;
         font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif; }}
  h1 {{ font-size: 18px; font-weight: 600; margin: 0 0 4px; }}
  .sub {{ color: #9aa0a6; font-size: 13px; margin-bottom: 20px; }}
  .grid {{ display: grid; gap: 16px;
           grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }}
  .card {{ background: #1e2126; border: 1px solid #2c3036; border-radius: 10px; overflow: hidden; }}
  model-viewer {{ width: 100%; height: 320px; background: #24272c; display: block; }}
  .meta {{ padding: 10px 12px; }}
  .name {{ font-size: 13px; font-weight: 600; word-break: break-all; }}
  .facts {{ color: #9aa0a6; font-size: 12px; margin-top: 3px; }}
  a {{ color: #8ab4f8; text-decoration: none; }}
  .empty {{ color: #9aa0a6; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="sub">{count} model(s) &middot; drag to orbit, scroll to zoom &middot; {folder}</div>
<div class="grid">
{cards}
</div>
</body>
</html>
"""

CARD = """  <div class="card">
    <model-viewer src="{src}" camera-controls auto-rotate shadow-intensity="1"
                  environment-image="neutral" alt="{name}"></model-viewer>
    <div class="meta">
      <div class="name"><a href="{src}" download>{name}</a></div>
      <div class="facts">{facts}</div>
    </div>
  </div>
"""


def read_manifest(folder):
    """Pull face counts and timings out of batch_gen.py's manifest, if there is one."""
    path = os.path.join(folder, 'manifest.csv')
    if not os.path.exists(path):
        return {}
    import csv
    with open(path, newline='') as handle:
        return {row['name']: row for row in csv.DictReader(handle) if row.get('name')}


def describe(path, record):
    facts = [f'{os.path.getsize(path) / 1e6:.1f} MB']
    if record:
        if record.get('faces'):
            facts.append(f'{int(record["faces"]):,} faces')
        if record.get('total_seconds'):
            facts.append(f'{record["total_seconds"]}s')
    return ' &middot; '.join(facts)


def build_page(folder):
    models = sorted(f for f in os.listdir(folder) if f.lower().endswith(MODEL_EXTENSIONS))
    manifest = read_manifest(folder)
    if models:
        cards = ''.join(
            CARD.format(src=html.escape(name), name=html.escape(name),
                        facts=describe(os.path.join(folder, name),
                                       manifest.get(os.path.splitext(name)[0])))
            for name in models)
    else:
        cards = ('  <p class="empty">No .glb or .gltf files here. '
                 'OBJ is not supported by the web viewer - use f3d or Blender for those.</p>')
    return PAGE.format(title=os.path.basename(os.path.abspath(folder)) or 'Generated assets',
                       script=VIEWER_SCRIPT, count=len(models),
                       folder=html.escape(os.path.abspath(folder)), cards=cards)


class GalleryHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the folder, but answers / with a generated gallery instead of a file listing."""

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            body = build_page(self.directory).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):
        pass  # keep the terminal quiet


class ReusableServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_server(host, port, handler, attempts=20):
    """Bind the requested port, or the first free one from 8000 up when none was asked for."""
    if port is not None:
        try:
            return ReusableServer((host, port), handler)
        except OSError as error:
            raise SystemExit(f'Cannot use port {port}: {error}. '
                             f'Leave --port off to pick a free one automatically.')
    for candidate in range(8000, 8000 + attempts):
        try:
            return ReusableServer((host, candidate), handler)
        except OSError:
            continue
    raise SystemExit(f'No free port found between 8000 and {8000 + attempts - 1}.')


def main():
    parser = argparse.ArgumentParser(description='Preview generated meshes in a browser.')
    parser.add_argument('folder', help='Folder containing .glb/.gltf files.')
    parser.add_argument('--port', type=int, default=None,
                        help='Defaults to the first free port from 8000 upwards.')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--no_browser', action='store_true', help='Do not open a browser window.')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f'Not a folder: {args.folder}')

    handler = functools.partial(GalleryHandler, directory=os.path.abspath(args.folder))
    server = start_server(args.host, args.port, handler)
    with server:
        url = f'http://{args.host}:{server.server_address[1]}/'
        # flush explicitly: stdout is block-buffered when redirected, which would
        # otherwise hide the chosen port.
        print(f'Serving {os.path.abspath(args.folder)} at {url}', flush=True)
        print('Press Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nStopped.')


if __name__ == '__main__':
    main()
