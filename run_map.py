"""Generate the hotspot map and serve it through a local web server."""

import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

from visualize_hotspots import OUTPUT_FILE, create_map, load_results


PROJECT_DIR = Path(__file__).resolve().parent
PORT = 8000


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    create_map(load_results()).save(OUTPUT_FILE)

    handler = http.server.SimpleHTTPRequestHandler
    server = ReusableTCPServer(("127.0.0.1", PORT), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{PORT}/outputs/{OUTPUT_FILE.name}"
    print(f"Map generated: {OUTPUT_FILE}")
    print(f"Opening: {url}")
    print("Keep this terminal open while viewing the map.")
    print("Press Ctrl+C to stop the local server.")

    try:
        webbrowser.open(url)
        server_thread.join()
    except KeyboardInterrupt:
        print("\nStopping local map server.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    # The server must serve the project root, not the outputs folder.
    import os

    os.chdir(PROJECT_DIR)
    main()
