"""
DoodleLens Web Server Launcher
Serves the web studio locally on http://localhost:8000 and opens the browser.
"""

import http.server
import socketserver
import os
import webbrowser
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
os.chdir(WEB_DIR)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable caching during development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def main():
    global PORT
    for port_attempt in range(PORT, PORT + 20):
        try:
            with socketserver.TCPServer(("", port_attempt), CustomHandler) as httpd:
                url = f"http://localhost:{port_attempt}"
                print("\n" + "=" * 60)
                print(f"[DoodleLens Web Studio] Running live at: {url}")
                print(f"Press Ctrl+C in this terminal to stop the web server.")
                print("=" * 60 + "\n")
                
                # Open browser
                webbrowser.open(url)
                
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\n🛑 DoodleLens Web Server stopped.")
                return
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 10048:
                continue
            raise

if __name__ == "__main__":
    main()
