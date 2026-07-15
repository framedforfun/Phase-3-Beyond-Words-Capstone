from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from sprint4_app.inference import CLASS_NAMES, MODEL_PATH, predict_image
except ModuleNotFoundError:
    from inference import CLASS_NAMES, MODEL_PATH, predict_image


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def parse_multipart_image(body: bytes, content_type: str) -> tuple[bytes, str]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Missing multipart boundary")

    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
    delimiter = b"--" + boundary.encode()

    for part in body.split(delimiter):
        if b"Content-Disposition:" not in part:
            continue
        header_blob, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue

        headers = header_blob.decode("utf-8", errors="ignore")
        if 'name="image"' not in headers:
            continue

        filename = "uploaded-image"
        disposition = next(
            (line for line in headers.splitlines() if line.lower().startswith("content-disposition:")),
            "",
        )
        for item in disposition.split(";"):
            item = item.strip()
            if item.startswith("filename="):
                filename = item.split("=", 1)[1].strip().strip('"') or filename
                break

        payload = payload.removesuffix(b"\r\n")
        payload = payload.removesuffix(b"--")
        return payload, filename

    raise ValueError('Expected an uploaded file field named "image"')


class Sprint4Handler(BaseHTTPRequestHandler):
    server_version = "BeyondWordsSprint4/1.0"

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/health":
            self.send_json(
                {
                    "status": "ok",
                    "model_found": MODEL_PATH.exists(),
                    "classes": len(CLASS_NAMES),
                }
            )
            return

        if parsed_path in {"/", "/index.html"}:
            self.serve_file(STATIC_DIR / "index.html")
            return

        safe_path = Path(unquote(parsed_path.lstrip("/")))
        requested = (STATIC_DIR / safe_path).resolve()
        if STATIC_DIR.resolve() not in requested.parents:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if requested.is_file():
            self.serve_file(requested)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")

        try:
            body = self.rfile.read(content_length)
            image_bytes, filename = parse_multipart_image(body, content_type)
            result = predict_image(image_bytes)
            result["filename"] = filename
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def serve_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    print(f"Starting Sprint 4 demo at http://{host}:{port}")
    print(f"Using model checkpoint: {MODEL_PATH}")
    server = ThreadingHTTPServer((host, port), Sprint4Handler)
    server.serve_forever()


if __name__ == "__main__":
    run()
