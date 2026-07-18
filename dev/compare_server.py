#!/usr/bin/env python
"""Static server plus independent NumPy comparison endpoint for WebGPU output."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import ssl
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent
CASES_ROOT = ROOT / "cases"
RESULTS_ROOT = ROOT / "results"
MOTION_STATS_ROOT = ROOT.parent / "ardy" / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40" / "stats" / "motion"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

DTYPES = {
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "float16": np.dtype("<f2"),
    "int64": np.dtype("<i8"),
    "int32": np.dtype("<i4"),
    "bool": np.dtype("u1"),
}

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/octet-stream", ".bin")


def load_manifest(case_id: str) -> dict:
    if not case_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in case_id):
        raise ValueError("invalid case id")
    path = CASES_ROOT / case_id / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    finite_mask = np.isfinite(actual64)
    all_finite = bool(finite_mask.all())
    nan_count = int(np.isnan(actual64).sum())
    inf_count = int(np.isinf(actual64).sum())

    if not all_finite:
        return {
            "all_finite": False,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "max_abs_error": math.inf,
            "mean_abs_error": math.inf,
            "max_rel_error": math.inf,
            "mean_rel_error": math.inf,
            # Keep the API strict JSON even when the browser produced NaN/Inf.
            "cosine_similarity": None,
        }

    abs_error = np.abs(actual64 - reference64)
    rel_error = abs_error / np.maximum(np.abs(reference64), 1.0e-8)
    actual_flat = actual64.ravel()
    reference_flat = reference64.ravel()
    denom = np.linalg.norm(actual_flat) * np.linalg.norm(reference_flat)
    cosine = float(np.dot(actual_flat, reference_flat) / denom) if denom else float(actual_flat.size == 0)
    return {
        "all_finite": True,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "max_abs_error": float(abs_error.max(initial=0.0)),
        "mean_abs_error": float(abs_error.mean()) if abs_error.size else 0.0,
        "max_rel_error": float(rel_error.max(initial=0.0)),
        "mean_rel_error": float(rel_error.mean()) if rel_error.size else 0.0,
        "cosine_similarity": cosine,
    }


class CompareHandler(SimpleHTTPRequestHandler):
    server_version = "ArdyWebGPUCompare/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if self.path.startswith("/vendor/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, status: HTTPStatus, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/agent/next":
            client_id = parse_qs(parsed.query).get("client_id", [""])[0]
            if not client_id:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "missing client_id"})
                return
            with self.server.state_lock:
                job = next((item for item in self.server.jobs if item["status"] == "pending"), None)
                if job is not None:
                    job["status"] = "running"
                    job["client_id"] = client_id
                    job["started_at"] = datetime.now(timezone.utc).isoformat()
                    payload = dict(job)
                else:
                    payload = None
            self.send_json(HTTPStatus.OK, {"job": payload})
            return
        if parsed.path == "/api/control/status":
            with self.server.state_lock:
                payload = {
                    "clients": list(self.server.clients.values()),
                    "jobs": list(self.server.jobs),
                }
            self.send_json(HTTPStatus.OK, payload)
            return
        if parsed.path == "/api/demo/next":
            query = parse_qs(parsed.query)
            client_id = query.get("client_id", [""])[0]
            if not client_id:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "missing client_id"})
                return
            try:
                protocol = int(query.get("protocol", ["1"])[0])
            except ValueError:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid protocol"})
                return
            with self.server.state_lock:
                job = next(
                    (
                        item
                        for item in self.server.demo_jobs
                        if item["status"] == "pending"
                        and int(item.get("minimum_protocol", 1)) <= protocol
                    ),
                    None,
                )
                if job is not None:
                    job["status"] = "running"
                    job["client_id"] = client_id
                    job["started_at"] = datetime.now(timezone.utc).isoformat()
                    payload = dict(job)
                else:
                    payload = None
            self.send_json(HTTPStatus.OK, {"job": payload})
            return
        if parsed.path == "/api/demo/status":
            with self.server.state_lock:
                payload = {
                    "clients": list(self.server.demo_clients.values()),
                    "jobs": list(self.server.demo_jobs),
                    "reports": list(self.server.demo_reports[-100:]),
                }
            self.send_json(HTTPStatus.OK, payload)
            return
        if parsed.path == "/api/demo/motion-stats":
            # The checkpoint stores 5 global-root + 4 local-root + 325 body
            # statistics. Browser motion tensors use 5 global-root + 325 body,
            # matching MotionRepBase.stats rather than the raw 334-vector.
            raw_mean = np.load(MOTION_STATS_ROOT / "mean.npy").astype(np.float32)
            raw_std = np.load(MOTION_STATS_ROOT / "std.npy").astype(np.float32)
            keep = np.r_[np.arange(5), np.arange(9, raw_mean.size)]
            mean = raw_mean[keep]
            std = raw_std[keep]
            self.send_json(
                HTTPStatus.OK,
                {
                    "mean": mean.tolist(),
                    "std_eps": np.sqrt(std**2 + np.float32(1.0e-5)).tolist(),
                },
            )
            return
        if parsed.path == "/api/results":
            RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            rows = []
            for path in sorted(RESULTS_ROOT.glob("*.json"), reverse=True):
                try:
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
            self.send_json(HTTPStatus.OK, rows[:100])
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]

        if parts == ["api", "agent", "heartbeat"]:
            try:
                payload = self.read_json_body()
                client_id = str(payload["client_id"])
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            row = {
                **payload,
                "client_id": client_id,
                "remote_address": self.client_address[0],
                "last_seen": datetime.now(timezone.utc).isoformat(),
            }
            with self.server.state_lock:
                self.server.clients[client_id] = row
            self.send_json(HTTPStatus.OK, {"ok": True})
            return

        if parts in (["api", "demo", "heartbeat"], ["api", "demo", "report"]):
            try:
                payload = self.read_json_body()
                client_id = str(payload["client_id"])
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            now = datetime.now(timezone.utc).isoformat()
            row = {
                **payload,
                "client_id": client_id,
                "remote_address": self.client_address[0],
                "timestamp": now,
            }
            with self.server.state_lock:
                if parts[-1] == "heartbeat":
                    self.server.demo_clients[client_id] = row
                else:
                    self.server.demo_reports.append(row)
                    if len(self.server.demo_reports) > 1000:
                        del self.server.demo_reports[:-1000]
            self.send_json(HTTPStatus.OK, {"ok": True})
            return

        if parts == ["api", "demo", "enqueue"]:
            if self.client_address[0] not in {"127.0.0.1", "::1"}:
                self.send_json(HTTPStatus.FORBIDDEN, {"error": "enqueue is restricted to server localhost"})
                return
            try:
                payload = self.read_json_body()
                action = str(payload["action"])
                if action not in {
                    "start",
                    "pause",
                    "restart",
                    "benchmark",
                    "validate",
                    "precision_probe",
                    "waypoint",
                    "clear_waypoints",
                    "prompt",
                    "postprocess",
                    "reload",
                }:
                    raise ValueError(f"unsupported demo action: {action}")
                command = payload.get("command", {})
                if not isinstance(command, dict):
                    raise ValueError("command must be an object")
                if action == "waypoint":
                    x = float(command["x"])
                    z = float(command["z"])
                    if not math.isfinite(x) or not math.isfinite(z):
                        raise ValueError("waypoint coordinates must be finite")
                    command = {
                        "x": x,
                        "z": z,
                        "frame_offset": int(command.get("frame_offset", 60)),
                    }
                    if command["frame_offset"] < 1 or command["frame_offset"] > 10000:
                        raise ValueError("frame_offset out of range")
                elif action == "prompt":
                    command = {"prompt_id": int(command["prompt_id"])}
                    if command["prompt_id"] < 0:
                        raise ValueError("prompt_id must be non-negative")
                elif action == "postprocess":
                    mode = str(command["mode"])
                    if mode not in {"raw", "interp", "seam", "root", "pose", "balanced", "strong"}:
                        raise ValueError(f"unsupported postprocess mode: {mode}")
                    command = {"mode": mode}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            job = {
                "job_id": uuid.uuid4().hex,
                "action": action,
                "command": command,
                "status": "pending",
                "minimum_protocol": (
                    4 if action == "postprocess"
                    else (3 if action in {"prompt", "reload"}
                    else (2 if action == "precision_probe" else 1)
                    )
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            with self.server.state_lock:
                self.server.demo_jobs.append(job)
            self.send_json(HTTPStatus.CREATED, job)
            return

        if len(parts) == 4 and parts[:3] == ["api", "demo", "complete"]:
            job_id = parts[3]
            try:
                payload = self.read_json_body()
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            with self.server.state_lock:
                job = next((item for item in self.server.demo_jobs if item["job_id"] == job_id), None)
                if job is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "unknown demo job"})
                    return
                job["status"] = "passed" if payload.get("passed") else "failed"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                job["result"] = payload
                response = dict(job)
            self.send_json(HTTPStatus.OK, response)
            return

        if parts == ["api", "control", "enqueue"]:
            if self.client_address[0] not in {"127.0.0.1", "::1"}:
                self.send_json(HTTPStatus.FORBIDDEN, {"error": "enqueue is restricted to server localhost"})
                return
            try:
                payload = self.read_json_body()
                case_id = str(payload["case_id"])
                load_manifest(case_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            job = {
                "job_id": uuid.uuid4().hex,
                "case_id": case_id,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            with self.server.state_lock:
                self.server.jobs.append(job)
            self.send_json(HTTPStatus.CREATED, job)
            return

        if len(parts) == 4 and parts[:3] == ["api", "agent", "complete"]:
            job_id = parts[3]
            try:
                payload = self.read_json_body()
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            with self.server.state_lock:
                job = next((item for item in self.server.jobs if item["job_id"] == job_id), None)
                if job is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "unknown job"})
                    return
                job["status"] = "passed" if payload.get("passed") else "failed"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                job["result"] = payload
                response = dict(job)
            self.send_json(HTTPStatus.OK, response)
            return

        if len(parts) != 4 or parts[:2] != ["api", "compare"]:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        _, _, case_id, output_name = parts
        try:
            manifest = load_manifest(case_id)
            output_spec = next(item for item in manifest["outputs"] if item["name"] == output_name)
        except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            content_length = -1
        if content_length < 0 or content_length > MAX_UPLOAD_BYTES:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid upload length"})
            return

        dtype_name = self.headers.get("X-Tensor-Dtype", output_spec["dtype"])
        if dtype_name != output_spec["dtype"] or dtype_name not in DTYPES:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": f"unexpected dtype: {dtype_name}"})
            return

        dtype = DTYPES[dtype_name]
        expected_elements = int(np.prod(output_spec["shape"], dtype=np.int64))
        expected_bytes = expected_elements * dtype.itemsize
        if content_length != expected_bytes:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"byte length mismatch: got {content_length}, expected {expected_bytes}"},
            )
            return

        raw = self.rfile.read(content_length)
        actual = np.frombuffer(raw, dtype=dtype).copy().reshape(output_spec["shape"])
        reference_dtype_name = output_spec.get("reference_dtype", dtype_name)
        if reference_dtype_name not in DTYPES:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": f"unexpected reference dtype: {reference_dtype_name}"})
            return
        reference_path = ROOT / output_spec["reference_url"]
        reference = np.fromfile(reference_path, dtype=DTYPES[reference_dtype_name]).reshape(output_spec["shape"])
        metrics = compute_metrics(actual, reference)
        tolerances = output_spec.get("tolerances", manifest["tolerances"])
        primary_passed = bool(
            metrics["all_finite"]
            and metrics["max_abs_error"] <= tolerances["max_abs_error"]
            and metrics["mean_abs_error"] <= tolerances["mean_abs_error"]
            and metrics["cosine_similarity"] is not None
            and metrics["cosine_similarity"] >= tolerances["min_cosine_similarity"]
        )
        secondary_comparisons = []
        for secondary_spec in output_spec.get("secondary_references", []):
            secondary_dtype_name = secondary_spec["dtype"]
            if secondary_dtype_name not in DTYPES:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"unexpected secondary reference dtype: {secondary_dtype_name}"},
                )
                return
            secondary_reference = np.fromfile(
                ROOT / secondary_spec["url"], dtype=DTYPES[secondary_dtype_name]
            ).reshape(output_spec["shape"])
            secondary_metrics = compute_metrics(actual, secondary_reference)
            secondary_tolerances = secondary_spec["tolerances"]
            secondary_passed = bool(
                secondary_metrics["all_finite"]
                and secondary_metrics["max_abs_error"] <= secondary_tolerances["max_abs_error"]
                and secondary_metrics["mean_abs_error"] <= secondary_tolerances["mean_abs_error"]
                and secondary_metrics["cosine_similarity"] is not None
                and secondary_metrics["cosine_similarity"] >= secondary_tolerances["min_cosine_similarity"]
            )
            secondary_comparisons.append(
                {
                    "label": secondary_spec["label"],
                    "dtype": secondary_dtype_name,
                    "passed": secondary_passed,
                    "metrics": secondary_metrics,
                    "tolerances": secondary_tolerances,
                }
            )
        passed = primary_passed and all(item["passed"] for item in secondary_comparisons)

        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        raw_path = RESULTS_ROOT / f"{stamp}_{case_id}_{output_name}.bin"
        json_path = RESULTS_ROOT / f"{stamp}_{case_id}_{output_name}.json"
        raw_path.write_bytes(raw)
        result = {
            "timestamp": now.isoformat(),
            "case_id": case_id,
            "output_name": output_name,
            "dtype": dtype_name,
            "reference_dtype": reference_dtype_name,
            "shape": output_spec["shape"],
            "passed": passed,
            "metrics": metrics,
            "tolerances": tolerances,
            "secondary_comparisons": secondary_comparisons,
            "client": {
                "user_agent": self.headers.get("User-Agent", ""),
                "webgpu_adapter": self.headers.get("X-WebGPU-Adapter", ""),
            },
            "raw_output": raw_path.name,
        }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.send_json(HTTPStatus.OK, result)

    def read_json_body(self, max_bytes: int = 1024 * 1024) -> dict:
        try:
            content_length = int(self.headers.get("Content-Length", "-1"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if content_length < 0 or content_length > max_bytes:
            raise ValueError("invalid JSON body length")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload


class CompareServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.state_lock = threading.Lock()
        self.clients: dict[str, dict] = {}
        self.jobs: list[dict] = []
        self.demo_clients: dict[str, dict] = {}
        self.demo_jobs: list[dict] = []
        self.demo_reports: list[dict] = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--certfile", type=Path)
    parser.add_argument("--keyfile", type=Path)
    args = parser.parse_args()
    if bool(args.certfile) != bool(args.keyfile):
        parser.error("--certfile and --keyfile must be provided together")
    server = CompareServer((args.bind, args.port), CompareHandler)
    scheme = "http"
    if args.certfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile=args.certfile, keyfile=args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"ARDY WebGPU compare server: {scheme}://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
