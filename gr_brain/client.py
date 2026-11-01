"""A small client for the Brain API (your access token, Brain endpoints only)."""

from __future__ import annotations

import base64
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from gr_brain import __version__


class APIError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class BrainClient:
    def __init__(self, api: str, token: Optional[str], *, timeout: float = 60.0,
                 transport: Optional[httpx.BaseTransport] = None):
        if not token:
            raise APIError(401, "not signed in - create a token on the Brain page, then run `gr-brain login --token ...`")
        self.api = api.rstrip("/")
        self._http = httpx.Client(
            base_url=self.api, timeout=timeout, transport=transport,
            headers={"Authorization": f"Bearer {token}", "User-Agent": f"gr-brain/{__version__}"},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "BrainClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- plumbing ------------------------------------------------------------
    def _call(self, method: str, path: str, **kw) -> Any:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise APIError(0, f"could not reach {self.api}: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:300]
            if isinstance(detail, list):        # FastAPI validation errors
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            elif isinstance(detail, dict):
                detail = detail.get("message") or str(detail)
            raise APIError(r.status_code, str(detail or r.reason_phrase))
        return r.json() if r.content else None

    # -- projects & recordings -----------------------------------------------
    def projects(self) -> List[Dict[str, Any]]:
        return self._call("GET", "/api/v1/brain/projects")

    def project(self, project_id: str) -> Dict[str, Any]:
        for p in self.projects():
            if p["id"] == project_id:
                return p
        raise APIError(404, f"project {project_id} not found (or not yours)")

    def episodes(self, project_id: str, robot: str) -> List[Dict[str, Any]]:
        return self._call("GET", f"/api/v1/projects/{project_id}/brain/episodes", params={"robot": robot})

    def dataset(self, project_id: str, dataset: str, robot: str) -> Dict[str, Any]:
        """Every episode of one dataset (format gr-brain-dataset/1)."""
        return self._call("GET", f"/api/v1/projects/{project_id}/brain/datasets/{dataset}/download",
                          params={"robot": robot}, timeout=300)

    # -- models --------------------------------------------------------------
    def models(self, project_id: str, robot: str) -> List[Dict[str, Any]]:
        return self._call("GET", f"/api/v1/projects/{project_id}/brain/models", params={"robot": robot})

    def upload_model(self, project_id: str, *, robot: str, name: str, onnx_bytes: bytes,
                     adapter: Dict[str, Any], info: Dict[str, Any], licence: str = "") -> Dict[str, Any]:
        body = {
            "robot": robot, "name": name, "licence": licence, "adapter": adapter, "info": info,
            "content_base64": base64.b64encode(onnx_bytes).decode("ascii"),
        }
        return self._call("POST", f"/api/v1/projects/{project_id}/brain/models", json=body, timeout=300)

    # -- server verification -------------------------------------------------
    def request_verification(self, project_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("POST", f"/api/v1/projects/{project_id}/brain/verifications", json=body)

    def verification(self, verification_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/api/v1/brain/verifications/{verification_id}")

    def wait_verification(self, verification_id: str, *, poll_s: float = 3.0, timeout_s: float = 2400,
                          on_poll: Optional[Callable[[Dict[str, Any]], None]] = None,
                          sleep: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        sleep = sleep or time.sleep
        deadline = time.monotonic() + timeout_s
        while True:
            v = self.verification(verification_id)
            if on_poll:
                on_poll(v)
            if v["status"] in ("done", "failed"):
                return v
            if time.monotonic() > deadline:
                raise APIError(0, f"verification {verification_id} is still {v['status']} after {timeout_s:.0f} s")
            sleep(poll_s)
