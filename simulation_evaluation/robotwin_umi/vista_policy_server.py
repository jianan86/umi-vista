#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from rpc_common import recv_message, send_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--debug-image-dir", type=Path, default=None)
    parser.add_argument("--image-channel-order", default="rgb", choices=["rgb", "bgr"])
    parser.add_argument("--enable-fisheye", default="true")
    return parser.parse_args()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class VistaPolicyRPC:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        project_root = args.project_root.resolve()
        lerobot_root = args.lerobot_root.resolve()
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys.path.insert(0, str(lerobot_root / "src"))
        sys.path.insert(0, str(lerobot_root / "third_party" / "pi_transformers" / "src"))

        from vista_umi_model import LerobotVistaUMI

        debug_dir = None
        if args.debug_image_dir is not None:
            debug_dir = str(args.debug_image_dir / args.task_name)
        self.model = LerobotVistaUMI(
            task_name=args.task_name,
            pretrained_checkpoint_path=str(args.policy_path),
            image_channel_order=args.image_channel_order,
            debug_image_dir=debug_dir,
            save_debug_images=False,
        )
        self.model.enable_fisheye = as_bool(args.enable_fisheye)
        self.policy_path = str(args.policy_path)
        print(
            json.dumps(
                {
                    "event": "server_ready",
                    "server": "vista_policy_server",
                    "task": args.task_name,
                    "policy_path": self.policy_path,
                    "pi0_step": self.model.pi0_step,
                    "enable_fisheye": self.model.enable_fisheye,
                    "image_channel_order": args.image_channel_order,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def health(self) -> dict[str, Any]:
        return {
            "server": "vista_policy_server",
            "task": self.args.task_name,
            "policy_path": self.policy_path,
            "pi0_step": int(self.model.pi0_step),
            "enable_fisheye": bool(self.model.enable_fisheye),
            "image_channel_order": self.args.image_channel_order,
        }

    def reset(self) -> dict[str, Any]:
        self.model.reset_obsrvationwindows()
        return {"reset": True}

    def set_language(self, instruction: str) -> dict[str, Any]:
        self.model.set_language(str(instruction))
        return {"instruction": str(instruction)}

    def _to_vista_images(self, images: dict[str, np.ndarray]) -> list[Any]:
        left = images.get("left")
        right = images.get("right")
        head = images.get("head")
        if left is None or right is None:
            raise ValueError(f"VISTA RoboTwin-UMI RPC requires left/right images, got keys={list(images)}")
        return [head, right, left]

    def infer_chunk(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        if instruction is not None:
            self.model.set_language(str(instruction))
        self.model.update_observation_window(self._to_vista_images(images), np.asarray(state, dtype=np.float32))
        actions = self.model.get_action()[: self.model.pi0_step]
        print(
            json.dumps(
                {"event": "infer_chunk", "task": self.args.task_name, "shape": list(actions.shape)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return {"actions": actions, "pi0_step": int(self.model.pi0_step)}

    def update_observation(self, images: dict[str, np.ndarray], state: np.ndarray) -> dict[str, Any]:
        self.model.update_observation_window(self._to_vista_images(images), np.asarray(state, dtype=np.float32))
        return {"updated": True}

    def dispatch(self, request: dict[str, Any]) -> Any:
        cmd = request.get("cmd")
        if cmd == "health":
            return self.health()
        if cmd == "reset":
            return self.reset()
        if cmd == "set_language":
            return self.set_language(request["instruction"])
        if cmd == "infer_chunk":
            return self.infer_chunk(
                images=request["images"],
                state=request["state"],
                instruction=request.get("instruction"),
            )
        if cmd == "update_observation":
            return self.update_observation(images=request["images"], state=request["state"])
        if cmd == "shutdown":
            return {"shutdown": True}
        raise ValueError(f"unknown command: {cmd}")


def main() -> None:
    args = parse_args()
    if not (args.policy_path / "config.json").is_file():
        raise FileNotFoundError(f"missing policy config: {args.policy_path / 'config.json'}")
    server = VistaPolicyRPC(args)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
        sock.listen(1)
        print(f"[vista-rpc-server] listening {args.host}:{args.port}", flush=True)
        should_stop = False
        while not should_stop:
            conn, addr = sock.accept()
            print(f"[vista-rpc-server] client connected {addr}", flush=True)
            with conn:
                while True:
                    try:
                        request = recv_message(conn)
                    except ConnectionError:
                        break
                    try:
                        result = server.dispatch(request)
                        send_message(conn, {"ok": True, "result": result})
                        if request.get("cmd") == "shutdown":
                            should_stop = True
                            break
                    except Exception as exc:
                        tb = traceback.format_exc()
                        print(tb, flush=True)
                        send_message(conn, {"ok": False, "error": f"{exc}\n{tb}"})
                        break


if __name__ == "__main__":
    main()
