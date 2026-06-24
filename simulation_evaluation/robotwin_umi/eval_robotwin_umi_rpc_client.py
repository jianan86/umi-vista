#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from rpc_common import RPCClient


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def log_eval_env() -> None:
    spec = importlib.util.find_spec("lerobot")
    print(f"[robotwin-umi-client] lerobot spec in eval env: {spec}", flush=True)


def encode_obs_umi(observation):
    obs = observation["observation"]
    left_rgb = obs["left_camera"]["rgb"]
    right_rgb = obs["right_camera"]["rgb"]
    head_rgb = obs.get("head_camera", {}).get("rgb")
    if head_rgb is None:
        head_rgb = np.zeros_like(left_rgb)
    endpose = observation["endpose"]
    state = np.asarray(
        endpose["left_endpose"]
        + [endpose["left_gripper"]]
        + endpose["right_endpose"]
        + [endpose["right_gripper"]],
        dtype=np.float32,
    )
    return {"head": head_rgb, "left": left_rgb, "right": right_rgb}, state


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception as exc:
        raise SystemExit(f"No Task: {task_name}") from exc


def get_camera_config(parent_directory: str, camera_type: str):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def eval_step(task_env, client: RPCClient, observation, skip_get_obs_within_replan: bool):
    images, state = encode_obs_umi(observation)
    response = client.call("infer_chunk", images=images, state=state)
    actions = np.asarray(response["actions"], dtype=np.float32)[: int(response.get("pi0_step", 50))]
    for action in actions:
        task_env.take_action(action, action_type="ee")
        if skip_get_obs_within_replan:
            continue
        observation = task_env.get_obs()
        images, state = encode_obs_umi(observation)
        client.call("update_observation", images=images, state=state)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean_umi_new")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--policy-name", default="vista_rpc")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tag", default="vista_robotwin_umi_rpc")
    parser.add_argument("--ckpt-setting", default="pretrained_model")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--instruction-source", default="generated_template")
    parser.add_argument("--eval-result-root", type=Path, required=True)
    parser.add_argument("--eval-video-log", default="false")
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--skip-get-obs-within-replan", default="true")
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, required=True)
    parser.add_argument("--rpc-timeout", type=float, default=180.0)
    return parser.parse_args()


def main():
    log_eval_env()
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    script_dir = robotwin_root / "script"
    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "description" / "utils"))
    sys.path.insert(0, str(script_dir))

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions
    from test_render import Sapien_TEST

    print("[robotwin-umi-client] Sapien_TEST start", flush=True)
    Sapien_TEST()
    print("[robotwin-umi-client] Sapien_TEST done", flush=True)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"./task_config/{args.task_config}.yml", "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args["task_name"] = args.task_name
    task_args["task_config"] = args.task_config
    task_args["ckpt_setting"] = args.ckpt_setting
    task_args["eval_video_log"] = as_bool(args.eval_video_log)
    task_args["render_freq"] = int(args.render_freq)

    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type_value):
        robot_file = embodiment_types[embodiment_type_value]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["policy_name"] = args.policy_name
    task_args["eval_mode"] = True

    save_dir = (
        args.eval_result_root
        / args.tag
        / args.task_name
        / args.policy_name
        / args.task_config
        / str(args.ckpt_setting)
        / current_time.replace(" ", "_").replace(":", "-")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if task_args["eval_video_log"]:
        camera = get_camera_config(str(script_dir), task_args["camera"]["head_camera_type"])
        video_size = f"{camera['w']}x{camera['h']}"
        task_args["eval_video_save_dir"] = save_dir

    print("============= Config =============")
    print(f"Task: {args.task_name}")
    print(f"Task config: {args.task_config}")
    print(f"Policy path: {args.policy_path}")
    print(f"RPC: {args.rpc_host}:{args.rpc_port}")
    print(f"Test num: {args.test_num}")
    print(f"Skip get_obs within replan: {as_bool(args.skip_get_obs_within_replan)}")
    print(f"Instruction source: {args.instruction_source}/{args.instruction_type}")
    print("==================================")

    task_env = class_decorator(args.task_name)
    task_env.suc = 0
    task_env.test_num = 0
    st_seed = 100000 * (1 + int(args.seed))
    now_seed = st_seed
    now_id = 0
    succ_seed = 0
    successes = 0
    expert_check = True

    with RPCClient(args.rpc_host, args.rpc_port, timeout=args.rpc_timeout) as client:
        print(f"[robotwin-umi-client] health: {client.call('health')}", flush=True)
        while succ_seed < int(args.test_num):
            render_freq = task_args["render_freq"]
            task_args["render_freq"] = 0
            episode_info = None
            if expert_check:
                try:
                    task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
                    episode_info = task_env.play_once()
                    task_env.close_env()
                except UnStableError:
                    task_env.close_env()
                    now_seed += 1
                    task_args["render_freq"] = render_freq
                    continue
                except Exception:
                    traceback.print_exc()
                    task_env.close_env()
                    now_seed += 1
                    task_args["render_freq"] = render_freq
                    continue

            if (not expert_check) or (task_env.plan_success and task_env.check_success()):
                succ_seed += 1
            else:
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

            task_args["render_freq"] = render_freq
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args.task_name, episode_info_list, int(args.test_num))
            instruction = np.random.choice(results[0][args.instruction_type])
            task_env.set_instruction(instruction=instruction)
            client.call("reset")
            client.call("set_language", instruction=task_env.get_instruction())
            print(f"[robotwin-umi-client] env instruction: {task_env.get_instruction()}", flush=True)

            ffmpeg = None
            if task_env.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", video_size,
                        "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                        "-vcodec", "libx264", "-crf", "23",
                        f"{task_env.eval_video_path}/episode{task_env.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                task_env._set_eval_video_ffmpeg(ffmpeg)

            succ = False
            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                eval_step(task_env, client, observation, as_bool(args.skip_get_obs_within_replan))
                if task_env.eval_success:
                    succ = True
                    break

            if task_env.eval_video_path is not None:
                task_env._del_eval_video_ffmpeg()

            if succ:
                successes += 1
                print("Success!", flush=True)
            else:
                print("Fail!", flush=True)

            now_id += 1
            task_env.close_env(clear_cache=((succ_seed + 1) % int(task_args["clear_cache_freq"]) == 0))
            task_env.test_num += 1
            now_seed += 1
            print(
                f"{args.task_name} | {args.policy_name} | {args.task_config} | "
                f"Success rate: {successes}/{task_env.test_num} = {successes / max(task_env.test_num, 1):.4f}",
                flush=True,
            )

    result = {
        "timestamp": current_time,
        "task": args.task_name,
        "policy_name": args.policy_name,
        "policy_path": str(args.policy_path),
        "task_config": args.task_config,
        "test_num": int(args.test_num),
        "successes": int(successes),
        "success_rate": float(successes / int(args.test_num)),
        "skip_get_obs_within_replan": as_bool(args.skip_get_obs_within_replan),
        "instruction_source": args.instruction_source,
        "instruction_type": args.instruction_type,
        "save_dir": str(save_dir),
        "rpc_host": args.rpc_host,
        "rpc_port": int(args.rpc_port),
    }
    with (save_dir / "_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with (save_dir / "_result.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
