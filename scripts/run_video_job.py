import argparse
import os
import queue
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import AppConfig
from src.database.db import Database
from src.orchestrator.engine import Orchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", required=True, choices=["LEFT_RIGHT", "TOP", "FRONT_REAR"])
    parser.add_argument("--name", default="TestJob")
    args = parser.parse_args()

    cfg = AppConfig.load()
    print(f"Loaded config: profile={cfg.stitching.profile}, text_thresh={cfg.detection.threshold_text}")

    cmd_q = queue.Queue()
    evt_q = queue.Queue()
    db = Database()

    orch = Orchestrator(command_queue=cmd_q, event_queue=evt_q, config=cfg, db=db)
    orch.start()

    cmd_q.put({
        "action": "START_JOB",
        "name": args.name,
        "video_path": str(Path(args.video).resolve()),
        "mode": args.mode,
    })

    t0 = time.time()
    last_log = ""
    while True:
        try:
            evt = evt_q.get(timeout=0.5)
            e_type = evt.type
            data = evt.data
            if e_type == "PROGRESS":
                stage = data.get("stage", "")
                msg = data.get("message", "")
                pct = data.get("progress", 0.0)
                log = data.get("log", "")
                if log and log != last_log:
                    print(f"[{time.time()-t0:6.1f}s] {log}", flush=True)
                    last_log = log
                elif msg and msg != last_log:
                    print(f"[{time.time()-t0:6.1f}s] [{stage} {pct*100:5.1f}%] {msg}", flush=True)
                    last_log = msg
            elif e_type == "CONTAINER_PROCESSED":
                print(f"[{time.time()-t0:6.1f}s] Container processed: {data}", flush=True)
            elif e_type == "JOB_COMPLETED":
                print(f"[{time.time()-t0:6.1f}s] >>> SUCCESS: Job completed! <<<", flush=True)
                break
            elif e_type in ("JOB_FAILED", "JOB_CANCELLED"):
                print(f"[{time.time()-t0:6.1f}s] >>> ERROR: Job ended with {e_type}: {data} <<<", flush=True)
                sys.exit(1)
        except queue.Empty:
            pass

    orch.stop()
    print(f"Finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
