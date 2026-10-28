# What's running (this project only)

Kept current by Claude: updated whenever a process is started, stopped or paused.
Last updated: 2026-10-28 08:05 EDT

| PID | What | Started | Log | State |
|---|---|---|---|---|
| 714838 | Stage A worker: `python stage_a.py work` (cwd `model/`), GPU. Runs `research/stage_a/queue.jsonl`, now A2 families | 2026-10-27 22:34 | `data/stage_a_work.log` | running |
| 364317 | Audio lab web server: `python model/audio_lab.py --port 8765` (127.0.0.1), CPU | 2026-10-26 21:41 | none | running |

## Controls

* Pause / resume: `kill -STOP <pid>` / `kill -CONT <pid>` (a paused process keeps its GPU memory)
* Stop the worker cleanly: `kill -TERM 714838` (checkpoints at the next 500 steps, then exits; if it
  is paused, also `kill -CONT 714838`). Restart: `cd model && nohup ../.venv/bin/python stage_a.py work >> ../data/stage_a_work.log 2>&1 &`
* Progress: `tail data/stage_a_work.log`, or `python model/stage_a.py status`

Other Python processes on this machine (e.g. `run_grid.py` in data_poisoning_detection) are not
this project's.
