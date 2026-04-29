# monitor/

Periodic `claude -p` monitor for verlog runs. Posts status to Slack, gates
mutating tool calls behind a Slack reaction ack via `claude-tools/slack/gate.sh`.

## Setup

```bash
# 1. From the repo root, init the claude-tools submodule:
git submodule update --init claude-tools

# 2. Configure (per host/user):
cp monitor/monitor.env.example monitor/monitor.env
# edit monitor.env — at minimum set CLUSTER_ID, SMOKETEST_DIR, USER_NAME,
# WANDB_PROJECT, CONDA_ENV_PATH

# 3. One-time-per-host setup for Slack token (see claude-tools/slack/README.md):
#    - drop xoxb token at $SLACK_TOKEN_FILE (chmod 600)
#    - invite the bot into your management channel

# 4. Launch:
bash monitor/launch_monitor.sh
```

## Files

| File | Purpose | Committed? |
|---|---|---|
| `verlog_monitor.sh` | the per-iteration `claude -p` loop | yes |
| `launch_monitor.sh` | tmux launcher; renders settings template at startup | yes |
| `monitor-settings.template.json` | `claude -p --settings` template (REPO_ROOT placeholder) | yes |
| `monitor.env.example` | example config | yes |
| `monitor.env` | live config (slack channel, paths, cluster id) | **no** (gitignored) |
| `_monitor-settings.json` | generated from the template at launch time | **no** (gitignored) |

## Where the verlog-specific bits live

The monitor's `PROMPT` references `${SMOKETEST_DIR}/logs/*.events.log`,
`${SMOKETEST_DIR}/ckpts/`, `${WANDB_PROJECT}`, etc. — all of those come from
`monitor.env`, so a different user can point this at a different scratch dir
and a different wandb project without editing the script.

## What lives in `claude-tools/` instead

Everything project-agnostic: `slack/slack_post.sh`, `slack/gate.sh`,
`slack/settings.template.json`, `slack/README.md`. The verlog monitor sources
`slack_post.sh` and points the hook at `gate.sh`.
