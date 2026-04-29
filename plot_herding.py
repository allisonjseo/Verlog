"""
Parse agent log files and plot second_vote_matches_first (herding) metric over training.

Usage: python plot_herding.py
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from datetime import datetime

LOG_FILES = {
    "KL=1e-2":  "logs/agent_model_train_2_16384_kl1e2_100.log",
    "KL=3e-3":  "logs/agent_model_train_2_16384_kl3e3_100.log",
    "combined": "logs/agent_model_train_4_16384_combined.log",
    "group":    "logs/train_4_16384_group/agent_model_train_4_16384_group.log",
    "train_4_16384": "logs/train_4_16384/agent_model_train_4_16384.log",
}

HEADER_RE = re.compile(
    r"^=== (\S+) env=(\d+) turn=(\d+) agent=(\S+) prompt_tokens=\d+ response_tokens=\d+ ===$"
)
VOTE_RE = re.compile(r"<VOTE>\s*(\d+)\s*</VOTE>")

ROLLING_WINDOW = 50


def parse_log(path):
    """
    Returns a list of (episode_start_ts, second_vote_matches_first) tuples,
    sorted by episode start time.

    second_vote_matches_first (strict: votes must be each professor's FIRST action):
        1  - second professor's first action was a vote matching the first professor's first-turn vote
        0  - second professor's first action was a vote for a different student
        -1 - fewer than 2 professors voted as their first action
    """
    # episode_key -> {"start_ts": datetime, "first_turn_votes": [(turn, choice)]}
    # first_turn_votes only includes votes that occurred on the agent's FIRST action in the episode
    # episode_key = (env_id, episode_index_for_that_env)
    episodes = {}
    env_episode_count = defaultdict(int)

    # per-episode set of agents who have already acted (to detect first turns)
    episode_seen_agents = defaultdict(set)

    # Track current entry state while scanning
    current_header = None
    in_output = False
    output_lines = []

    # per-env turn counter to detect episode boundary
    env_last_turn = {}

    def flush_output(header, output_text):
        """Process the collected [OUTPUT] block for a completed entry."""
        if header is None:
            return
        env_id = header["env"]
        ep_key = (env_id, header["episode_idx"])
        if ep_key not in episodes:
            episodes[ep_key] = {"start_ts": header["ts"], "first_turn_votes": []}

        agent = header["agent"]
        is_first_turn = agent not in episode_seen_agents[ep_key]
        episode_seen_agents[ep_key].add(agent)

        # Only record the vote if it's this agent's first action in the episode
        if is_first_turn:
            vote_match = VOTE_RE.search(output_text)
            if vote_match:
                choice = int(vote_match.group(1))
                episodes[ep_key]["first_turn_votes"].append((header["turn"], choice))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            m = HEADER_RE.match(line)
            if m:
                # Flush previous output
                flush_output(current_header, "\n".join(output_lines))
                output_lines = []
                in_output = False

                ts_str, env_id, turn, agent = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

                # Detect new episode for this env: turn resets to 0 (or is 0)
                prev_turn = env_last_turn.get(env_id, -1)
                if turn == 0 and prev_turn >= 0:
                    # New episode starting for this env
                    env_episode_count[env_id] += 1
                elif env_id not in env_last_turn:
                    pass  # First ever entry for this env, episode_count stays 0

                env_last_turn[env_id] = turn
                ep_idx = env_episode_count[env_id]

                current_header = {
                    "ts": ts,
                    "env": env_id,
                    "turn": turn,
                    "agent": agent,
                    "episode_idx": ep_idx,
                }
                continue

            if line.strip() == "[OUTPUT]":
                in_output = True
                continue

            if line.strip() == "[PROMPT]":
                in_output = False
                continue

            if in_output:
                output_lines.append(line)

    # Flush last entry
    flush_output(current_header, "\n".join(output_lines))

    # Compute metric for each episode
    results = []
    for ep_key, ep in episodes.items():
        # Only first-turn votes, sorted by turn number
        votes_sorted = sorted(ep["first_turn_votes"], key=lambda x: x[0])
        if len(votes_sorted) >= 2:
            match = int(votes_sorted[0][1] == votes_sorted[1][1])
        else:
            match = -1
        results.append((ep["start_ts"], match))

    # Sort by episode start time
    results.sort(key=lambda x: x[0])
    return results


def rolling_mean(values, window):
    """Compute rolling mean, ignoring -1 (insufficient votes)."""
    out = []
    for i in range(len(values)):
        window_vals = [v for v in values[max(0, i - window + 1):i + 1] if v >= 0]
        out.append(np.mean(window_vals) if window_vals else float("nan"))
    return out


def plot(all_results, out_path="herding_metric.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, results in all_results.items():
        matches = [r[1] for r in results]
        valid_episodes = [(i, v) for i, v in enumerate(matches) if v >= 0]
        if not valid_episodes:
            continue
        xs, ys = zip(*valid_episodes)
        rolling = rolling_mean(ys, ROLLING_WINDOW)

        # Scatter raw values (faint)
        ax.scatter(xs, ys, alpha=0.05, s=6)
        # Rolling average (solid)
        ax.plot(xs, rolling, label=label, linewidth=2)

    ax.set_xlabel("Episode index (chronological)")
    ax.set_ylabel(f"Second vote matches first  (rolling avg, w={ROLLING_WINDOW})")
    ax.set_title("Herding: does the 2nd professor's first action copy the 1st professor's first-turn vote?")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return out_path


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    all_results = {}
    for label, rel_path in LOG_FILES.items():
        full_path = os.path.join(base, rel_path)
        if not os.path.exists(full_path):
            print(f"WARNING: not found: {full_path}")
            continue
        print(f"Parsing {label}...")
        results = parse_log(full_path)
        valid = sum(1 for _, v in results if v >= 0)
        match_rate = np.mean([v for _, v in results if v >= 0]) if valid else float("nan")
        print(f"  {len(results)} episodes, {valid} with >=2 votes, match rate: {match_rate:.1%}")
        all_results[label] = results

    plot(all_results)


if __name__ == "__main__":
    main()
