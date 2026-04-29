# Vote Log Probability - Clean Architecture Design

## Principle: Keep VERL Generic

VERL should remain environment-agnostic and simply provide generic rollout metadata that any environment can optionally use.

## Clean Interface Design

### Option 1: Extended Action Dict (RECOMMENDED)

**VERL Side**: Pass action as a dict with optional metadata
```python
action = {
    "text": "<THINK>...</THINK><VOTE>2</VOTE>",  # The actual action text
    "metadata": {  # Optional: any environment can ignore this
        "response_ids": [1234, 5678, ...],
        "response_logprobs": [-0.5, -1.2, ...],
        "response_tokens": ["<", "THINK", ">", ...],
        "top_logprobs": [  # Top alternatives for each position
            [{"token": "<", "logprob": -0.5}, ...],
            [{"token": "THINK", "logprob": -1.2}, ...],
            ...
        ]
    }
}
```

**Environment Side**: Extract what it needs
```python
def step(self, action):
    if isinstance(action, dict) and "metadata" in action:
        metadata = action["metadata"]
        action_text = action.get("text", action.get("action_text", ""))
        # Environment-specific: parse metadata if needed
        self._process_rollout_metadata(metadata, action_text)
    else:
        action_text = action  # Backward compatible: string action

    # Continue with normal step logic...
```

**Benefits**:
- ✅ VERL stays generic - just passes data
- ✅ No new methods or state management
- ✅ Backward compatible - environments can ignore metadata
- ✅ Any environment can use rollout metadata however it wants
- ✅ Single call to step() - no coordination needed

---

## Implementation

### VERL Changes (Minimal & Generic)

**File**: `verl/experimental/agent_loop/tool_agent_loop.py`

**Change**: Around line 350-360, when calling `env.step()`

```python
# BEFORE:
observations, rewards, terminations, truncations, infos = env.step(
    {active_agent: response_text}
)

# AFTER:
# Build action with optional metadata
action_dict = {
    "text": response_text,
}

# Optionally include rollout metadata if available
if output.log_probs:
    action_dict["metadata"] = {
        "response_ids": response_ids,
        "response_logprobs": response_logprobs,
        "response_tokens": [
            self.tokenizer.decode([tid]) for tid in response_ids
        ],
    }

    # Include top logprobs if available (vLLM provides this)
    if hasattr(output, 'top_log_probs') and output.top_log_probs:
        action_dict["metadata"]["top_logprobs"] = output.top_log_probs

observations, rewards, terminations, truncations, infos = env.step(
    {active_agent: action_dict}
)
```

**That's it!** VERL just passes more data. It doesn't know or care what the environment does with it.

---

### Hiring Environment Changes

**File**: `verl/envs/hiring_env/env.py`

**1. Update step() to handle both formats:**

```python
def step(self, action: str | Dict[str, Any]) -> Tuple[...]:
    active_agent = self.episode_state["active_agent"]

    # Extract action text and optional metadata
    action_text = ""
    rollout_metadata = None

    if isinstance(action, dict):
        # Check if this is the new format with metadata
        if "text" in action:
            action_text = action["text"]
            rollout_metadata = action.get("metadata")
        # Handle multi-agent dict: {agent_id: action_data}
        elif active_agent in action:
            agent_action = action[active_agent]
            if isinstance(agent_action, dict) and "text" in agent_action:
                action_text = agent_action["text"]
                rollout_metadata = agent_action.get("metadata")
            else:
                action_text = agent_action
        else:
            # Fallback: treat dict values as action text
            action_text = next(iter(action.values())) if action else ""
    else:
        # String action (backward compatible)
        action_text = action

    # Parse action
    parsed = self._parse_action(action_text)

    # Process metadata if this is a VOTE action
    if parsed["type"] == "vote" and rollout_metadata is not None:
        self._process_vote_logprobs(
            agent_id=active_agent,
            vote_choice=parsed["choice"],
            metadata=rollout_metadata
        )

    # Continue with rest of step logic...
```

**2. Add metadata processing method:**

```python
def _process_vote_logprobs(
    self,
    agent_id: str,
    vote_choice: int,
    metadata: Dict[str, Any]
):
    """
    Process rollout metadata to extract vote preference probabilities.

    This method is hiring-environment-specific and parses the generic
    rollout metadata to extract vote-related metrics.
    """
    from .vote_logprob_utils import (
        extract_vote_token_index,
        categorize_vote_logprobs
    )

    # Extract data from generic metadata
    response_ids = metadata.get("response_ids", [])
    response_logprobs = metadata.get("response_logprobs", [])
    response_tokens = metadata.get("response_tokens", [])
    top_logprobs = metadata.get("top_logprobs", [])

    if not response_ids or not response_logprobs:
        return  # No logprob data available

    # Find vote token index
    full_text = "".join(response_tokens)
    vote_token_idx = extract_vote_token_index(full_text, response_tokens)

    if vote_token_idx is None:
        return  # Couldn't find vote token

    # Categorize probabilities
    vote_prefs = categorize_vote_logprobs(
        token_logprobs=response_logprobs,
        top_logprobs=top_logprobs,
        vote_token_idx=vote_token_idx,
        num_students=self.students_per_batch,
        tokenizer=self.tokenizer
    )

    # Store for episode metrics
    self.vote_logprob_data.append({
        'agent_id': agent_id,
        'actual_choice': vote_choice,
        'turn': self.episode_state["step_count"],
        **vote_prefs
    })
```

**3. Add to episode metrics** (already designed in vote_logprob_utils.py):

```python
def _calculate_episode_metrics(self, agent_rewards):
    # ... existing metrics ...

    # Add vote preference analysis
    if self.vote_logprob_data:
        from .vote_logprob_utils import aggregate_episode_vote_preferences
        metrics["vote_preferences"] = aggregate_episode_vote_preferences(
            self.vote_logprob_data
        )

    return metrics
```

---

## Summary of Changes

### VERL (Generic - Works for ANY Environment)

**One file**: `verl/experimental/agent_loop/tool_agent_loop.py`

**One change**: Pass action as dict with optional metadata instead of just string

```python
# 5 lines added
action_dict = {"text": response_text}
if output.log_probs:
    action_dict["metadata"] = {
        "response_ids": response_ids,
        "response_logprobs": response_logprobs,
        "response_tokens": [self.tokenizer.decode([tid]) for tid in response_ids],
    }
```

### Hiring Environment (Environment-Specific)

**Files modified**:
1. `env.py`: Update `step()` to extract metadata, add `_process_vote_logprobs()`
2. `vote_logprob_utils.py`: Already created - utility functions

**Lines added**: ~50 lines total

---

## Benefits of This Design

### For VERL:
- ✅ Minimal changes (~5 lines)
- ✅ Stays completely generic
- ✅ No knowledge of what environments do with metadata
- ✅ No coupling to hiring environment
- ✅ Backward compatible

### For Environments:
- ✅ Optional - can ignore metadata completely
- ✅ Each environment interprets metadata however it wants
- ✅ Hiring env can extract vote probabilities
- ✅ Other envs could use for different purposes (e.g., tool choice probabilities)
- ✅ Clean separation of concerns

### For Future Extensibility:
- ✅ Easy to add more metadata fields
- ✅ Easy to create other environment-specific loggers
- ✅ Could create generic logprob analyzers that work across environments
- ✅ No technical debt or coupling

---

## Alternative: Environment Wrapper (If Even Cleaner Interface Wanted)

If you want to keep `step()` signature completely unchanged, you could use a wrapper:

```python
class LogProbTrackingWrapper(gym.Wrapper):
    """Wrapper that extracts and stores rollout metadata."""

    def __init__(self, env):
        super().__init__(env)
        self._pending_metadata = None

    def step(self, action):
        # Extract metadata if present
        if isinstance(action, dict) and "metadata" in action:
            self._pending_metadata = action["metadata"]
            action_text = action.get("text", action)
        else:
            action_text = action

        # Let environment process - it can access self._pending_metadata
        # if it knows about this wrapper
        obs, rew, term, trunc, info = self.env.step(action_text)

        self._pending_metadata = None
        return obs, rew, term, trunc, info
```

But I think the extended action dict is cleaner and more explicit.

---

## Recommendation

**Go with Option 1 (Extended Action Dict)** because:
1. VERL changes are minimal and generic
2. No new wrapper infrastructure needed
3. Clear, explicit interface
4. Backward compatible
5. Environment has full control over how to use metadata
