# Running Coding Agents with SWE-bench

The `swebench/inference/run_agent.py` script allows you to run local coding agents on SWE-bench tasks inside the same Docker containers used for evaluation.

## Usage

```bash
python -m swebench.inference.run_agent \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split test \
    --agent_command "claude-code --problem-statement-file \$PROBLEM_STATEMENT" \
    --output_dir ./agent_results \
    --max_workers 1
```

## Key Features

- **Docker Integration**: Runs agents inside the same Docker containers used for SWE-bench evaluation
- **Problem Statement Parameterization**: Uses `$PROBLEM_STATEMENT` placeholder in commands
- **Output Collection**: Captures both agent output and git diff of changes made
- **Resume Capability**: Can resume from where it left off if interrupted
- **Parallel Processing**: Supports running multiple instances in parallel

## Parameters

- `--dataset_name`: SWE-bench dataset to use (e.g., "SWE-bench/SWE-bench_Lite")
- `--split`: Dataset split ("test", "dev", etc.)
- `--agent_command`: Command template with `$PROBLEM_STATEMENT` placeholder
- `--output_dir`: Directory to save results
- `--max_workers`: Number of parallel workers (default: 1)
- `--timeout`: Timeout per instance in seconds (default: 1800)
- `--force_rebuild`: Force rebuild Docker images
- `--instance_ids`: Specific instances to run (optional)

## Agent Command Examples

### Claude Code
```bash
--agent_command "claude-code --problem-statement-file \$PROBLEM_STATEMENT"
```

### Custom Script
```bash
--agent_command "python /path/to/my_agent.py --problem \$PROBLEM_STATEMENT"
```

### With Additional Options  
```bash
--agent_command "my-agent --input \$PROBLEM_STATEMENT --max-tokens 4000 --temperature 0.1"
```

## Output Format

Results are written in JSONL format compatible with SWE-bench evaluation:

```json
{
    "instance_id": "repo__issue-123",
    "model_patch": "diff --git a/file.py b/file.py\n...",
    "model_name_or_path": "agent",
    "agent_command": "claude-code --problem-statement-file /tmp/problem_statement.txt",
    "agent_output": "Agent's stdout/stderr output...",
    "git_diff": "Git diff of changes made..."
}
```

## Docker Container Behavior

1. **Container Creation**: Uses the same Docker images as SWE-bench evaluation
2. **Problem Statement**: Written to `/tmp/problem_statement.txt` inside container
3. **Working Directory**: Agent runs in the repository root (`/testbed` typically)
4. **Change Detection**: Git diff captures all changes made by the agent
5. **Cleanup**: Containers are automatically stopped and removed after each run

## Example Workflow

1. **Run Agent**: Execute coding agent on dataset instances
```bash
python -m swebench.inference.run_agent \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split test \
    --agent_command "claude-code --problem-statement-file \$PROBLEM_STATEMENT" \
    --output_dir ./results
```

2. **Evaluate Results**: Use standard SWE-bench evaluation
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path ./results/agent__SWE-bench_Lite__test.jsonl \
    --max_workers 4 \
    --run_id agent_evaluation
```

## Tips

- Start with `--max_workers 1` to avoid resource contention
- Use shorter timeouts for faster agents, longer for complex ones
- The `$PROBLEM_STATEMENT` file contains the full GitHub issue description
- Agent should modify files in the current working directory
- Changes are automatically captured via git diff
- Test with a few instances first using `--instance_ids`

