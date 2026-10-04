# Experiment execution preferences

- Default to parallel execution for independent experiments, including Slurm
  array tasks, when they can run concurrently without conflicting writes.
- Do not default independent sweeps to a single active task (`%1`). Allow all
  tasks in the submitted comparison to run concurrently, subject to the user's
  explicit resource limits and scheduler availability.
- Use separate output/checkpoint directories per experiment. Keep steps with
  actual dependencies or shared mutable outputs sequential.
- State the GPU requirement per task and the maximum concurrent total when
  providing submission instructions.
