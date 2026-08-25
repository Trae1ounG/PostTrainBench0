# Reference environment

PostTrainBench⁰ runs directly on a single Linux GPU host. It does not require a
cluster scheduler or a separate job-submission service. The Docker image pins
the evaluator stack used by the reference implementation:

| Component | Version |
|---|---|
| Operating system | Debian 12 compatible Linux |
| Python | 3.11 |
| PyTorch | 2.6.0 with CUDA 12.4 wheels |
| vLLM | 0.8.5, V0 engine (`VLLM_USE_V1=0`) |
| Transformers | 4.56.0 |
| NumPy | 1.26.4 |
| Datasets | 4.2.0 |
| PyArrow | 25.0.1 |

The host only needs a compatible NVIDIA driver, the model weights, the Agent
CLI, and enough visible GPUs for the configured run. The launcher starts one
local controller process; that process schedules candidate evaluations across
the GPUs exposed to the container.

## Two permission boundaries

The trusted controller and the Agent do not run with the same permissions.

1. The trusted controller owns vLLM and GPU access. It can read task data and
   write append-only attempts and the final checkpoint.
2. Bubblewrap starts Agent-authored commands in a new mount and network
   namespace. The Agent sees `/home/agent` read/write and `/models/base`
   read-only. It does not see task data, scorers, trusted logs, credentials, or
   GPU devices. Python and installed packages are available read-only.

The model API CLI runs in a separate networked control process. Commands
requested by the Agent are executed inside the network-disabled Agent shell.

## Docker scope

The image contains the Python/evaluator stack and the exact evaluation
snapshot. Model weights, Agent credentials, the Agent CLI, and run outputs stay
as external mounts. Docker must already be able to see the requested NVIDIA
GPUs; `docker/run.sh` then launches the benchmark directly.
