from __future__ import annotations

from utils.worker_extn import WorkerExtension


class PostTrainBenchWorker(WorkerExtension):
    """RandOPT worker with exact base restoration and checkpoint replay."""

    def store_base_weights(self) -> bool:
        import torch

        self._base_weights = {}
        self._base_weights_cpu = {}
        with torch.no_grad():
            for name, parameter in self.model_runner.model.named_parameters():
                anchor = parameter.detach().clone()
                self._base_weights[name] = anchor
                self._base_weights_cpu[name] = anchor.cpu().clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def reset_to_base_weights(self) -> bool:
        import torch

        if not hasattr(self, "_base_weights"):
            raise RuntimeError("base weights have not been stored")
        with torch.no_grad():
            for name, parameter in self.model_runner.model.named_parameters():
                anchor = self._base_weights[name]
                anchor.copy_(self._base_weights_cpu[name])
                parameter.copy_(anchor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def apply_noise_program(self, terms: list[dict]) -> bool:
        self.reset_to_base_weights()
        for term in terms:
            self.perturb_self_weights(int(term["seed"]), float(term["scale"]), False)
        return True

    def save_posttrainbench0_checkpoint(self, path: str) -> dict:
        import torch

        state = {name: parameter.detach().cpu() for name, parameter in self.model_runner.model.named_parameters()}
        torch.save(state, path)
        return {"parameter_count": len(state), "tensor_elements": sum(tensor.numel() for tensor in state.values())}

    def load_posttrainbench0_checkpoint(self, path: str) -> dict:
        import torch

        saved = torch.load(path, map_location="cpu", weights_only=True)
        current = dict(self.model_runner.model.named_parameters())
        if set(saved) != set(current):
            raise ValueError("checkpoint parameter names do not match the base model")
        with torch.no_grad():
            for name, parameter in current.items():
                tensor = saved[name]
                if tensor.shape != parameter.shape:
                    raise ValueError(f"shape mismatch for {name}")
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"parameter_count": len(saved), "tensor_elements": sum(tensor.numel() for tensor in saved.values())}
