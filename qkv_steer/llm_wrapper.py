from __future__ import annotations

import torch

from qkv_steer._vendor.cam import attribution_field, load_detector
from qkv_steer._vendor.dataset import normalize
from qkv_steer._vendor.qkv_hooks import VIEWS, capture_all, qkv_hooks
from qkv_steer._vendor.run_extraction import build_prompt_ids, load_llm, resolve_stop_tokens, truncate_runon
from qkv_steer._vendor.tensor_ops import build_feature_field
from qkv_steer.hooks import register_steer_hooks, remove_hooks


class LLM_Wrapper:
    def __init__(self, cfg, device, detector_checkpoint: str | None = None):
        self.cfg = cfg
        self.device = device
        self.model, self.tokenizer = load_llm(cfg)
        self.stop_ids = resolve_stop_tokens(self.tokenizer, self.model)
        self.n_segments = cfg.extract.n_segments
        if self.n_segments is None:
            from qkv_steer._vendor.qkv_hooks import read_geometry
            self.n_segments = read_geometry(self.model).n_layers

        self.detector = None
        self.stats = None
        if detector_checkpoint is not None:
            self.detector, ckpt = load_detector(cfg, detector_checkpoint, device)
            self.stats = ckpt["stats"]

    def _generate_raw(self, prompt: str, max_new_tokens: int, steer_pushes=None):
        input_ids = build_prompt_ids(prompt, self.tokenizer, self.device)

        handles = []
        if steer_pushes:
            handles = register_steer_hooks(self.model, steer_pushes, self.n_segments)
        try:
            (activations, gen_ids), = capture_all(
                self.model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.stop_ids,
            )
        finally:
            remove_hooks(handles)

        kept_ids, kept_text = truncate_runon(gen_ids, self.tokenizer)
        n_kept = kept_ids.shape[0]
        activations = {v: activations[v][:n_kept] for v in VIEWS}
        return activations, kept_text

    def generate(self, prompt: str, max_new_tokens: int, steer_pushes=None) -> dict:
        activations, text = self._generate_raw(prompt, max_new_tokens, steer_pushes=steer_pushes)
        field = build_feature_field(activations, self.n_segments)
        return {"response": text, "field": field}

    def _to_detector_input(self, field: torch.Tensor):
        if self.stats is None:
            raise RuntimeError("detector not loaded -- pass detector_checkpoint to LLM_Wrapper")
        normed = normalize(field, self.stats)
        images = normed.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        mask = torch.ones(1, field.shape[0], dtype=torch.bool, device=self.device)
        return images, mask

    def detect(self, field: torch.Tensor) -> float:
        images, mask = self._to_detector_input(field)
        with torch.no_grad():
            logit = self.detector(images, mask)
        return torch.sigmoid(logit).item()

    def saliency(self, field: torch.Tensor) -> torch.Tensor:
        images, mask = self._to_detector_input(field)
        attribution, _ = attribution_field(
            self.detector, images, mask, signed=True, per_projection=True
        )
        return attribution
