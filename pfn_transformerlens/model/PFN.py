from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformer_lens import ActivationCache, HookedTransformer

from .bucketizer import Bucketizer
from .configs import (
    BasePFNConfig,
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from .PFNMasks import create_custom_mask_hook


@dataclass
class DistributionPrediction:
    """Prediction output for distributional models.

    Attributes:
        probs: Probabilities over buckets for each position.
        y_grid: Bucket representative values (midpoints). These are reference
            points for the mode/expectation, not the full continuous sample space.
            Actual samples from generate() will be uniformly distributed within
            buckets (and in half-normal tails for unbounded support).
        logits: Optional raw logits over buckets.
    """

    probs: Float[torch.Tensor, "... seq d_vocab"]
    y_grid: Float[torch.Tensor, " d_vocab"]
    logits: Optional[Float[torch.Tensor, "... seq d_vocab"]] = None


@dataclass
class PointPrediction:
    preds: Float[torch.Tensor, "... seq 1"]


@dataclass
class ClassificationPrediction:
    # TODO: fix!
    probs: Float[torch.Tensor, "... seq num_classes"]
    logits: Optional[Float[torch.Tensor, "... seq num_classes"]] = None


class BasePFN(nn.Module, ABC):
    """Abstract base class for PFN models with shared functionality.

    Device Placement
    ----------------
    To move the model to a different device (GPU/MPS/CPU), always use:
        model = model.to(device)  # Correct - moves all components

    Do NOT use:
        model.transformer = model.transformer.to(device)  # Wrong - leaves input_proj behind

    The model will raise a helpful error if it detects a device mismatch between
    components during forward passes.
    """

    transformer: HookedTransformer
    input_proj: nn.Module
    _bucketizer: Optional[Bucketizer]

    def __init__(self, config: BasePFNConfig):
        super().__init__()
        self.config = config
        self._bucketizer = None
        self._setup_bucketizer()
        self._setup_input_proj()
        self.transformer = HookedTransformer(config)

        # move input_proj to same device as transformer at initialization
        if hasattr(self, "input_proj"):
            self.input_proj = self.input_proj.to(self.transformer.cfg.device)
        if hasattr(self, "x_proj"):
            self.x_proj = self.x_proj.to(self.transformer.cfg.device)
        if hasattr(self, "y_embed"):
            self.y_embed = self.y_embed.to(self.transformer.cfg.device)

        if not config.use_pos_emb:
            self.transformer.W_pos.requires_grad = False
            self.transformer.W_pos.data.zero_()

    def _setup_bucketizer(self) -> None:
        """Initialize bucketizer if needed for distribution predictions."""
        if isinstance(self.config, SupervisedRegressionPFNConfig):
            if self.config.prediction_type == "distribution":
                self._bucketizer = Bucketizer.from_config(self.config)
        elif isinstance(self.config, UnsupervisedPFNConfig):
            if (
                self.config.input_type == "continuous"
                and self.config.prediction_type == "distribution"
            ):
                self._bucketizer = Bucketizer.from_config(self.config)

    @abstractmethod
    def _setup_input_proj(self) -> None:
        """Setup input projection layer (Embedding or Linear)."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        pass

    def get_bucket_values(
        self,
        y: Float[torch.Tensor, "batch seq"],
    ) -> Int[torch.Tensor, "batch seq"]:
        """Map targets into discrete bucket indices via the configured bucketizer."""
        return self.bucketizer.bucketize(y)

    def get_y_values(
        self,
        bucket_indices: Int[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq"]:
        """Convert bucket indices back to their representative continuous y values."""
        return self.bucketizer.decode(bucket_indices)

    def log_bucket_densities(
        self,
        logits: Float[torch.Tensor, "... d_vocab"],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        """Scale logits by bucket widths to obtain piecewise-constant log densities."""
        return self.bucketizer.log_bucket_densities(logits)

    @property
    def bucketizer(self) -> Bucketizer:
        if self._bucketizer is None:
            raise RuntimeError(
                "Bucketizer is only available when prediction_type='distribution'."
            )
        return self._bucketizer

    def _prepare_transformer_input(
        self,
        hidden: Float[torch.Tensor, "batch seq d_model"],
    ) -> tuple[
        Float[torch.Tensor, "batch seq d_model"],
        Float[torch.Tensor, "batch seq d_model"] | None,
    ]:
        """Mimic HookedTransformer.input_to_embed when bypassing internal embeddings.

        We construct the residual stream (token embedding plus positional information)
        because we call HookedTransformer with start_at_layer=0 and therefore skip the
        built-in embedding logic. This keeps positional handling consistent across
        discrete and continuous inputs.
        """
        residual = self.transformer.hook_embed(hidden)

        if not getattr(self.config, "use_pos_emb", True):
            return residual, None

        pos_type = self.transformer.cfg.positional_embedding_type
        if pos_type not in {"standard", "shortformer", "rotary", "alibi"}:
            raise ValueError(
                f"Unsupported positional_embedding_type='{pos_type}' for PFN models."
            )

        seq_len = residual.shape[1]
        if seq_len > self.transformer.cfg.n_ctx:
            raise ValueError(
                f"sequence length {seq_len} exceeds model context "
                f"{self.transformer.cfg.n_ctx}"
            )

        shortformer_pos_embed: Float[torch.Tensor, "batch seq d_model"] | None = None

        if pos_type in {"standard", "shortformer"}:
            pos = self.transformer.W_pos[:seq_len, :].to(
                device=residual.device, dtype=residual.dtype
            )
            pos = pos.unsqueeze(0)
            if residual.shape[0] != 1:
                pos = pos.expand(residual.shape[0], -1, -1)
            pos = self.transformer.hook_pos_embed(pos)

            if pos_type == "standard":
                residual = residual + pos
            else:  # shortformer
                shortformer_pos_embed = pos

        # rotary and alibi are applied inside the transformer blocks, so nothing to add here.
        return residual, shortformer_pos_embed

    @abstractmethod
    def predict_on_prompt(
        self, *args, **kwargs
    ) -> (
        DistributionPrediction
        | PointPrediction
        | ClassificationPrediction
        | Tuple[
            DistributionPrediction | PointPrediction | ClassificationPrediction,
            ActivationCache,
        ]
    ):
        """Return predictions aligned with the configured prediction_type.

        For distributional configs, returns bucket probabilities (optionally logits) and the
        continuous y-grid. For point configs, returns direct regression estimates.
        For classification configs, returns class probabilities.
        Inputs may be provided with or without an explicit batch dimension.

        Supports optional cache return for mechanistic interpretability analysis.
        """
        pass


class UnsupervisedPFN(BasePFN):
    """Unsupervised PFN for next-token prediction without x/y interleaving.

    Standard GPT-2 style transformer for sequence modeling: p(x*|x_1:n).
    Supports both discrete and continuous inputs with point or distributional predictions.
    """

    config: UnsupervisedPFNConfig
    input_proj: nn.Embedding | nn.Linear

    def __init__(self, config: UnsupervisedPFNConfig):
        super().__init__(config)
        if self.config.input_type == "discrete":
            # share weights with transformer embedding so gradients accumulate
            self.input_proj.weight = self.transformer.embed.W_E

    def _setup_input_proj(self) -> None:
        """Setup input projection based on input type."""
        if self.config.input_type == "discrete":
            self.input_proj = nn.Embedding(self.config.d_vocab, self.config.d_model)
        else:
            self.input_proj = nn.Linear(1, self.config.d_model)

    def forward(
        self,
        y: Float[torch.Tensor, "batch seq"],
        return_cache: bool = False,
    ) -> (
        Float[torch.Tensor, "batch seq d_vocab"]
        | Tuple[Float[torch.Tensor, "batch seq d_vocab"], ActivationCache]
    ):
        """Forward pass for unsupervised sequence modeling.

        No x/y interleaving - just process the sequence y directly with causal masking.
        This is for approximating p(x*|x_1:n) where the entire sequence is observations.

        Parameters
        ----------
        y : Float[torch.Tensor, "batch seq"]
            Sequence tokens (discrete indices or continuous values).
        return_cache : bool
            Whether to return activation cache.

        Returns
        -------
        Float[torch.Tensor, "batch seq d_vocab"]
            Logits for next token prediction at each position.
            logits[:, t, :] = distribution over next token given y[:, :t+1].
        """
        batch_size, seq_len = y.shape

        if self.config.input_type == "discrete":
            y_long = y.long()
            if return_cache:
                logits, cache = self.transformer.run_with_cache(
                    y_long,
                    return_type="logits",
                )
                return logits, cache
            logits = self.transformer(
                y_long,
                return_type="logits",
            )
            return logits

        else:
            y_reshaped = y.unsqueeze(-1)
            hidden = self.input_proj(y_reshaped)

        residual, shortformer_pos_embed = self._prepare_transformer_input(hidden)

        if return_cache:
            logits, cache = self.transformer.run_with_cache(
                residual,
                start_at_layer=0,
                return_type="logits",
                shortformer_pos_embed=shortformer_pos_embed,
            )
            return logits, cache
        else:
            logits = self.transformer(
                residual,
                start_at_layer=0,
                return_type="logits",
                shortformer_pos_embed=shortformer_pos_embed,
            )
            return logits

    @torch.no_grad()
    def generate(
        self,
        num_generate: int,
        prompt: Float[torch.Tensor, "K_init"] | None = None,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> Float[torch.Tensor, "total_len"]:
    # TODO: add return logits option? perhaps even cache too?
        """Generate sequence autoregressively.

        Standard GPT-2 style generation where the model predicts the next token
        given all previous tokens. The model's own predictions feed back as context.

        Parameters
        ----------
        num_generate : int
            Number of new tokens to generate.
        prompt : Float[torch.Tensor, "K_init"] | None
            Initial context tokens. If None, generation starts from scratch.
        sample : bool, default True
            Whether to sample from predicted distribution or take mode. For
            continuous distributions, sampling uses :meth:`Bucketizer.sample`
            (inverse-CDF within buckets). For point predictions, this parameter
            is ignored.
        temperature : float, default 1.0
            Sampling temperature for distribution predictions.

        Returns
        -------
        Float[torch.Tensor, "total_len"]
            Generated sequence including the prompt.
            total_len = K_init + num_generate

        Raises
        ------
        ValueError
            If temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        # initialize context with prompt or empty
        if prompt is not None:
            context = prompt.to(device)
        else:
            context = torch.empty(0, device=device)

        # generate num_generate new tokens
        for _ in range(num_generate):
            # run forward pass on current sequence
            if context.shape[0] == 0:
                # special case: empty context, predict first token
                # use a dummy token as input (will be ignored in practice)
                context_batch = torch.zeros(1, 1, device=device)
            else:
                context_batch = context.unsqueeze(0)  # (1, seq)

            logits = self(context_batch)  # (1, seq, d_vocab)

            # get prediction for next token at last position
            logits_last = logits[0, -1, :]  # (d_vocab,)

            # sample or take mode based on prediction type
            if self.config.prediction_type == "distribution":
                if sample:
                    if self.config.input_type == "discrete":
                        probs = F.softmax(logits_last / temperature, dim=-1)
                        token_new = torch.multinomial(probs, 1).squeeze(-1)
                        token_new = token_new.float()
                    else:  # continuous with bucketing
                        # sample continuously within buckets using inverse CDF
                        token_new = self.bucketizer.sample(
                            logits_last.unsqueeze(0), temperature
                        )[0]
                else:
                    probs = F.softmax(logits_last / temperature, dim=-1)
                    if self.config.input_type == "discrete":
                        token_new = torch.argmax(probs, dim=-1).float()
                    else:  # continuous with bucketing
                        # deterministic: use bucket midpoint (mode of uniform within bucket)
                        bucket_idx = torch.argmax(probs, dim=-1)
                        token_new = self.get_y_values(bucket_idx.unsqueeze(0))[0]

            else:  # point prediction
                # logits is the predicted value
                token_new = logits_last[0]  # point predictions have d_vocab_out=1

            # add to context
            if context.shape[0] == 0:
                context = token_new.unsqueeze(0)
            else:
                context = torch.cat(
                    [context, token_new.to(context.dtype).unsqueeze(0)],
                    dim=0,
                )

        if was_training:
            self.train()

        return context

    @torch.no_grad()
    def predict_on_prompt(
        self,
        y: Float[torch.Tensor, "... seq"],
        *,
        temperature: float = 1.0,
        return_logits: bool = False,
        return_cache: bool = False,
    ) -> (
        DistributionPrediction
        | PointPrediction
        | Tuple[DistributionPrediction | PointPrediction, ActivationCache]
    ):
        """Return predictions aligned with the configured prediction_type.

        For distributional configs, returns bucket probabilities (optionally logits)
        and the continuous y-grid. The `y_grid` contains bucket representatives
        (midpoints) suitable for inspecting modes/expectations; sampling continuous
        values should instead use :meth:`generate` or :meth:`Bucketizer.sample`.
        For point configs, returns direct regression estimates. Inputs may be
        provided with or without an explicit batch dimension.

        Parameters
        ----------
        y : Float[torch.Tensor, "... seq"]
            Input sequence tokens.
        temperature : float, default 1.0
            Temperature for softmax (distribution predictions only).
        return_logits : bool, default False
            Whether to include raw logits in the prediction output.
        return_cache : bool, default False
            Whether to return the transformer activation cache.
            If True, returns tuple of (prediction, cache).

        Returns
        -------
        DistributionPrediction | PointPrediction | Tuple
            Prediction results, optionally with activation cache.
        """
        was_training = self.training
        self.eval()

        squeeze_batch = y.dim() == 1
        y_b = y.unsqueeze(0) if squeeze_batch else y

        device = next(self.parameters()).device
        y_b = y_b.to(device)

        if return_cache:
            logits, cache = self(y_b, return_cache=True)
        else:
            logits = self(y_b)

        if self.config.prediction_type == "distribution":
            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            probs = F.softmax(logits / temperature, dim=-1)

            if self.config.input_type == "continuous":
                bucket_indices = torch.arange(self.config.d_vocab, device=device)
                y_grid = self.get_y_values(bucket_indices).detach().cpu()
            else:
                y_grid = torch.arange(self.config.d_vocab, dtype=torch.float32)

            probs_out = probs[0] if squeeze_batch else probs
            logits_out = (
                (logits[0] if squeeze_batch else logits) if return_logits else None
            )
            result = DistributionPrediction(
                probs=probs_out.detach().cpu(),
                y_grid=y_grid,
                logits=logits_out.detach().cpu() if logits_out is not None else None,
            )
            if was_training:
                self.train()
            return (result, cache) if return_cache else result

        preds_out = logits[0] if squeeze_batch else logits
        result_point = PointPrediction(
            preds=preds_out.detach().cpu(),
        )
        if was_training:
            self.train()
        return (result_point, cache) if return_cache else result_point


class SupervisedPFN(BasePFN):
    """Supervised PFN with x/y interleaving and custom attention masks.

    Supports both autoregressive-pfn and gpt2 mask types for in-context learning.
    """

    config: SupervisedRegressionPFNConfig | ClassificationPFNConfig
    input_proj: nn.Linear
    x_proj: nn.Linear
    y_embed: nn.Embedding

    def __init__(self, config: SupervisedRegressionPFNConfig | ClassificationPFNConfig):
        super().__init__(config)

    def _setup_input_proj(self) -> None:
        """Setup input projection for x/y tokens."""
        if (
            isinstance(self.config, ClassificationPFNConfig)
            and self.config.y_type == "categorical"
        ):
            # separate embeddings for continuous x and categorical y
            self.x_proj = nn.Linear(self.config.input_dim, self.config.d_model)
            self.y_embed = nn.Embedding(self.config.num_classes, self.config.d_model)
        else:
            # continuous y: use concatenated projection
            self.input_proj = nn.Linear(self.config.input_dim + 1, self.config.d_model)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq input_dim"],
        y: Float[torch.Tensor, "batch seq"],
        return_cache: bool = False,
    ) -> (
        Float[torch.Tensor, "batch seq d_vocab"]
        | Tuple[Float[torch.Tensor, "batch seq d_vocab"], ActivationCache]
    ):
        """Forward pass through the supervised PFN model.

        Processes input features and targets through the transformer with the configured
        attention masking strategy. The model interleaves x and y tokens, applies the
        appropriate attention mask, and returns logits for the next y predictions.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch seq input_dim"]
            Input feature tokens for each training example.
        y : Float[torch.Tensor, "batch seq"]
            Target values corresponding to each input token.
        return_cache : bool, default False
            Whether to return the activation cache from the transformer.
            If True, returns a tuple of (logits, cache).

        Returns
        -------
        Float[torch.Tensor, "batch seq d_vocab"] or Tuple
            Logits for next y predictions, optionally with activation cache.
            The logits are projected from the transformer output to vocabulary size.

        Notes
        -----
        The model supports two masking strategies:
        - "autoregressive-pfn": Custom PFN attention mask for in-context learning
        - "gpt2": Standard causal attention mask

        The input is automatically interleaved as (x1, y1, x2, y2, ...) before
        being passed to the transformer.
        """
        if self.config.mask_type == "autoregressive-pfn":
            return self._forward_autoregressive_pfn(x, y, return_cache)
        elif self.config.mask_type == "gpt2":
            return self._forward_gpt2(x, y, return_cache)
        else:
            raise ValueError(f"Invalid mask type: {self.config.mask_type}")

    def output_proj(
        self, logits: Float[torch.Tensor, "batch 2*seq d_vocab"]
    ) -> Float[torch.Tensor, "batch seq d_vocab"]:
        """Project transformer logits to vocabulary logits (de-interleave)."""
        return logits[:, ::2, :]  # batch, odd indices, d_vocab

    @torch.no_grad()
    def generate(
        self,
        x_distribution: torch.distributions.Distribution,
        num_generate: int,
        prompt_x: Float[torch.Tensor, "K_init input_dim"] | None = None,
        prompt_y: Float[torch.Tensor, "K_init"] | None = None,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[
        Float[torch.Tensor, "total_len input_dim"], Float[torch.Tensor, "total_len"]
    ]:
        """Generate (x, y) pairs autoregressively.

        The model generates y values conditioned on x, where x values are sampled from
        x_distribution and y values are predicted by the model. The model's own y
        predictions feed back into the context for future predictions.

        Parameters
        ----------
        x_distribution : Distribution
            Distribution to sample x values from.
        num_generate : int
            Number of new (x, y) pairs to generate.
        prompt_x : Float[torch.Tensor, "K_init input_dim"] | None
            Initial x context. If None, generation starts from scratch.
        prompt_y : Float[torch.Tensor, "K_init"] | None
            Initial y context. Must be provided if prompt_x is provided.
        sample : bool, default True
            Whether to sample from predicted distribution or take mode. For
            continuous distributions, sampling uses :meth:`Bucketizer.sample`
            (inverse-CDF within buckets). For point predictions, this parameter
            is ignored.
        temperature : float, default 1.0
            Sampling temperature for distribution predictions.

        Returns
        -------
        tuple[Float[torch.Tensor, "total_len input_dim"], Float[torch.Tensor, "total_len"]]
            Generated (x, y) sequences including the prompt.
            total_len = K_init + num_generate

        Raises
        ------
        ValueError
            If temperature <= 0.
        AssertionError
            If prompt_x and prompt_y have mismatched lengths.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        # initialize context with prompt or empty
        if prompt_x is not None and prompt_y is not None:
            assert prompt_x.shape[0] == prompt_y.shape[0], (
                f"prompt_x and prompt_y must have same length, "
                f"got {prompt_x.shape[0]} and {prompt_y.shape[0]}"
            )
            x_context = prompt_x.to(device)
            y_context = prompt_y.to(device)
        elif prompt_x is None and prompt_y is None:
            x_context = torch.empty(0, self.config.input_dim, device=device)
            y_context = torch.empty(0, device=device)
        else:
            raise ValueError("Both prompt_x and prompt_y must be provided or both None")

        # generate num_generate new pairs
        for _ in range(num_generate):
            # sample new x
            x_new = x_distribution.sample((1, self.config.input_dim)).to(device)

            # build current context: all previous (x,y) pairs + new x
            # we need to predict y for the new x
            x_full = torch.cat([x_context, x_new], dim=0)

            # for prediction, we need a y placeholder at the last position
            # the model will predict what this should be
            # we use a dummy value (0.0) since it will be masked out
            y_dummy = torch.zeros(1, device=device)
            y_full = torch.cat([y_context, y_dummy], dim=0)

            # run forward pass (batch size 1)
            x_batch = x_full.unsqueeze(0)  # (1, seq, input_dim)
            y_batch = y_full.unsqueeze(0)  # (1, seq)

            logits = self(x_batch, y_batch)  # (1, seq, d_vocab)

            # get prediction for the last position
            logits_last = logits[0, -1, :]  # (d_vocab,)

            # sample or take mode based on prediction type
            if isinstance(self.config, ClassificationPFNConfig):
                probs = F.softmax(logits_last / temperature, dim=-1)
                if sample:
                    y_new = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    y_new = torch.argmax(probs, dim=-1)
                y_new = y_new.float()  # convert to float for consistency

            elif (
                isinstance(self.config, SupervisedRegressionPFNConfig)
                and self.config.prediction_type == "distribution"
            ):
                if sample:
                    # sample continuously within buckets using inverse CDF
                    y_new = self.bucketizer.sample(
                        logits_last.unsqueeze(0), temperature
                    )[0]
                else:
                    # deterministic: use bucket midpoint (mode of uniform within bucket)
                    probs = F.softmax(logits_last / temperature, dim=-1)
                    bucket_idx = torch.argmax(probs, dim=-1)
                    y_new = self.get_y_values(bucket_idx.unsqueeze(0))[0]

            else:  # point prediction
                # logits is actually the predicted value
                y_new = logits_last[0]  # point predictions have d_vocab=1

            # add to context
            x_context = torch.cat([x_context, x_new], dim=0)
            if y_context.shape[0] == 0:
                y_context = y_new.unsqueeze(0)
            else:
                y_context = torch.cat(
                    [y_context, y_new.to(y_context.dtype).unsqueeze(0)],
                    dim=0,
                )

        if was_training:
            self.train()

        return x_context, y_context

    @torch.no_grad()
    def predict_on_prompt(
        self,
        x: Float[torch.Tensor, "... seq input_dim"],
        y: Float[torch.Tensor, "... seq"],
        *,
        temperature: float = 1.0,
        return_logits: bool = False,
        return_cache: bool = False,
    ) -> (
        DistributionPrediction
        | PointPrediction
        | ClassificationPrediction
        | Tuple[
            DistributionPrediction | PointPrediction | ClassificationPrediction,
            ActivationCache,
        ]
    ):
        """Return predictions aligned with the configured prediction_type.

        For distributional configs, this returns bucket probabilities (optionally
        logits) and the continuous y-grid. The y-grid contains bucket midpoints for
        density inspection; to draw continuous samples use :meth:`generate` or
        :meth:`Bucketizer.sample`. For point configs, this returns direct regression
        estimates. For classification configs, this returns class probabilities.
        Inputs may be provided with or without an explicit batch dimension.

        Parameters
        ----------
        x : Float[torch.Tensor, "... seq input_dim"]
            Input features.
        y : Float[torch.Tensor, "... seq"]
            Target values.
        temperature : float, default 1.0
            Temperature for softmax (distribution/classification predictions).
        return_logits : bool, default False
            Whether to include raw logits in the prediction output.
        return_cache : bool, default False
            Whether to return the transformer activation cache.
            If True, returns tuple of (prediction, cache).

        Returns
        -------
        DistributionPrediction | PointPrediction | ClassificationPrediction | Tuple
            Prediction results, optionally with activation cache.
        """

        was_training = self.training
        self.eval()

        squeeze_batch = x.dim() == 2
        if squeeze_batch:
            x_b = x.unsqueeze(0)
            y_b = y.unsqueeze(0) if y.dim() == 1 else y
        else:
            x_b = x
            y_b = y

        device = next(self.parameters()).device
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        if return_cache:
            logits, cache = self(x_b, y_b, return_cache=True)
        else:
            logits = self(x_b, y_b)

        if isinstance(self.config, ClassificationPFNConfig):
            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            probs = F.softmax(logits / temperature, dim=-1)
            probs_out = probs[0] if squeeze_batch else probs
            logits_out = (
                (logits[0] if squeeze_batch else logits) if return_logits else None
            )
            result = ClassificationPrediction(
                probs=probs_out.detach().cpu(),
                logits=logits_out.detach().cpu() if logits_out is not None else None,
            )
            if was_training:
                self.train()
            return (result, cache) if return_cache else result

        if (
            isinstance(self.config, SupervisedRegressionPFNConfig)
            and self.config.prediction_type == "distribution"
        ):
            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            probs = F.softmax(logits / temperature, dim=-1)
            bucket_indices = torch.arange(self.config.d_vocab, device=device)
            y_grid = self.get_y_values(bucket_indices).detach().cpu()
            probs_out = probs[0] if squeeze_batch else probs
            logits_out = (
                (logits[0] if squeeze_batch else logits) if return_logits else None
            )
            result = DistributionPrediction(
                probs=probs_out.detach().cpu(),
                y_grid=y_grid,
                logits=logits_out.detach().cpu() if logits_out is not None else None,
            )
            if was_training:
                self.train()
            return (result, cache) if return_cache else result

        preds_out = logits[0] if squeeze_batch else logits
        result_point = PointPrediction(
            preds=preds_out.detach().cpu(),
        )
        if was_training:
            self.train()
        return (result_point, cache) if return_cache else result_point

    def _forward_autoregressive_pfn(
        self,
        x: Float[torch.Tensor, "batch seq input_dim"],
        y: Float[torch.Tensor, "batch seq"],
        return_cache: bool = False,
    ) -> (
        Float[torch.Tensor, "batch seq d_vocab"]
        | Tuple[Float[torch.Tensor, "batch seq d_vocab"], ActivationCache]
    ):
        """Interleave x/y tokens, apply the PFN attention mask, and produce next-y logits.

        Parameters
        ----------
        x: Float[torch.Tensor, "batch seq input_dim"]
            Feature tokens for each training example.
        y: Float[torch.Tensor, "batch seq"]
            Targets; interleaved with x so each pair forms a (x, y) token block.
        return_cache: bool
            Whether to also capture the raw `ActivationCache`. Note this cache preserves the
            interleaved sequence order; you'll need to flatten/slice manually when post-processing.
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device

        # Prepare input embeddings based on y type
        if (
            isinstance(self.config, ClassificationPFNConfig)
            and self.config.y_type == "categorical"
        ):
            # project x and embed y separately to avoid leaking labels into x tokens
            x_proj = self.x_proj(x)  # (batch, seq, d_model)
            y_embed = self.y_embed(y.long())  # (batch, seq, d_model)
            hidden = torch.zeros(
                batch_size,
                2 * seq_len,
                self.config.d_model,
                device=device,
                dtype=x_proj.dtype,
            )
            hidden[:, ::2, :] = x_proj
            hidden[:, 1::2, :] = x_proj + y_embed
        else:
            # continuous y: concatenate x and y, then project
            xy_combined = torch.zeros(
                batch_size,
                2 * seq_len,
                input_dim + 1,
                device=device,
                dtype=x.dtype,
            )
            xy_combined[:, :, :input_dim] = x.repeat_interleave(2, dim=1)
            xy_combined[:, 1::2, -1] = y
            hidden = self.input_proj(xy_combined)
        residual, shortformer_pos_embed = self._prepare_transformer_input(hidden)

        # Create mask hook
        mask_hook = create_custom_mask_hook(
            self.config.mask_type, 2 * seq_len, 2 * seq_len
        )

        # Pass through transformer with custom mask
        with self.transformer.hooks(
            fwd_hooks=[(lambda name: name.endswith("attn_scores"), mask_hook)]
        ):
            if return_cache:
                logits, cache = self.transformer.run_with_cache(
                    residual,
                    start_at_layer=0,
                    return_type="logits",
                    shortformer_pos_embed=shortformer_pos_embed,
                )
            else:
                logits = self.transformer(
                    residual,
                    start_at_layer=0,
                    return_type="logits",  # assert Tensor output
                    shortformer_pos_embed=shortformer_pos_embed,
                )

        predictions = self.output_proj(logits)

        if return_cache:
            return predictions, cache
        else:
            return predictions

    def _forward_gpt2(
        self,
        x: Float[torch.Tensor, "batch seq input_dim"],
        y: Float[torch.Tensor, "batch seq"],
        return_cache: bool = False,
    ) -> (
        Float[torch.Tensor, "batch seq d_vocab"]
        | Tuple[Float[torch.Tensor, "batch seq d_vocab"], ActivationCache]
    ):
        """Standard GPT-style causal pass without a custom attention hook."""

        batch_size, seq_len, input_dim = x.shape
        device = x.device

        # Prepare input embeddings based on y type
        if (
            isinstance(self.config, ClassificationPFNConfig)
            and self.config.y_type == "categorical"
        ):
            # project x and embed y separately, then interleave as separate tokens
            x_proj = self.x_proj(x)  # (batch, seq, d_model)
            y_embed = self.y_embed(y.long())  # (batch, seq, d_model)

            hidden = torch.zeros(
                batch_size,
                2 * seq_len,
                self.config.d_model,
                device=device,
                dtype=x_proj.dtype,
            )
            hidden[:, ::2, :] = x_proj  # x at even positions
            hidden[:, 1::2, :] = x_proj + y_embed  # x plus y at odd positions
        else:
            # continuous y: concatenate x and y, then project
            xy_combined = torch.zeros(
                batch_size,
                2 * seq_len,
                input_dim + 1,
                device=device,
                dtype=x.dtype,
            )
            xy_combined[:, :, :input_dim] = x
            xy_combined[:, 1::2, -1] = y
            hidden = self.input_proj(xy_combined)
        residual, shortformer_pos_embed = self._prepare_transformer_input(hidden)

        if return_cache:
            logits, cache = self.transformer.run_with_cache(
                residual,
                start_at_layer=0,
                return_type="logits",
                shortformer_pos_embed=shortformer_pos_embed,
            )
        else:
            logits = self.transformer(
                residual,
                start_at_layer=0,
                return_type="logits",
                shortformer_pos_embed=shortformer_pos_embed,
            )

        predictions = self.output_proj(logits)

        if return_cache:
            return predictions, cache
        return predictions


def PFNModel(config: BasePFNConfig) -> BasePFN:
    """Factory function to create the appropriate PFN model based on config type.

    Maintains backward compatibility while using the new class hierarchy.

    Parameters
    ----------
    config : BasePFNConfig
        Configuration determining which model type to instantiate.

    Returns
    -------
    BasePFN
        UnsupervisedPFN for UnsupervisedPFNConfig, SupervisedPFN otherwise.
    """
    if isinstance(config, UnsupervisedPFNConfig):
        return UnsupervisedPFN(config)
    elif isinstance(config, (SupervisedRegressionPFNConfig, ClassificationPFNConfig)):
        return SupervisedPFN(config)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")
