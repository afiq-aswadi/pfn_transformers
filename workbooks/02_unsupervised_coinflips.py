"""
Cookbook 2: Unsupervised Beta-Bernoulli (Coinflips)

Demonstrates training a PFN to learn coinflip sequences with varying bias:
    p ~ Beta(2, 2)
    observations ~ Bernoulli(p)

The PFN learns to perform posterior updating: as it sees more flips,
predictions converge to the true bias.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pfn_transformerlens import (
    train,
    TrainingConfig,
    UnsupervisedConfig,
    UnsupervisedBayesian,
)
from pfn_transformerlens.bayes import Prior, Likelihood


def main():
    output_dir = Path(__file__).parent / "outputs" / "02_coinflips"
    output_dir.mkdir(parents=True, exist_ok=True)

    # prior: Beta(2, 2) distribution over bias
    prior = Prior(
        base_distribution=torch.distributions.Beta(
            torch.tensor(10.0), torch.tensor(2.0)
        )
    )

    # likelihood: Bernoulli parameterized by sampled bias
    def bernoulli_parameterizer(theta: torch.Tensor, x: torch.Tensor) -> dict:
        # broadcast theta (bias) to match sequence length
        seq_len = x.shape[0]
        return {"probs": theta.expand(seq_len)}

    likelihood = Likelihood(
        base_distribution=torch.distributions.Bernoulli(0.5),  # dummy base
        parameterizer=bernoulli_parameterizer,
        input_dim=1,
    )

    # wrap to cast to long for discrete model
    class DiscreteBernoulliGenerator:
        def __init__(self, base_gen):
            self.base_gen = base_gen
            self.input_dim = base_gen.input_dim

        def generate(self, seq_len: int) -> torch.Tensor:
            y = self.base_gen.generate(seq_len)
            return y.long()

    base_gen = UnsupervisedBayesian(prior=prior, likelihood=likelihood)
    data_gen = DiscreteBernoulliGenerator(base_gen)

    # model config: discrete inputs (0 or 1)
    model_config = UnsupervisedConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_head=32,
        n_ctx=32,
        d_vocab=2,  # binary outcomes: 0 or 1
        input_type="discrete",
        prediction_type="distribution",
        act_fn="gelu",
    )

    # training
    training_config = TrainingConfig(
        batch_size=128,
        seq_len=32,
        num_steps=300,
        learning_rate=1e-3,
        warmup_steps=30,
        log_every=100,
        use_wandb=False,
    )

    model = train(data_gen, model_config, training_config)

    # test: show posterior updating
    device = next(model.parameters()).device
    true_bias = 0.7
    num_flips = 16

    # generate coinflips
    flips = torch.bernoulli(torch.full((num_flips,), true_bias)).long().to(device)

    # track predicted probability as we see more flips
    predicted_probs = []
    for i in range(1, num_flips + 1):
        context = flips[:i]
        with torch.no_grad():
            pred = model.predict_on_prompt(context.unsqueeze(0))
        # probability of heads (class 1)
        prob_heads = pred.probs[0, -1, 1].item()
        predicted_probs.append(prob_heads)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # plot 1: predicted probability vs observations
    ax = axes[0]
    ax.plot(range(1, num_flips + 1), predicted_probs, label="Predicted P(heads)")
    ax.axhline(true_bias, color="r", linestyle="--", label="True bias")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Predicted P(heads)")
    ax.set_title("Posterior Updating with More Data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plot 2: cumulative average vs predicted
    cumsum = flips.cpu().numpy().cumsum()
    cumavg = cumsum / (torch.arange(1, num_flips + 1).numpy())
    ax = axes[1]
    ax.plot(range(1, num_flips + 1), cumavg, label="Empirical average")
    ax.plot(range(1, num_flips + 1), predicted_probs, label="PFN prediction")
    ax.axhline(true_bias, color="r", linestyle="--", label="True bias")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Probability")
    ax.set_title("Empirical vs Predicted Probabilities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "posterior_updating.png", dpi=400)
    plt.close()

    print(f"Saved: {output_dir / 'posterior_updating.png'}")


if __name__ == "__main__":
    main()
