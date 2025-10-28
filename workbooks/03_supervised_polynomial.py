"""
Cookbook 3: Supervised Polynomial Regression

Demonstrates training a PFN to learn polynomial functions:
    y = a0 + a1*x + a2*x^2 + ε
where coefficients ~ N(0, 1) and ε ~ N(0, 0.1)
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from pfn_transformerlens.sampler.data_generator import DeterministicFunctionGenerator
from pfn_transformerlens.model.configs import SupervisedRegressionPFNConfig
from pfn_transformerlens.train import train, TrainingConfig


def polynomial_function(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Polynomial: y = a0 + a1*x + a2*x^2"""
    x_flat = x.squeeze(-1)  # assume 1D input
    return coeffs[0] + coeffs[1] * x_flat + coeffs[2] * x_flat**2


def main():
    output_dir = Path(__file__).parent / "outputs" / "03_polynomial"
    output_dir.mkdir(parents=True, exist_ok=True)

    # prior over 3 coefficients
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(3), torch.ones(3)),
        reinterpreted_batch_ndims=1
    )

    data_gen = DeterministicFunctionGenerator(
        prior=prior,
        function=polynomial_function,
        input_dim=1,
        noise_std=0.1,
    )

    model_config = SupervisedRegressionPFNConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_head=32,
        n_ctx=128,
        d_vocab=50,
        input_dim=1,
        prediction_type="distribution",
        bucket_type="uniform",
        y_min=-10.0,
        y_max=10.0,
        mask_type="autoregressive-pfn",
        act_fn="gelu",
    )

    training_config = TrainingConfig(
        batch_size=128,
        num_steps=300,
        learning_rate=1e-3,
        warmup_steps=30,
        log_every=100,
        use_wandb=False,
    )

    model = train(data_gen, model_config, training_config)

    # test on multiple polynomials
    device = next(model.parameters()).device
    num_test = 3
    num_context = 15
    x_range = torch.linspace(-2, 2, 40).unsqueeze(-1)  # 15 + 40 = 55 pairs = 110 tokens < 128

    fig, axes = plt.subplots(1, num_test, figsize=(15, 4))

    for idx in range(num_test):
        # sample random polynomial
        coeffs = torch.randn(3)

        # context points
        x_context = torch.rand(num_context, 1) * 4 - 2  # uniform in [-2, 2]
        y_context = polynomial_function(x_context, coeffs) + 0.1 * torch.randn(num_context)

        # predictions on dense grid
        x_all = torch.cat([x_context, x_range], dim=0).to(device)
        y_all = torch.cat([y_context, torch.zeros(len(x_range))], dim=0).to(device)

        with torch.no_grad():
            pred = model.predict_on_prompt(x_all, y_all)

        query_probs = pred.probs[num_context:].cpu()
        y_grid = pred.y_grid.cpu()
        y_pred_mean = (query_probs * y_grid).sum(dim=-1)
        y_pred_std = torch.sqrt((query_probs * (y_grid - y_pred_mean.unsqueeze(-1))**2).sum(dim=-1))

        # true curve
        y_true = polynomial_function(x_range, coeffs)

        ax = axes[idx]
        ax.scatter(x_context.squeeze(), y_context, label='Context', alpha=0.6, s=50)
        ax.plot(x_range.squeeze(), y_true, 'g--', label='True', alpha=0.7)
        ax.plot(x_range.squeeze(), y_pred_mean, 'b-', label='Predicted')
        ax.fill_between(
            x_range.squeeze(),
            y_pred_mean - 2*y_pred_std,
            y_pred_mean + 2*y_pred_std,
            alpha=0.3
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Polynomial {idx+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "polynomial_predictions.png", dpi=150)
    plt.close()

    print(f"Saved: {output_dir / 'polynomial_predictions.png'}")


if __name__ == "__main__":
    main()
