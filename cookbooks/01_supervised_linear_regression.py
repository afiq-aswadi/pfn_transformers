"""
Cookbook 1: Supervised Linear Regression

Demonstrates training a PFN to learn linear functions of the form:
    y = w·x + ε
where w ~ N(0, 1) and ε ~ N(0, 0.1)

The PFN learns to perform in-context learning: given examples from one linear
function, it predicts on new points with uncertainty quantification.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from pfn_transformerlens import (
    train,
    TrainingConfig,
    RegressionConfig,
    DeterministicGenerator,
)


def linear_function(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Linear function: y = w·x (element-wise product, then sum over features)"""
    return (w * x).sum(dim=-1)


def main():
    print("=" * 60)
    print("Cookbook 1: Supervised Linear Regression")
    print("=" * 60)

    # setup
    input_dim = 5
    output_dir = Path(__file__).parent / "outputs" / "01_linear_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    # data generator: sample weight vectors from standard normal
    # use Independent to create a multivariate distribution
    print("\nSetting up data generator...")
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim)),
        reinterpreted_batch_ndims=1,
    )
    data_gen = DeterministicGenerator(
        prior=prior,
        function=linear_function,
        input_dim=input_dim,
        noise_std=0.1,
    )

    # model config: small model for quick training
    print("Configuring model...")
    model_config = RegressionConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_head=32,
        n_ctx=128,
        d_vocab=50,  # number of buckets for distribution prediction
        input_dim=input_dim,
        prediction_type="distribution",
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
        mask_type="autoregressive-pfn",
        act_fn="gelu",
    )

    # training config: short training for demo
    print("Configuring training...")
    training_config = TrainingConfig(
        batch_size=128,
        num_steps=500,  # reduced for faster demo
        learning_rate=1e-3,
        warmup_steps=50,
        log_every=100,
        use_wandb=False,
    )

    # train the model
    print(f"\nTraining for {training_config.num_steps} steps...")
    model = train(data_gen, model_config, training_config)
    print("Training complete!")

    # test in-context learning on multiple functions
    print("\nGenerating plots...")
    device = next(model.parameters()).device
    num_test_functions = 3
    num_context = 20
    num_query = 30

    fig, axes = plt.subplots(1, num_test_functions, figsize=(15, 4))

    for idx in range(num_test_functions):
        # sample a random linear function (weight vector)
        w_true = torch.randn(input_dim)

        # generate context examples
        x_context = torch.randn(num_context, input_dim)
        y_context = linear_function(x_context, w_true) + 0.1 * torch.randn(num_context)

        # generate query points
        x_query = torch.randn(num_query, input_dim)
        y_query_true = linear_function(x_query, w_true)

        # predict with PFN (move tensors to device)
        x_all = torch.cat([x_context, x_query], dim=0).to(device)
        y_all = torch.cat([y_context, torch.zeros(num_query)], dim=0).to(device)

        with torch.no_grad():
            pred = model.predict_on_prompt(x_all, y_all)

        # extract predictions for query points
        query_probs = pred.probs[num_context:].cpu()
        y_grid = pred.y_grid.cpu()

        # compute mean and std from distribution
        y_pred_mean = (query_probs * y_grid).sum(dim=-1)
        y_pred_std = torch.sqrt(
            (query_probs * (y_grid - y_pred_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
        )

        # plot: compare predictions vs true values
        ax = axes[idx]
        x_plot = torch.arange(num_query)
        ax.plot(x_plot, y_query_true.cpu(), "o", label="True", alpha=0.7)
        ax.plot(x_plot, y_pred_mean, "s", label="Predicted", alpha=0.7)
        ax.fill_between(
            x_plot,
            y_pred_mean - 2 * y_pred_std,
            y_pred_mean + 2 * y_pred_std,
            alpha=0.3,
            label="±2σ",
        )
        ax.set_xlabel("Query Point")
        ax.set_ylabel("y")
        ax.set_title(f"Function {idx + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "in_context_predictions.png", dpi=150)
    print(f"Saved: {output_dir / 'in_context_predictions.png'}")
    plt.close()

    # detailed plot for one function
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # regenerate for detailed analysis
    w_true = torch.randn(input_dim)
    x_context = torch.randn(num_context, input_dim)
    y_context = linear_function(x_context, w_true) + 0.1 * torch.randn(num_context)
    x_query = torch.randn(num_query, input_dim)
    y_query_true = linear_function(x_query, w_true)

    x_all = torch.cat([x_context, x_query], dim=0).to(device)
    y_all = torch.cat([y_context, torch.zeros(num_query)], dim=0).to(device)

    with torch.no_grad():
        pred = model.predict_on_prompt(x_all, y_all)

    # plot 1: predictions vs true
    query_probs = pred.probs[num_context:].cpu()
    y_grid = pred.y_grid.cpu()
    y_pred_mean = (query_probs * y_grid).sum(dim=-1)
    y_pred_std = torch.sqrt(
        (query_probs * (y_grid - y_pred_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
    )

    ax = axes[0]
    ax.scatter(y_query_true.cpu(), y_pred_mean, alpha=0.6)
    ax.errorbar(
        y_query_true.cpu(), y_pred_mean, yerr=2 * y_pred_std, fmt="none", alpha=0.3
    )
    lim_min = min(y_query_true.min().item(), y_pred_mean.min().item())
    lim_max = max(y_query_true.max().item(), y_pred_mean.max().item())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y (mean)")
    ax.set_title("Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plot 2: predicted distribution for a few query points
    ax = axes[1]
    for i in range(min(5, num_query)):
        ax.plot(y_grid, query_probs[i], alpha=0.7, label=f"Query {i + 1}")
        ax.axvline(y_query_true[i].item(), color=f"C{i}", linestyle="--", alpha=0.5)
    ax.set_xlabel("y")
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Distributions (dashed = true value)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "detailed_analysis.png", dpi=150)
    print(f"Saved: {output_dir / 'detailed_analysis.png'}")
    plt.close()

    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
