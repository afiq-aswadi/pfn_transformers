"""Tests for model checkpointing and loading.

Tests cover:
- Checkpoint saving during training
- Model loading from checkpoint
- State preservation (weights, optimizer state)
- Prediction consistency after loading
"""

import tempfile
from pathlib import Path

import pytest
import torch

from pfn_transformerlens.model.configs import (
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.train import TrainingConfig, train
from pfn_transformerlens.checkpointing import load_checkpoint


class SimpleRegressionGenerator:
    """Simple generator for regression testing."""

    def __init__(self) -> None:
        self.input_dim = 2

    def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(seq_len, self.input_dim)
        y = x[:, 0] + x[:, 1]  # Simple linear combination
        return x, y


class SimpleUnsupervisedGenerator:
    """Simple generator for unsupervised testing."""

    def __init__(self) -> None:
        self.input_dim = 1

    def generate(self, seq_len: int) -> torch.Tensor:
        return torch.randint(0, 2, (seq_len,))


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""

    def test_checkpoint_is_saved(self) -> None:
        """Test that checkpoints are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=10,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Check that checkpoints exist
            ckpt_dir = Path(tmpdir)
            checkpoints = list(ckpt_dir.glob("*.pt"))
            assert len(checkpoints) > 0

            # Should have checkpoints at steps 5 and 10
            expected_files = ["checkpoint_step_5.pt", "checkpoint_step_10.pt"]
            saved_files = [ckpt.name for ckpt in checkpoints]
            for expected in expected_files:
                assert expected in saved_files

    def test_checkpoint_contains_required_keys(self) -> None:
        """Test that saved checkpoints contain all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load checkpoint and check keys
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            checkpoint = torch.load(ckpt_path, weights_only=False)

            required_keys = [
                "step",
                "model_state_dict",
                "optimizer_state_dict",
                "model_config",
                "training_config",
            ]
            for key in required_keys:
                assert key in checkpoint

    def test_no_checkpoint_when_disabled(self) -> None:
        """Test that checkpoints are not saved when save_checkpoint=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=10,
                save_checkpoint=False,  # Disabled
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # No checkpoints should exist
            ckpt_dir = Path(tmpdir)
            checkpoints = list(ckpt_dir.glob("*.pt"))
            assert len(checkpoints) == 0


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    def test_load_model_basic(self) -> None:
        """Test basic model loading from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load the model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            from pfn_transformerlens.model.PFN import BasePFN

            assert isinstance(loaded_model, BasePFN)
            assert not loaded_model.training  # Should be in eval mode

    def test_load_model_missing_file(self) -> None:
        """Test that loading nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent_checkpoint.pt")

    def test_load_model_weights_match(self) -> None:
        """Test that loaded model has same weights as trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            trained_model = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load the model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            # Compare weights
            for (name1, param1), (name2, param2) in zip(
                trained_model.named_parameters(), loaded_model.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)

    def test_load_model_predictions_match(self) -> None:
        """Test that loaded model produces same predictions as trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            trained_model = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Generate test data
            device = next(trained_model.parameters()).device
            x_test = torch.randn(1, 8, 2, device=device)
            y_test = torch.randn(1, 8, device=device)

            # Get predictions from trained model
            trained_model.eval()
            with torch.no_grad():
                preds_trained = trained_model(x_test, y_test)

            # Load and get predictions from loaded model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            with torch.no_grad():
                preds_loaded = loaded_model(x_test, y_test)

            # Predictions should match
            assert torch.allclose(preds_trained, preds_loaded)

    def test_load_model_device_auto(self) -> None:
        """Test automatic device selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load with auto device
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path), device="auto")

            # Should be on some device
            device = next(loaded_model.parameters()).device
            assert device.type in ["cuda", "mps", "cpu"]

    def test_load_model_device_cpu(self) -> None:
        """Test loading to CPU device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load to CPU
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path), device="cpu")

            device = next(loaded_model.parameters()).device
            assert device.type == "cpu"


class TestCheckpointingUnsupervisedModels:
    """Test checkpointing with unsupervised models."""

    def test_unsupervised_model_checkpoint_and_load(self) -> None:
        """Test checkpoint save/load with unsupervised models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = UnsupervisedPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=2,
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleUnsupervisedGenerator()
            trained_model = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load the model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            # Test predictions match
            device = next(trained_model.parameters()).device
            y_test = torch.randint(0, 2, (1, 8), device=device)

            trained_model.eval()
            with torch.no_grad():
                preds_trained = trained_model(y_test)

            with torch.no_grad():
                preds_loaded = loaded_model(y_test)

            assert torch.allclose(preds_trained, preds_loaded)


class TestResumeTraining:
    """Test resuming training from checkpoint."""

    def test_resume_from_checkpoint(self) -> None:
        """Test that training can be resumed from a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            # First training run
            training_config_1 = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config_1,
            )

            # Resume training
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            training_config_2 = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=10,  # Continue to step 10
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            model_2 = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config_2,
                resume_from=str(ckpt_path),
            )

            # Model should have continued training
            assert model_2 is not None

            # Checkpoint for step 10 should exist
            ckpt_path_10 = Path(tmpdir) / "checkpoint_step_10.pt"
            assert ckpt_path_10.exists()
