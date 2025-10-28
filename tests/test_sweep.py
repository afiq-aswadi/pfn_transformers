"""Tests for wandb sweep functionality."""

from collections.abc import Callable
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from pfn_transformerlens.sweep import (
    SweepConfig,
    DataConfig,
    create_data_generator,
    get_available_gpus,
    parse_gpu_spec,
    run_sweep_agents,
    train_one_run,
)


class TestGPUDetection:
    """Test GPU detection and parsing."""

    def test_get_available_gpus_no_cuda(self):
        """Test GPU detection when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            gpus = get_available_gpus()
            assert gpus == []

    def test_get_available_gpus_with_cuda(self):
        """Test GPU detection with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=4):
                gpus = get_available_gpus()
                assert gpus == [0, 1, 2, 3]

    def test_parse_gpu_spec_no_gpus_available(self):
        """Test parsing fails when no GPUs available."""
        with patch("pfn_transformerlens.sweep.get_available_gpus", return_value=[]):
            with pytest.raises(ValueError, match="No CUDA devices found"):
                parse_gpu_spec(None)

    def test_parse_gpu_spec_all(self):
        """Test parsing 'all' specification."""
        with patch(
            "pfn_transformerlens.sweep.get_available_gpus", return_value=[0, 1, 2]
        ):
            assert parse_gpu_spec("all") == [0, 1, 2]
            assert parse_gpu_spec(None) == [0, 1, 2]

    def test_parse_gpu_spec_string(self):
        """Test parsing comma-separated GPU IDs."""
        with patch(
            "pfn_transformerlens.sweep.get_available_gpus", return_value=[0, 1, 2, 3]
        ):
            assert parse_gpu_spec("0,2") == [0, 2]
            assert parse_gpu_spec("1, 3") == [1, 3]

    def test_parse_gpu_spec_list(self):
        """Test parsing list of GPU IDs."""
        with patch(
            "pfn_transformerlens.sweep.get_available_gpus", return_value=[0, 1, 2]
        ):
            assert parse_gpu_spec([0, 1]) == [0, 1]

    def test_parse_gpu_spec_invalid_ids(self):
        """Test parsing fails with invalid GPU IDs."""
        with patch("pfn_transformerlens.sweep.get_available_gpus", return_value=[0, 1]):
            with pytest.raises(ValueError, match="Invalid GPU IDs"):
                parse_gpu_spec("0,5")


class TestSweepConfig:
    """Test sweep configuration."""

    def test_default_config(self):
        """Test default sweep configuration."""
        config = SweepConfig()
        assert config.method == "grid"
        assert config.metric_name == "final_test_mse"
        assert config.metric_goal == "minimize"
        assert config.num_tasks == (2, 5, 10, 20)

    def test_to_wandb_config_basic(self):
        """Test conversion to wandb config format."""
        config = SweepConfig(num_tasks=(2, 5))
        wandb_config = config.to_wandb_config()

        assert wandb_config["method"] == "grid"
        assert wandb_config["metric"] == {
            "name": "final_test_mse",
            "goal": "minimize",
        }
        assert wandb_config["parameters"]["num_tasks"]["values"] == [2, 5]

    def test_to_wandb_config_with_optional_params(self):
        """Test conversion with optional parameters."""
        config = SweepConfig(
            num_tasks=(2, 5),
            learning_rate=(1e-4, 1e-3),
            d_model=(32, 64),
        )
        wandb_config = config.to_wandb_config()

        params = wandb_config["parameters"]
        assert params["num_tasks"]["values"] == [2, 5]
        assert params["learning_rate"]["values"] == [1e-4, 1e-3]
        assert params["d_model"]["values"] == [32, 64]


class TestDataGenerator:
    """Test data generator creation."""

    def test_create_data_generator(self):
        """Test data generator creation."""
        num_tasks = 5
        generator, config = create_data_generator(num_tasks)

        assert isinstance(config, DataConfig)
        assert config.num_tasks == num_tasks
        assert config.task_type == "linear_regression"
        assert config.input_dim == 1

        # test generator produces correct shapes
        x = torch.randn(10, 1)
        task_param = torch.randn(1)
        y = generator.function(x, task_param)
        assert y.shape == (10,)

    def test_data_generator_sampling(self):
        """Test sampling from data generator."""
        from pfn_transformerlens.sampler.dataloader import sample_batch

        generator, _ = create_data_generator(num_tasks=3)

        batch_size = 8
        seq_len = 16
        x, y = sample_batch(generator, batch_size=batch_size, seq_len=seq_len)

        assert x is not None
        assert y is not None
        assert x.shape == (batch_size, seq_len, 1)
        assert y.shape == (batch_size, seq_len)


class TestSweepAgents:
    """Test sweep agent launching."""

    @patch.dict(
        "os.environ", {"WANDB_PROJECT": "test-project", "WANDB_ENTITY": "test-entity"}
    )
    @patch("pfn_transformerlens.sweep.parse_gpu_spec")
    @patch("subprocess.Popen")
    def test_run_sweep_agents_basic(self, mock_popen: Mock, mock_parse_gpu: Mock):
        """Test launching sweep agents on multiple GPUs."""
        mock_parse_gpu.return_value = [0, 1]
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        # launch agents but interrupt immediately
        with pytest.raises(KeyboardInterrupt):
            run_sweep_agents("test-sweep-id", gpus="0,1")
            raise KeyboardInterrupt()

        # verify subprocess.Popen was called correctly
        assert mock_popen.call_count == 2
        calls = mock_popen.call_args_list

        # check first GPU
        assert calls[0][0][0] == [
            "wandb",
            "agent",
            "test-entity/test-project/test-sweep-id",
        ]
        assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"

        # check second GPU
        assert calls[1][0][0] == [
            "wandb",
            "agent",
            "test-entity/test-project/test-sweep-id",
        ]
        assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "1"

    @patch("pfn_transformerlens.sweep.parse_gpu_spec")
    def test_run_sweep_agents_missing_env(self, mock_parse_gpu: Mock):
        """Test agents fail without wandb env vars."""
        mock_parse_gpu.return_value = [0]

        with pytest.raises(ValueError, match="Must provide project/entity"):
            run_sweep_agents("test-sweep-id")


class TestTrainOneRun:
    """Test training function called by wandb agent."""

    @patch("wandb.init")
    @patch("wandb.config")
    @patch("wandb.log")
    @patch("pfn_transformerlens.sweep.train")
    def test_train_one_run_basic(
        self,
        mock_train: Mock,
        mock_log: Mock,
        mock_config: Mock,
        mock_init: Mock,
    ):
        """Test single training run with wandb integration."""
        # setup wandb.config mock
        mock_config.num_tasks = 5
        mock_config.get = MagicMock(
            side_effect=lambda k, d: {"learning_rate": 1e-3, "d_model": 64}.get(k, d)
        )
        mock_config.update = MagicMock()

        # setup trained model mock
        mock_model = MagicMock()
        device = torch.device("cpu")
        mock_model.parameters.return_value = iter([torch.zeros(1, device=device)])
        mock_model.eval = MagicMock()

        # mock forward pass to return correct shape
        def mock_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len = y.shape
            return torch.randn(batch_size, seq_len, 1)

        mock_model.side_effect = mock_forward
        mock_train.return_value = mock_model

        # run training
        train_one_run()

        # verify wandb calls
        mock_init.assert_called_once()
        mock_train.assert_called_once()
        mock_log.assert_called_once()
        # Note: wandb.finish() now called by WandbLogger inside train()

        # verify logged metric
        log_call = mock_log.call_args[0][0]
        assert "final_test_mse" in log_call
        assert isinstance(log_call["final_test_mse"], float)


class TestParallelSweeps:
    """Test parallel sweep agent functionality."""

    @patch.dict(
        "os.environ", {"WANDB_PROJECT": "test-project", "WANDB_ENTITY": "test-entity"}
    )
    @patch("pfn_transformerlens.sweep.parse_gpu_spec")
    @patch("subprocess.Popen")
    def test_multiple_agents_different_gpus(
        self, mock_popen: Mock, mock_parse_gpu: Mock
    ):
        """Test multiple agents launch on different GPUs."""
        mock_parse_gpu.return_value = [0, 1, 2, 3]

        # create mock processes
        mock_procs = []
        for i in range(4):
            proc = MagicMock()
            proc.pid = 1000 + i
            mock_procs.append(proc)

        mock_popen.side_effect = mock_procs

        # launch agents but interrupt immediately
        with pytest.raises(KeyboardInterrupt):
            run_sweep_agents("test-sweep", gpus="0,1,2,3")
            raise KeyboardInterrupt()

        # verify 4 agents launched
        assert mock_popen.call_count == 4

        # verify each agent gets unique GPU
        for i, call in enumerate(mock_popen.call_args_list):
            env = call[1]["env"]
            assert env["CUDA_VISIBLE_DEVICES"] == str(i)

    @patch.dict(
        "os.environ", {"WANDB_PROJECT": "test-project", "WANDB_ENTITY": "test-entity"}
    )
    @patch("pfn_transformerlens.sweep.parse_gpu_spec")
    @patch("subprocess.Popen")
    def test_agent_process_cleanup_on_interrupt(
        self, mock_popen: Mock, mock_parse_gpu: Mock
    ):
        """Test agents are properly terminated on KeyboardInterrupt."""
        mock_parse_gpu.return_value = [0, 1]

        mock_procs = []
        for i in range(2):
            proc = MagicMock()
            proc.pid = 2000 + i
            # first wait() raises interrupt, subsequent calls return None
            proc.wait.side_effect = [KeyboardInterrupt(), None]
            mock_procs.append(proc)

        mock_popen.side_effect = mock_procs

        # run agents - interrupt happens during wait loop
        with pytest.raises(KeyboardInterrupt):
            run_sweep_agents("test-sweep", gpus="0,1")

        # verify all processes get terminated and cleaned up
        for proc in mock_procs:
            proc.terminate.assert_called_once()
            # all processes get wait called at least once during cleanup
            assert proc.wait.call_count >= 1

    @patch("wandb.agent")
    def test_agent_pulls_multiple_configs(self, mock_agent: Mock):
        """Test agent can run multiple sweep configurations."""
        from pfn_transformerlens.sweep import run, RunArgs

        # simulate agent running 3 configs
        configs_run = []

        def mock_agent_fn(sweep_id: str, function: Callable, count: int | None):
            for i in range(count or 1):
                configs_run.append(f"config_{i}")
            return None

        mock_agent.side_effect = mock_agent_fn

        args = RunArgs(sweep_id="test-sweep-multi", count=3)
        run(args)

        # verify agent was called with correct parameters
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args
        assert call_args[1]["count"] == 3

    @patch("pfn_transformerlens.sweep.run_sweep_agents")
    def test_run_parallel_command(self, mock_run_agents: Mock):
        """Test run-parallel command integration."""
        from pfn_transformerlens.sweep import run_parallel, RunParallelArgs

        args = RunParallelArgs(
            sweep_id="test-sweep-parallel",
            gpus="0,1,2",
            project="test-project",
            entity="test-entity",
        )

        run_parallel(args)

        # verify run_sweep_agents called with correct parameters
        mock_run_agents.assert_called_once()
        call_args = mock_run_agents.call_args
        assert call_args[0][0] == "test-sweep-parallel"
        assert call_args[1]["gpus"] == "0,1,2"
        assert call_args[1]["project"] == "test-project"
        assert call_args[1]["entity"] == "test-entity"

    @patch("wandb.init")
    @patch("wandb.config")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("pfn_transformerlens.sweep.train")
    def test_concurrent_training_runs_different_configs(
        self,
        mock_train: Mock,
        mock_finish: Mock,
        mock_log: Mock,
        mock_config: Mock,
        mock_init: Mock,
    ):
        """Test simulating concurrent agents with different configs."""
        # simulate two different configs being run
        configs = [
            {"num_tasks": 2, "learning_rate": 1e-4, "d_model": 32},
            {"num_tasks": 10, "learning_rate": 1e-3, "d_model": 128},
        ]

        for config in configs:
            # setup mock for this config
            mock_config.num_tasks = config["num_tasks"]
            mock_config.get = MagicMock(side_effect=lambda k, d: config.get(k, d))
            mock_config.update = MagicMock()

            # setup trained model mock
            mock_model = MagicMock()
            device = torch.device("cpu")
            mock_model.parameters.return_value = iter([torch.zeros(1, device=device)])
            mock_model.eval = MagicMock()

            def mock_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size, seq_len = y.shape
                return torch.randn(batch_size, seq_len, 1)

            mock_model.side_effect = mock_forward
            mock_train.return_value = mock_model

            # run training
            train_one_run()

        # verify train called for each config
        assert mock_train.call_count == 2

        # verify different configs were used
        train_calls = mock_train.call_args_list
        assert train_calls[0][1]["model_config"].d_model in [32, 128]
        assert train_calls[1][1]["model_config"].d_model in [32, 128]


@pytest.mark.integration
class TestEndToEnd:
    """Integration tests for sweep workflow."""

    @patch("wandb.sweep")
    def test_create_sweep(self, mock_sweep: Mock):
        """Test creating a sweep."""
        from pfn_transformerlens.sweep import create, CreateArgs

        mock_sweep.return_value = "test-sweep-123"

        args = CreateArgs(project="test-project")
        create(args)

        mock_sweep.assert_called_once()
        call_args = mock_sweep.call_args
        assert call_args[1]["project"] == "test-project"

        # verify sweep config structure
        sweep_config = call_args[0][0]
        assert "method" in sweep_config
        assert "metric" in sweep_config
        assert "parameters" in sweep_config

    @patch("wandb.agent")
    def test_run_sweep(self, mock_agent: Mock):
        """Test running a sweep agent."""
        from pfn_transformerlens.sweep import run, RunArgs

        args = RunArgs(sweep_id="test-sweep-123", count=5)
        run(args)

        mock_agent.assert_called_once()
        call_args = mock_agent.call_args
        assert call_args[0][0] == "test-sweep-123"
        assert call_args[1]["count"] == 5
        assert callable(call_args[1]["function"])

    @patch.dict(
        "os.environ", {"WANDB_PROJECT": "parallel-test", "WANDB_ENTITY": "test-org"}
    )
    @patch("wandb.sweep")
    @patch("pfn_transformerlens.sweep.run_sweep_agents")
    def test_full_parallel_sweep_workflow(
        self, mock_run_agents: Mock, mock_sweep: Mock
    ):
        """Test complete workflow: create sweep then run parallel agents."""
        from pfn_transformerlens.sweep import (
            create,
            run_parallel,
            CreateArgs,
            RunParallelArgs,
        )

        # step 1: create sweep
        mock_sweep.return_value = "sweep-xyz"
        create_args = CreateArgs(project="parallel-test")
        create(create_args)

        sweep_id = mock_sweep.return_value

        # step 2: run parallel agents
        parallel_args = RunParallelArgs(
            sweep_id=sweep_id,
            gpus="0,1",
            project="parallel-test",
            entity="test-org",
        )
        run_parallel(parallel_args)

        # verify workflow
        mock_sweep.assert_called_once()
        mock_run_agents.assert_called_once()
        assert mock_run_agents.call_args[0][0] == sweep_id
