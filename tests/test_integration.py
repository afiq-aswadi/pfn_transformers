"""Integration tests for end-to-end PFN workflows.

TODO: Add comprehensive integration tests covering full pipelines:
      - Data generation → training → inference → evaluation
      - Checkpointing → save → load → resume training
      - Multiple model types (supervised regression, classification, unsupervised)
      - Device placement across entire pipeline (CPU/MPS/CUDA)

These tests should verify that all components work together correctly,
not just individually. Current test suite has good unit tests but lacks
integration coverage.
"""

import pytest


@pytest.mark.skip(reason="Integration tests not yet implemented")
def test_supervised_regression_pipeline():
    """Test full supervised regression pipeline: generate → train → infer → evaluate."""
    # TODO: Implement
    pass


@pytest.mark.skip(reason="Integration tests not yet implemented")
def test_checkpoint_save_load_resume():
    """Test checkpoint lifecycle: train → save → load → resume → verify."""
    # TODO: Implement
    pass


@pytest.mark.skip(reason="Integration tests not yet implemented")
def test_unsupervised_pipeline():
    """Test full unsupervised pipeline: generate → train → infer → evaluate."""
    # TODO: Implement
    pass


@pytest.mark.skip(reason="Integration tests not yet implemented")
def test_classification_pipeline():
    """Test full classification pipeline: generate → train → infer → evaluate."""
    # TODO: Implement
    pass


@pytest.mark.skip(reason="Integration tests not yet implemented")
def test_device_placement_across_pipeline():
    """Test that device placement works correctly across entire training pipeline."""
    # TODO: Implement for CPU, MPS, and CUDA (if available)
    pass
