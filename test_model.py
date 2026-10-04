import pytest
import torch

from model import HashGridEncoding, model_from_config


def test_hybrid_layout_uses_dense_coarse_levels_and_hashed_fine_levels():
    encoding = HashGridEncoding(
        n_levels=3,
        n_features=2,
        log2_hashmap_size=6,
        base_resolution=2,
        max_resolution=16,
        layout="hybrid",
    )

    assert encoding.level_sizes == (16, 49, 64)
    assert encoding.level_offsets.tolist() == [0, 16, 65]
    assert encoding.dense_strides.tolist() == [4, 7, 0]
    assert encoding.embedding.weight.shape == (129, 2)
    assert encoding(torch.tensor([[0.0, 0.0], [1.0, 1.0]])).shape == (2, 6)


def test_per_level_hash_budgets_control_packed_offsets():
    encoding = HashGridEncoding(
        n_levels=3,
        n_features=2,
        level_log2_hashmap_sizes=[4, 5, 6],
        base_resolution=16,
        max_resolution=64,
    )

    assert encoding.level_sizes == (16, 32, 64)
    assert encoding.level_offsets.tolist() == [0, 16, 48]
    assert encoding.hash_masks.tolist() == [15, 31, 63]


def test_per_level_hash_budget_count_must_match_levels():
    with pytest.raises(ValueError, match="one value per level"):
        HashGridEncoding(n_levels=3, level_log2_hashmap_sizes=[4, 5])


def test_paper_architecture_uses_requested_hidden_layers_and_midpoint_skip():
    network = model_from_config(
        {
            "input_dim": 2,
            "layer_size": 64,
            "num_layers": 5,
            "architecture": "paper-skip-v1",
            "hidden_layer_count": 4,
            "hashgrid_config": {
                "n_levels": 12,
                "n_features": 2,
                "log2_hashmap_size": 4,
            },
        }
    )

    assert len(network.layers) == 5
    assert network.skip_index == 1
    assert network.layers[1].out_features == 40
    assert network.layers[2].in_features == 64
    assert network(torch.rand(3, 2)).shape == (3, 3)


def test_legacy_architecture_keeps_existing_layer_semantics():
    network = model_from_config(
        {
            "input_dim": 2,
            "layer_size": 64,
            "num_layers": 2,
            "hashgrid_config": {
                "n_levels": 12,
                "n_features": 2,
                "log2_hashmap_size": 4,
            },
        }
    )

    assert len(network.layers) == 2
    assert network.skip_index == -1
