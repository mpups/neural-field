import pytest
import torch

from model import HashGridEncoding


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
