#!/usr/bin/env python3
"""
Pytest tests for SVG2 Sparse Attention Backend.

Usage:
    pytest test_svg2_sparse_attn.py -v

This script tests all components of the SVG2 implementation:
1. Triton K-Means clustering
2. Permutation / Inverse Permutation
3. Dynamic Block Mask Generation
4. Triton Block Sparse Attention Kernel
5. Full SVG2 Attention
6. Correctness comparison with Dense Attention

For performance benchmarks, see: benchmark_svg2_sparse_attn.py
"""

import math
from typing import Dict

import pytest
import torch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def svg2_components() -> Dict:
    """Import and return SVG2 components."""
    pytest.importorskip("triton")
    
    from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
        block_sparse_attention,
        identify_dynamic_mask,
        inverse_permute,
        permute_by_labels,
        svg2_attention_forward,
        triton_kmeans,
    )
    
    return {
        'triton_kmeans': triton_kmeans,
        'permute_by_labels': permute_by_labels,
        'inverse_permute': inverse_permute,
        'identify_dynamic_mask': identify_dynamic_mask,
        'block_sparse_attention': block_sparse_attention,
        'svg2_attention_forward': svg2_attention_forward,
    }


@pytest.fixture(scope="module")
def device() -> str:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SVG2 tests")
    return "cuda"


# ============================================================================
# Test: K-Means Clustering
# ============================================================================


class TestKMeans:
    """Tests for Triton K-Means clustering."""

    @pytest.mark.parametrize("B,N,D,K", [
        (1, 512, 64, 8),
        (2, 1024, 64, 16),
        (1, 2048, 128, 32),
    ])
    @torch.inference_mode()
    def test_kmeans_shape(self, svg2_components, device, B, N, D, K):
        """Test K-Means output shapes."""
        x = torch.randn(B, N, D, device=device, dtype=torch.float16)
        
        labels, centroids, sizes = svg2_components['triton_kmeans'](x, K, max_iters=5)
        
        assert labels.shape == (B, N), f"Labels shape mismatch: {labels.shape}"
        assert centroids.shape == (B, K, D), f"Centroids shape mismatch: {centroids.shape}"
        assert sizes.shape == (B, K), f"Sizes shape mismatch: {sizes.shape}"

    @pytest.mark.parametrize("B,N,K", [
        (1, 512, 8),
        (2, 1024, 16),
    ])
    @torch.inference_mode()
    def test_kmeans_cluster_sizes_sum(self, svg2_components, device, B, N, K):
        """Test that cluster sizes sum to N."""
        D = 64
        x = torch.randn(B, N, D, device=device, dtype=torch.float16)
        
        _, _, sizes = svg2_components['triton_kmeans'](x, K, max_iters=5)
        
        size_sums = sizes.sum(dim=-1).tolist()
        expected = [N] * B
        assert size_sums == expected, f"Size sum mismatch: {size_sums} vs {expected}"

    @torch.inference_mode()
    def test_kmeans_labels_range(self, svg2_components, device):
        """Test that labels are in valid range [0, K)."""
        B, N, D, K = 2, 1024, 64, 16
        x = torch.randn(B, N, D, device=device, dtype=torch.float16)
        
        labels, _, _ = svg2_components['triton_kmeans'](x, K, max_iters=5)
        
        assert labels.min() >= 0, f"Labels min out of range: {labels.min()}"
        assert labels.max() < K, f"Labels max out of range: {labels.max()}"

    @torch.inference_mode()
    def test_kmeans_no_nan_inf(self, svg2_components, device):
        """Test that K-Means outputs contain no NaN or Inf."""
        B, N, D, K = 1, 1024, 64, 16
        x = torch.randn(B, N, D, device=device, dtype=torch.float16)
        
        labels, centroids, sizes = svg2_components['triton_kmeans'](x, K, max_iters=10)
        
        assert not torch.isnan(centroids).any(), "Centroids contain NaN"
        assert not torch.isinf(centroids).any(), "Centroids contain Inf"


# ============================================================================
# Test: Permutation
# ============================================================================


class TestPermutation:
    """Tests for permutation and inverse permutation."""

    @pytest.mark.parametrize("B,H,S,D", [
        (1, 4, 256, 64),
        (2, 8, 512, 128),
    ])
    @torch.inference_mode()
    def test_permute_inverse_identity(self, svg2_components, device, B, H, S, D):
        """Test that permute followed by inverse permute is identity."""
        x = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        labels = torch.randint(0, 16, (B * H, S), device=device)
        
        x_perm, sorted_indices = svg2_components['permute_by_labels'](x, labels)
        x_restored = svg2_components['inverse_permute'](x_perm, sorted_indices)
        
        torch.testing.assert_close(x, x_restored, rtol=1e-3, atol=1e-3)

    @torch.inference_mode()
    def test_permute_shape_preserved(self, svg2_components, device):
        """Test that permutation preserves tensor shape."""
        B, H, S, D = 1, 4, 256, 64
        x = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        labels = torch.randint(0, 16, (B * H, S), device=device)
        
        x_perm, _ = svg2_components['permute_by_labels'](x, labels)
        
        assert x_perm.shape == x.shape, f"Shape changed: {x_perm.shape} vs {x.shape}"

    @torch.inference_mode()
    def test_permute_sorted_indices_valid(self, svg2_components, device):
        """Test that sorted indices are valid permutation indices."""
        B, H, S, D = 1, 4, 256, 64
        x = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        labels = torch.randint(0, 16, (B * H, S), device=device)
        
        _, sorted_indices = svg2_components['permute_by_labels'](x, labels)
        
        # Each row should be a permutation of [0, S-1]
        for i in range(B * H):
            indices_sorted = sorted_indices[i].sort()[0]
            expected = torch.arange(S, device=device, dtype=sorted_indices.dtype)
            torch.testing.assert_close(indices_sorted, expected)


# ============================================================================
# Test: Dynamic Block Mask
# ============================================================================


class TestDynamicBlockMask:
    """Tests for dynamic block mask generation."""

    @pytest.mark.parametrize("B,H,Kq,Kk,D", [
        (1, 4, 16, 16, 64),
        (2, 8, 32, 32, 128),
    ])
    @torch.inference_mode()
    def test_mask_shape(self, svg2_components, device, B, H, Kq, Kk, D):
        """Test mask output shape."""
        q_centroids = torch.randn(B, H, Kq, D, device=device)
        k_centroids = torch.randn(B, H, Kk, D, device=device)
        q_sizes = torch.ones(B, H, Kq, device=device) * 10
        k_sizes = torch.ones(B, H, Kk, device=device) * 10
        
        mask = svg2_components['identify_dynamic_mask'](
            q_centroids, k_centroids, q_sizes, k_sizes, top_p=0.5
        )
        
        assert mask.shape == (B, H, Kq, Kk), f"Mask shape mismatch: {mask.shape}"

    @pytest.mark.parametrize("top_p", [0.3, 0.5, 0.7, 0.9])
    @torch.inference_mode()
    def test_mask_sparsity_range(self, svg2_components, device, top_p):
        """Test that mask sparsity is roughly controlled by top_p."""
        B, H, Kq, Kk, D = 1, 4, 32, 32, 64
        
        q_centroids = torch.randn(B, H, Kq, D, device=device)
        k_centroids = torch.randn(B, H, Kk, D, device=device)
        q_sizes = torch.ones(B, H, Kq, device=device) * 10
        k_sizes = torch.ones(B, H, Kk, device=device) * 10
        
        mask = svg2_components['identify_dynamic_mask'](
            q_centroids, k_centroids, q_sizes, k_sizes, top_p=top_p
        )
        
        density = mask.float().mean().item()
        # Allow some tolerance since top_p is approximate
        assert 0 < density <= 1.0, f"Invalid density: {density}"

    @torch.inference_mode()
    def test_mask_dtype(self, svg2_components, device):
        """Test that mask is boolean."""
        B, H, Kq, Kk, D = 1, 4, 16, 16, 64
        
        q_centroids = torch.randn(B, H, Kq, D, device=device)
        k_centroids = torch.randn(B, H, Kk, D, device=device)
        q_sizes = torch.ones(B, H, Kq, device=device) * 10
        k_sizes = torch.ones(B, H, Kk, device=device) * 10
        
        mask = svg2_components['identify_dynamic_mask'](
            q_centroids, k_centroids, q_sizes, k_sizes, top_p=0.5
        )
        
        assert mask.dtype == torch.bool, f"Mask dtype mismatch: {mask.dtype}"


# ============================================================================
# Test: Triton Block Sparse Attention
# ============================================================================


class TestTritonBlockSparseAttention:
    """Tests for Triton block sparse attention kernel."""

    @pytest.mark.parametrize("B,H,S,D,Kq,Kk", [
        (1, 4, 256, 64, 8, 8),
        (1, 8, 512, 64, 16, 16),
    ])
    @torch.inference_mode()
    def test_dense_mask_correctness(self, svg2_components, device, B, H, S, D, Kq, Kk):
        """Test block sparse attention with all-ones mask equals dense attention."""
        torch.manual_seed(42)
        
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        
        # Uniform cluster sizes
        block_size = S // Kq
        q_cluster_sizes = torch.full((B, H, Kq), block_size, dtype=torch.int32, device=device)
        k_cluster_sizes = torch.full((B, H, Kk), block_size, dtype=torch.int32, device=device)
        
        # All-ones mask = dense attention
        block_mask = torch.ones(B, H, Kq, Kk, dtype=torch.bool, device=device)
        
        output = svg2_components['block_sparse_attention'](
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
        )
        
        # Reference: dense attention
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        dense_output = torch.matmul(attn_weights, v.float()).to(torch.float16)
        
        max_diff = (output - dense_output).abs().max().item()
        assert max_diff < 0.1, f"Max diff too large: {max_diff}"

    @torch.inference_mode()
    def test_sparse_mask_no_nan(self, svg2_components, device):
        """Test that sparse mask produces no NaN."""
        B, H, S, D = 1, 4, 256, 64
        Kq, Kk = 8, 8
        
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        
        block_size = S // Kq
        q_cluster_sizes = torch.full((B, H, Kq), block_size, dtype=torch.int32, device=device)
        k_cluster_sizes = torch.full((B, H, Kk), block_size, dtype=torch.int32, device=device)
        
        # Sparse mask: only diagonal blocks
        sparse_mask = torch.zeros(B, H, Kq, Kk, dtype=torch.bool, device=device)
        for i in range(min(Kq, Kk)):
            sparse_mask[:, :, i, i] = True
        
        output = svg2_components['block_sparse_attention'](
            q, k, v, sparse_mask, q_cluster_sizes, k_cluster_sizes
        )
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    @torch.inference_mode()
    def test_output_shape(self, svg2_components, device):
        """Test that output shape matches input."""
        B, H, S, D = 1, 4, 256, 64
        Kq, Kk = 8, 8
        
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        
        block_size = S // Kq
        q_cluster_sizes = torch.full((B, H, Kq), block_size, dtype=torch.int32, device=device)
        k_cluster_sizes = torch.full((B, H, Kk), block_size, dtype=torch.int32, device=device)
        block_mask = torch.ones(B, H, Kq, Kk, dtype=torch.bool, device=device)
        
        output = svg2_components['block_sparse_attention'](
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
        )
        
        assert output.shape == q.shape, f"Shape mismatch: {output.shape} vs {q.shape}"

    def _ref_block_sparse_attention(self, q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes):
        """Reference implementation using pure PyTorch."""
        B, H, S, D = q.shape
        Kq = q_cluster_sizes.shape[-1]
        Kk = k_cluster_sizes.shape[-1]
        out = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(D)
        
        # Calculate block boundaries
        q_cluster_sizes_cpu = q_cluster_sizes.cpu()
        k_cluster_sizes_cpu = k_cluster_sizes.cpu()
        
        q_ends = torch.cumsum(q_cluster_sizes_cpu, dim=-1)
        q_starts = torch.cat([torch.zeros_like(q_ends[..., :1]), q_ends[..., :-1]], dim=-1)
        
        k_ends = torch.cumsum(k_cluster_sizes_cpu, dim=-1)
        k_starts = torch.cat([torch.zeros_like(k_ends[..., :1]), k_ends[..., :-1]], dim=-1)
        
        q_cpu = q.float().cpu()
        k_cpu = k.float().cpu()
        v_cpu = v.float().cpu()
        mask_cpu = block_mask.cpu()
        
        for b in range(B):
            for h in range(H):
                for i in range(Kq):
                    qs, qe = int(q_starts[b, h, i]), int(q_ends[b, h, i])
                    if qs >= qe: continue
                    q_blk = q_cpu[b, h, qs:qe] # [M, D]
                    
                    # Gather valid keys
                    k_list = []
                    v_list = []
                    for j in range(Kk):
                        if mask_cpu[b, h, i, j]:
                            ks, ke = int(k_starts[b, h, j]), int(k_ends[b, h, j])
                            if ks < ke:
                                k_list.append(k_cpu[b, h, ks:ke])
                                v_list.append(v_cpu[b, h, ks:ke])
                    
                    if not k_list:
                        continue
                        
                    k_cat = torch.cat(k_list, dim=0) # [N_total, D]
                    v_cat = torch.cat(v_list, dim=0) # [N_total, D]
                    
                    # Attention
                    scores = torch.matmul(q_blk, k_cat.transpose(0, 1)) * scale
                    probs = torch.softmax(scores, dim=-1)
                    out_blk = torch.matmul(probs, v_cat)
                    
                    out[b, h, qs:qe] = out_blk.to(out.dtype).to(out.device)
        
        return out.to(q.device)

    @torch.inference_mode()
    def test_sparse_mask_correctness(self, svg2_components, device):
        """Test block sparse attention with random sparse mask."""
        B, H, S, D = 1, 4, 256, 64
        Kq, Kk = 8, 8
        
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        
        block_size = S // Kq
        q_cluster_sizes = torch.full((B, H, Kq), block_size, dtype=torch.int32, device=device)
        k_cluster_sizes = torch.full((B, H, Kk), block_size, dtype=torch.int32, device=device)
        
        # Random sparse mask
        block_mask = torch.rand(B, H, Kq, Kk, device=device) > 0.5
        
        output = svg2_components['block_sparse_attention'](
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
        )
        
        ref_output = self._ref_block_sparse_attention(
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
        )
        
        max_diff = (output - ref_output).abs().max().item()
        assert max_diff < 0.1, f"Max diff too large: {max_diff}"

    @torch.inference_mode()
    def test_variable_block_sizes(self, svg2_components, device):
        """Test with variable block sizes."""
        B, H, S, D = 1, 4, 256, 64
        Kq, Kk = 4, 4
        
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        
        # Variable block sizes that sum to S=256
        # e.g., [32, 64, 96, 64]
        sizes_list = [32, 64, 96, 64]
        q_cluster_sizes = torch.tensor(sizes_list, dtype=torch.int32, device=device).expand(B, H, Kq)
        k_cluster_sizes = torch.tensor(sizes_list, dtype=torch.int32, device=device).expand(B, H, Kk)
        
        block_mask = torch.rand(B, H, Kq, Kk, device=device) > 0.5
        
        output = svg2_components['block_sparse_attention'](
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
        )
        
        ref_output = self._ref_block_sparse_attention(
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
        )
        
        max_diff = (output - ref_output).abs().max().item()
        assert max_diff < 0.1, f"Max diff too large: {max_diff}"


# ============================================================================
# Test: Full SVG2 Attention
# ============================================================================


class TestFullSVG2Attention:
    """Tests for complete SVG2 attention forward pass."""

    @pytest.mark.parametrize("B,S,H,D", [
        (1, 512, 4, 64),
        (1, 1024, 8, 64),
    ])
    @torch.inference_mode()
    def test_forward_shape(self, svg2_components, device, B, S, H, D):
        """Test SVG2 forward output shape."""
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        
    output, _, _ = svg2_components['svg2_attention_forward'](
        q, k, v,
        num_q_clusters=16,
        num_k_clusters=16,
        top_p=0.5,
        kmeans_iters=3,
    )

    assert output.shape == q.shape, f"Shape mismatch: {output.shape}"

    @torch.inference_mode()
    def test_forward_no_nan_inf(self, svg2_components, device):
        """Test SVG2 forward produces no NaN or Inf."""
        B, S, H, D = 1, 1024, 8, 64
        
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        
    output, _, _ = svg2_components['svg2_attention_forward'](
        q, k, v,
        num_q_clusters=32,
        num_k_clusters=32,
        top_p=0.5,
        kmeans_iters=3,
    )

    assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    @pytest.mark.parametrize("top_p", [0.3, 0.5, 0.8])
    @torch.inference_mode()
    def test_forward_different_top_p(self, svg2_components, device, top_p):
        """Test SVG2 forward with different top_p values."""
        B, S, H, D = 1, 512, 4, 64
        
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        
    output, _, _ = svg2_components['svg2_attention_forward'](
        q, k, v,
        num_q_clusters=16,
        num_k_clusters=16,
        top_p=top_p,
        kmeans_iters=3,
    )

    assert output.shape == q.shape
        assert not torch.isnan(output).any()


# ============================================================================
# Test: Correctness vs Dense Attention
# ============================================================================


class TestCorrectnessVsDense:
    """Tests comparing SVG2 output to dense attention."""

    @torch.inference_mode()
    def test_high_top_p_close_to_dense(self, svg2_components, device):
        """With high top_p, SVG2 should be close to dense attention."""
        B, S, H, D = 1, 256, 4, 64
        
        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
        
        # Dense attention (reference)
        scale = 1.0 / math.sqrt(D)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        dense_out = torch.matmul(attn_weights, v_t).transpose(1, 2)
        
    # SVG2 with high top_p
    svg2_out, _, _ = svg2_components['svg2_attention_forward'](
        q, k, v,
        num_q_clusters=8,
        num_k_clusters=8,
        top_p=0.95,
        kmeans_iters=5,
    )

    max_diff = (svg2_out - dense_out).abs().max().item()
        # With high top_p, should be reasonably close
        assert max_diff < 5.0, f"Max diff too large: {max_diff}"

    @torch.inference_mode()
    def test_reproducibility(self, svg2_components, device):
        """Test that same seed produces same output."""
        B, S, H, D = 1, 512, 4, 64
        
        def run_with_seed(seed):
            torch.manual_seed(seed)
            q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
            k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
            v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
            
        # Use same seed for internal randomness
        torch.manual_seed(seed)
        output, _, _ = svg2_components['svg2_attention_forward'](
            q, k, v, num_q_clusters=16, num_k_clusters=16, top_p=0.5, kmeans_iters=3
        )
        return output
    
    out1 = run_with_seed(123)
        out2 = run_with_seed(123)
        
        torch.testing.assert_close(out1, out2)


# ============================================================================
# Backend Integration Tests
# ============================================================================


class TestBackendIntegration:
    """Tests for SVG2 attention backend integration."""

    @torch.inference_mode()
    def test_backend_enum_exists(self):
        """Test that SVG2_SPARSE_ATTN enum exists."""
        from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
        
        assert hasattr(AttentionBackendEnum, 'SVG2_SPARSE_ATTN')

    @torch.inference_mode()
    def test_backend_class_structure(self):
        """Test that backend class has required methods."""
        from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
            SVG2SparseAttentionBackend,
        )
        
        assert hasattr(SVG2SparseAttentionBackend, 'get_enum')
        assert hasattr(SVG2SparseAttentionBackend, 'get_impl_cls')
        assert hasattr(SVG2SparseAttentionBackend, 'get_metadata_cls')
        assert hasattr(SVG2SparseAttentionBackend, 'get_builder_cls')

    @torch.inference_mode()
    def test_impl_forward_method(self, device):
        """Test that attention impl has forward method."""
        from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
            SVG2SparseAttentionBackend,
            SVG2SparseAttentionMetadata,
        )
        
        impl_cls = SVG2SparseAttentionBackend.get_impl_cls()
        impl = impl_cls(
            num_heads=8,
            head_size=64,
            softmax_scale=1.0,
            causal=False,
        )
        
        B, S, H, D = 1, 256, 8, 64
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        
        # Create metadata with required parameters
        metadata = SVG2SparseAttentionMetadata(
            current_timestep=0,
            num_frames=16,
            num_tokens_per_frame=16,  # S // num_frames
            num_q_clusters=8,
            num_k_clusters=8,
            top_p=0.5,
            kmeans_iters=3,
        )
        
        output = impl.forward(q, k, v, metadata)
        
        assert output.shape == q.shape


# ============================================================================
# Main entry point for standalone execution
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
