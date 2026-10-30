"""Exact optimized NN tours: candidate exhaustion, tied boundaries and devices."""
import importlib.util
import unittest

import numpy as np
import torch

from globalsplat.compression import CodecConfig
from globalsplat.compression.token_order import (
    _compiled_greedy_graph, _compiled_greedy_knn, _graph_candidates, _greedy_row_order,
    _nearest_candidates, greedy_neighbor_order, hilbert_order_3d,
)
from globalsplat.compression.morton import morton_codes_3d

HAS_NUMBA = importlib.util.find_spec("numba") is not None


class NNOrderBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_row_matches_original_for_both_metrics_batches_and_short_inputs(self):
        generator = torch.Generator().manual_seed(111123)
        for points in (1, 2, 7, 37, 129):
            positions = torch.randn(2, points, 3, generator=generator)
            for metric, width in (("l1", 56), ("l2", 3)):
                values = torch.randn(2, points, width, generator=generator)
                original = greedy_neighbor_order(values, positions, metric=metric)
                for device in ("cpu", "input"):
                    candidate = greedy_neighbor_order(values, positions, metric=metric,
                                                      backend="row", distance_device=device)
                    torch.testing.assert_close(candidate.permutation, original.permutation, rtol=0, atol=0)

    @unittest.skipUnless(HAS_NUMBA, "optional numba unavailable")
    def test_knn_is_exact_across_candidate_counts_and_quantized_duplicate_values(self):
        generator = torch.Generator().manual_seed(9986)
        for points in (1, 2, 7, 37, 129):
            positions = torch.randn(2, points, 3, generator=generator)
            for metric, width in (("l1", 56), ("l2", 3)):
                for duplicate in (False, True):
                    values = torch.randn(2, points, width, generator=generator)
                    if duplicate:
                        values = values.round()
                        values[:, ::3] = 0
                    expected = greedy_neighbor_order(values, positions, metric=metric)
                    for k in (1, 2, 7, 64, 1000):
                        with self.subTest(points=points, metric=metric, duplicate=duplicate, k=k):
                            actual = greedy_neighbor_order(values, positions, metric=metric,
                                                           backend="knn", candidates=k)
                            torch.testing.assert_close(actual.permutation, expected.permutation, rtol=0, atol=0)
                            torch.testing.assert_close(actual.restore(actual.apply(values)), values, rtol=0, atol=0)

    @unittest.skipUnless(HAS_NUMBA, "optional numba unavailable")
    def test_boundary_tie_fallback_handles_topk_excluding_lowest_slot(self):
        distance = np.zeros((5, 5), dtype=np.float32)
        np.fill_diagonal(distance, np.inf)
        # A legal topk result in an all-tied row may omit slot 1 entirely.
        neighbors = np.array([[4], [4], [4], [4], [3]], dtype=np.int64)
        candidate_distances = np.zeros((5, 1), dtype=np.float32)
        order, _, ties = _compiled_greedy_knn()(distance, neighbors, candidate_distances, 0)
        np.testing.assert_array_equal(order, np.arange(5))
        self.assertGreater(ties, 0)

    @unittest.skipUnless(HAS_NUMBA, "optional numba unavailable")
    def test_exhausted_candidates_search_full_row_and_keep_all_tokens(self):
        points = torch.tensor([[0.], [2.], [1.], [4.]])
        distance = torch.cdist(points, points, p=1)
        neighbors, candidate_distances = _nearest_candidates(distance, 1)
        order, exhausted, _ = _compiled_greedy_knn()(
            distance.numpy(), neighbors.numpy(), candidate_distances.numpy(), 0)
        np.testing.assert_array_equal(order, [0, 2, 1, 3])
        self.assertGreater(exhausted, 0)

    @unittest.skipUnless(HAS_NUMBA, "optional numba unavailable")
    def test_graph_equals_exact_when_lists_cover_all_tokens(self):
        generator = torch.Generator().manual_seed(403)
        for points in (2, 7, 37, 129):
            positions = torch.randn(2, points, 3, generator=generator)
            for metric, width in (("l1", 56), ("l2", 3)):
                values = torch.randn(2, points, width, generator=generator)
                expected = greedy_neighbor_order(values, positions, metric=metric)
                for device in ("cpu", "input"):
                    actual = greedy_neighbor_order(values, positions, metric=metric, backend="graph",
                                                   candidates=1000, distance_device=device)
                    torch.testing.assert_close(actual.permutation, expected.permutation, rtol=0, atol=0)

    @unittest.skipUnless(HAS_NUMBA, "optional numba unavailable")
    def test_graph_short_lists_give_bijections_from_the_morton_start(self):
        generator = torch.Generator().manual_seed(404)
        for points in (1, 2, 7, 129, 1000):
            positions = torch.randn(2, points, 3, generator=generator)
            values = torch.randn(2, points, 56, generator=generator)
            values[:, ::5] = values[:, ::5].round()
            for k in (1, 4, 64):
                order = greedy_neighbor_order(values, positions, backend="graph", candidates=k)
                for scene in range(2):
                    permutation = order.permutation[scene]
                    self.assertTrue(torch.equal(permutation.sort().values, torch.arange(points)))
                    self.assertEqual(int(permutation[0]), int(morton_codes_3d(positions, 10)[scene].argmin()))
                torch.testing.assert_close(order.restore(order.apply(values)), values, rtol=0, atol=0)

    @unittest.skipUnless(HAS_NUMBA, "optional numba unavailable")
    def test_graph_exhausted_list_jumps_to_next_unvisited_on_the_curve(self):
        # Tie-free line; with one candidate, 1, 2 and 5 point back to visited tokens.
        values = torch.tensor([[0.], [1.], [2.5], [10.], [11.5], [4.5]])
        curve = np.array([0, 1, 2, 5, 3, 4], dtype=np.int64)
        rank = np.empty_like(curve)
        rank[curve] = np.arange(len(curve))
        neighbors, distances = _graph_candidates(values, "l1", 1)
        order, jumps = _compiled_greedy_graph()(neighbors.numpy(), distances.numpy(), curve, rank, 0)
        np.testing.assert_array_equal(order, [0, 1, 2, 5, 3, 4])
        self.assertEqual(jumps, 3)
        # Wraps around the curve when nothing unvisited lies after the current token.
        order, jumps = _compiled_greedy_graph()(neighbors.numpy(), distances.numpy(), curve[::-1].copy(),
                                               rank.max() - rank, 3)
        self.assertEqual(sorted(order.tolist()), list(range(6)))
        self.assertGreater(jumps, 0)

    def test_graph_candidates_match_dense_topk_in_row_blocks(self):
        generator = torch.Generator().manual_seed(405)
        scene = torch.randn(1100, 56, generator=generator)
        distance = torch.cdist(scene, scene, p=1, compute_mode="donot_use_mm_for_euclid_dist")
        expected_indices, expected_values = _nearest_candidates(distance.clone(), 64)
        indices, values = _graph_candidates(scene, "l1", 64)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        # A full shortlist finds the same L1 neighbors (summation order differs from cdist).
        indices, values = _graph_candidates(scene, "l1", 64, shortlist=1099)
        torch.testing.assert_close(values, expected_values, rtol=1e-5, atol=1e-4)
        indices, values = _graph_candidates(scene, "l1", 16, shortlist=32)
        torch.testing.assert_close(values, distance.gather(1, indices), rtol=1e-5, atol=1e-4)
        self.assertTrue((indices != torch.arange(1100)[:, None]).all())
        self.assertTrue((values[:, 1:] >= values[:, :-1]).all())
        # L2 lists come from one GEMM per block and rank by squared distance.
        euclid = torch.cdist(scene, scene, compute_mode="donot_use_mm_for_euclid_dist")
        expected_indices, expected_values = _nearest_candidates(euclid, 64)
        indices, values = _graph_candidates(scene, "l2", 64)
        torch.testing.assert_close(values.sqrt(), expected_values, rtol=1e-4, atol=1e-4)
        self.assertGreater((indices == expected_indices).float().mean().item(), .99)

    def test_score_order_metric_selects_the_nn_score_distance(self):
        CodecConfig(token_order="nn_score", score_order_metric="l2")
        for kwargs in (dict(score_order_metric="l2"), dict(token_order="nn_score", score_order_metric="cosine")):
            with self.assertRaises(ValueError):
                CodecConfig(**kwargs)
        # L1 and L2 disagree on which of two tokens is closer to the origin.
        values = torch.tensor([[[0., 0.], [1., 1.], [0., 1.8]]])
        positions = torch.zeros(1, 3, 3)
        l1 = greedy_neighbor_order(values, positions, metric="l1").permutation[0].tolist()
        l2 = greedy_neighbor_order(values, positions, metric="l2").permutation[0].tolist()
        self.assertEqual(l1, [0, 2, 1])
        self.assertEqual(l2, [0, 1, 2])

    @unittest.skipUnless(HAS_NUMBA, "optional numba unavailable")
    def test_graph_gemm_tours_are_bijections_close_to_exact_on_structured_scores(self):
        generator = torch.Generator().manual_seed(406)
        positions = torch.rand(2, 1024, 3, generator=generator)
        values = torch.sin(positions @ torch.randn(3, 56, generator=generator) * 3)
        exact = greedy_neighbor_order(values, positions, backend="row")
        approx = greedy_neighbor_order(values, positions, backend="graph_gemm", candidates=32)
        for scene in range(2):
            permutation = approx.permutation[scene]
            self.assertTrue(torch.equal(permutation.sort().values, torch.arange(1024)))
            self.assertEqual(int(permutation[0]), int(exact.permutation[scene, 0]))

    def test_nonfinite_invalid_options_and_empty_shapes_fail_early(self):
        values = torch.ones(1, 5, 3)
        for kwargs in (dict(backend="bad"), dict(candidates=0), dict(candidates=True),
                       dict(distance_device="cuda"), dict(distance_device="input"), dict(metric="bad")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                greedy_neighbor_order(values, values, **kwargs)
        for shape in ((0, 5, 3), (1, 0, 3), (1, 5, 0)):
            with self.assertRaises(ValueError):
                greedy_neighbor_order(torch.empty(shape), values)
        for bad in (float("inf"), float("nan")):
            changed = values.clone()
            changed[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                greedy_neighbor_order(changed, values, backend="row")
            with self.assertRaises(ValueError):
                greedy_neighbor_order(values, changed, backend="row")
        for kwargs in (dict(nn_order_backend="knn"), dict(nn_order_backend="graph"),
                       dict(token_order="nn_score", nn_order_candidates=0),
                       dict(token_order="nn_xyz", nn_order_distance_device="input")):
            with self.assertRaises(ValueError):
                CodecConfig(**kwargs)
        CodecConfig(token_order="nn_score", nn_order_backend="graph", nn_order_distance_device="input")

    def test_autocast_disabled_sort_keeps_feature_gradients(self):
        values = torch.randn(2, 17, 56, requires_grad=True)
        positions = torch.randn(2, 17, 3)
        expected = greedy_neighbor_order(values, positions)
        backends = ("row", "knn", "graph") if HAS_NUMBA else ("row",)
        for backend in backends:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = greedy_neighbor_order(values, positions, backend=backend)
            torch.testing.assert_close(actual.permutation, expected.permutation, rtol=0, atol=0)
            actual.apply(values).square().sum().backward()
            self.assertTrue(torch.isfinite(values.grad).all())
            self.assertGreater(values.grad.abs().sum().item(), 0)
            values.grad = None

    @unittest.skipUnless(HAS_NUMBA and torch.cuda.is_available(), "CUDA + numba required")
    def test_cuda_knn_matches_the_same_gpu_matrix_including_ties(self):
        for metric, width in (("l1", 56), ("l2", 3)):
            values = torch.randint(0, 4, (2, 257, width), device="cuda").float()
            positions = torch.zeros(2, 257, 3, device="cuda")
            distance = torch.cdist(values, values, p=1 if metric == "l1" else 2,
                                   compute_mode="donot_use_mm_for_euclid_dist")
            expected = torch.from_numpy(np.stack([_greedy_row_order(row, 0) for row in distance.cpu().numpy()])).cuda()
            for k in (1, 64, 256):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = greedy_neighbor_order(values, positions, metric=metric, backend="knn",
                                                   candidates=k, distance_device="input")
                torch.testing.assert_close(actual.permutation, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
