from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from dynamic_graph_provider import DynamicGraphUnavailableError, DynamicPCMCIProvider
from graphwavenet_dynamic_adapter import (
    BatchAwareNConv,
    DynamicGraphWaveNetAdapter,
    FullYearGraphWaveNetDataset,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned_us101_n_20_full_year"
GRAPH_DIR = PROJECT_DIR / "dynamic_graphs" / "per_window"
GRAPH_WAVENET_MODEL = (
    PROJECT_DIR.parent / "Graph-WaveNet-master" / "Graph-WaveNet-master" / "model.py"
)


class DynamicGraphInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = DynamicPCMCIProvider(
            DATA_DIR / "dynamic_pcmci_window_manifest.npz", GRAPH_DIR
        )

    def test_causal_boundary_mapping(self) -> None:
        with self.assertRaises(DynamicGraphUnavailableError):
            self.provider.resolve([2015])
        assignments = self.provider.resolve([2016, 2303, 2304, 2591])
        self.assertEqual([item.graph_slot for item in assignments], [0, 0, 1, 1])
        self.assertTrue(
            all(item.graph_time_index <= item.target_index for item in assignments)
        )

    def test_missing_future_graph_is_not_silently_reused(self) -> None:
        with self.assertRaisesRegex(DynamicGraphUnavailableError, "not generated"):
            self.provider.resolve([2592])

    def test_batched_supports(self) -> None:
        supports, assignments = self.provider.normalized_supports([2016, 2304])
        self.assertEqual(len(assignments), 2)
        self.assertEqual([support.shape for support in supports], [(2, 20, 20)] * 2)
        for support in supports:
            row_sums = support.sum(axis=-1)
            self.assertTrue(np.all((np.isclose(row_sums, 1.0)) | (row_sums == 0.0)))

    def test_batch_aware_convolution(self) -> None:
        convolution = BatchAwareNConv()
        x = torch.randn(2, 3, 20, 4)
        shared = torch.eye(20)
        batched = torch.stack([shared, shared], dim=0)
        self.assertTrue(torch.allclose(convolution(x, shared), x))
        self.assertTrue(torch.allclose(convolution(x, batched), x))

    def test_dataset_keeps_prediction_index(self) -> None:
        dataset = FullYearGraphWaveNetDataset(
            DATA_DIR / "full_year_cleaned.npz",
            DATA_DIR / "split_indices.npz",
            "train",
            provider=self.provider,
            only_generated_graphs=True,
        )
        x, y, target_index = dataset[0]
        self.assertEqual(tuple(x.shape), (12, 20, 3))
        self.assertEqual(tuple(y.shape), (12, 20))
        self.assertEqual(target_index, 2016)
        self.assertEqual(len(dataset), 576)

    @unittest.skipUnless(GRAPH_WAVENET_MODEL.exists(), "local Graph WaveNet source absent")
    def test_real_graphwavenet_forward(self) -> None:
        spec = importlib.util.spec_from_file_location("local_gwnet_model", GRAPH_WAVENET_MODEL)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        placeholders = [torch.eye(20), torch.eye(20)]
        base_model = module.gwnet(
            "cpu",
            20,
            supports=placeholders,
            gcn_bool=True,
            addaptadj=True,
            in_dim=3,
            out_dim=12,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=8,
            end_channels=16,
            blocks=1,
            layers=1,
        )
        model = DynamicGraphWaveNetAdapter(base_model, self.provider)
        history = torch.randn(2, 12, 20, 3)
        output = model.forward_history_batch(history, torch.tensor([2016, 2304]))
        self.assertIs(base_model.supports, placeholders)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 12)
        self.assertEqual(output.shape[2], 20)
        self.assertEqual([item.graph_slot for item in model.last_assignments], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
