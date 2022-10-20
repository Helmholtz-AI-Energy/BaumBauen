import unittest
import tempfile
from pathlib import Path
import numpy as np

from baumbauen.data.datasets import PhasespaceSet, generate_phasespace


class PhasespaceGenerationTest(unittest.TestCase):
    def test_reproducible(self):
        with tempfile.TemporaryDirectory() as root1:
            with tempfile.TemporaryDirectory() as root2:

                root1 = Path(root1)
                root2 = Path(root2)

                generate_phasespace(root1, [7., 3.], [1.], topologies=10, max_depth=3, max_children=2, min_children=2, train_events_per_top=3, val_events_per_top=1, test_events_per_top=1, seed=42)
                trds1 = PhasespaceSet(root1 / 'known', 'train')
                vads1 = PhasespaceSet(root1 / 'known', 'val')
                teds1 = PhasespaceSet(root1 / 'known', 'test')
                vads1_unknown = PhasespaceSet(root1 / 'unknown', 'val')
                teds1_unknown = PhasespaceSet(root1 / 'unknown', 'test')

                generate_phasespace(root2, [7., 3.], [1.], topologies=10, max_depth=3, max_children=2, min_children=2, train_events_per_top=3, val_events_per_top=1, test_events_per_top=1, seed=42)
                trds2 = PhasespaceSet(root2 / 'known', 'train')
                vads2 = PhasespaceSet(root2 / 'known', 'val')
                teds2 = PhasespaceSet(root2 / 'known', 'test')
                vads2_unknown = PhasespaceSet(root2 / 'unknown', 'val')
                teds2_unknown = PhasespaceSet(root2 / 'unknown', 'test')

                for i in range(len(trds1)):
                    self.assertTrue((trds1[i][0] == trds2[i][0]).all())
                    self.assertTrue((trds1[i][1] == trds2[i][1]).all())

                for i in range(len(vads1)):
                    self.assertTrue((vads1[i][0] == vads2[i][0]).all())
                    self.assertTrue((vads1[i][1] == vads2[i][1]).all())
                    self.assertTrue((vads1_unknown[i][0] == vads2_unknown[i][0]).all())
                    self.assertTrue((vads1_unknown[i][1] == vads2_unknown[i][1]).all())

                for i in range(len(teds1)):
                    self.assertTrue((teds1[i][0] == teds2[i][0]).all())
                    self.assertTrue((teds1[i][1] == teds2[i][1]).all())
                    self.assertTrue((teds1_unknown[i][0] == teds2_unknown[i][0]).all())
                    self.assertTrue((teds1_unknown[i][1] == teds2_unknown[i][1]).all())

    def test_different(self):
        with tempfile.TemporaryDirectory() as root1:
            with tempfile.TemporaryDirectory() as root2:

                root1 = Path(root1)
                root2 = Path(root2)

                generate_phasespace(root1, [7., 3.], [1.], topologies=5, max_depth=3, max_children=2, min_children=2, train_events_per_top=5, val_events_per_top=2, test_events_per_top=2, seed=42)
                trds1 = PhasespaceSet(root1 / 'known', 'train')
                vads1 = PhasespaceSet(root1 / 'known', 'val')
                teds1 = PhasespaceSet(root1 / 'known', 'test')
                vads1_unknown = PhasespaceSet(root1 / 'unknown', 'val')
                teds1_unknown = PhasespaceSet(root1 / 'unknown', 'test')

                generate_phasespace(root2, [7., 3.], [1.], topologies=5, max_depth=3, max_children=2, min_children=2, train_events_per_top=5, val_events_per_top=2, test_events_per_top=2, seed=43)
                trds2 = PhasespaceSet(root2 / 'known', 'train')
                vads2 = PhasespaceSet(root2 / 'known', 'val')
                teds2 = PhasespaceSet(root2 / 'known', 'test')
                vads2_unknown = PhasespaceSet(root2 / 'unknown', 'val')
                teds2_unknown = PhasespaceSet(root2 / 'unknown', 'test')

                # NOTE we don't test the labels to be different here, because for small tree depth labels will sometimes be identical randomly
                # NOTE curiously some but not all samples are identical

                for i in range(len(trds1)):
                    if trds1[i][0].shape == trds2[i][0].shape:
                        if not (trds1[i][0] == trds2[i][0]).all():
                            return
                    else:
                        return

                for i in range(len(vads1)):
                    if vads1[i][0].shape == vads2[i][0].shape:
                        if not (vads1[i][0] == vads2[i][0]).all():
                            return
                    else:
                        return

                    if vads1_unknown[i][0].shape == vads2_unknown[i][0].shape:
                        if not (vads1_unknown[i][0] == vads2_unknown[i][0]).all():
                            return
                    else:
                        return

                for i in range(len(teds1)):
                    if teds1[i][0].shape == teds2[i][0].shape:
                        if not (teds1[i][0] == teds2[i][0]).all():
                            return
                    else:
                        return

                    if teds1_unknown[i][0].shape == teds2_unknown[i][0].shape:
                        if not (teds1_unknown[i][0] == teds2_unknown[i][0]).all():
                            return
                    else:
                        return

                self.fail()

    def test_isomorphism(self):
        with tempfile.TemporaryDirectory() as root:
            generate_phasespace(root, [39., 19., 9., 4.], [1., 2.], topologies=2, max_depth=6, max_children=5, min_children=2, train_events_per_top=2, val_events_per_top=2, test_events_per_top=2, seed=42, iso_retries=10)

    def test_isomorphism_fail(self):
        try:
            with tempfile.TemporaryDirectory() as root:
                generate_phasespace(root, [7.], [1.], topologies=2, max_depth=2, max_children=2, min_children=2, train_events_per_top=2, val_events_per_top=2, test_events_per_top=2, seed=42, iso_retries=2)

        except RuntimeError:
            return

        self.fail()

    def test_phasespace_samples(self):
        with tempfile.TemporaryDirectory() as root1:

            root1 = Path(root1)

            generate_phasespace(
                root1,
                [7., 3.],
                [1.],
                topologies=5,
                max_depth=3,
                max_children=2,
                min_children=2,
                train_events_per_top=10,
                val_events_per_top=1,
                test_events_per_top=1,
                seed=42
            )

            samples = 3
            trds1 = PhasespaceSet(
                root1 / 'known',
                'train',
                samples=samples,
            )

            self.assertTrue(trds1.samples_per_topology == 1)
            self.assertTrue(len(trds1.x) == samples)

    def test_phasespace_samples_per_topology(self):
        with tempfile.TemporaryDirectory() as root1:

            root1 = Path(root1)

            generate_phasespace(
                root1,
                [7., 3.],
                [1.],
                topologies=5,
                max_depth=3,
                max_children=2,
                min_children=2,
                train_events_per_top=10,
                val_events_per_top=1,
                test_events_per_top=1,
                seed=42
            )

            samples_per_topology = 7
            trds1 = PhasespaceSet(
                root1 / 'known',
                'train',
                samples_per_topology=samples_per_topology,
            )

            self.assertTrue((trds1.samples_per_topology == samples_per_topology).all())
            self.assertTrue(np.array([f.shape[0] == samples_per_topology for f in trds1.x]).all())


if __name__ == '__main__':
    unittest.main()
