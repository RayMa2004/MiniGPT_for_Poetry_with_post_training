import unittest
from scheduler import WarmupCosine


class SchedulerTests(unittest.TestCase):
    def test_endpoints_and_state(self):
        class Optimizer:
            param_groups = [{"lr": 1.0}]
        optimizer = Optimizer()
        scheduler = WarmupCosine(optimizer, 1.0, 10, 2, 0.1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)
        self.assertAlmostEqual(scheduler.lr_for_step(10), 0.1)
        state = scheduler.state_dict()
        scheduler.step()
        scheduler.load_state_dict(state)
        self.assertEqual(scheduler.completed_steps, 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            WarmupCosine(type("O", (), {"param_groups": []})(), 1, 2, 2, 0)

    def test_zero_warmup_and_single_update(self):
        optimizer = type("O", (), {"param_groups": [{"lr": 1}]})()
        scheduler = WarmupCosine(optimizer, 1, 3, 0, 0.1)
        self.assertEqual(scheduler.lr_for_step(1), 1)
        self.assertAlmostEqual(scheduler.lr_for_step(3), 0.1)
        self.assertEqual(WarmupCosine(optimizer, 1, 1, 0, 0.1).lr_for_step(1), 1)
