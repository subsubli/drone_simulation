import unittest
import numpy as np
from path_progress import PathProgress


class ProgressTests(unittest.TestCase):
    def path(self, eight, direction):
        def at(t):
            phase = direction*2*np.pi*t
            return np.array([np.sin(phase), np.sin(2*phase)/2 if eight else np.cos(phase), 2.]), None
        return at

    def test_forward_reverse_and_crossings(self):
        for eight in (False, True):
            for direction in (-1, 1):
                at = self.path(eight, direction)
                for turns in (-2.3, 2.3):
                    tracker = PathProgress(at, 1., at(0)[0])
                    for t in np.linspace(0, turns, 2301)[1:]:
                        tracker.update(at(t)[0])
                    self.assertAlmostEqual(tracker.phase/tracker.n, turns, places=4)
                    self.assertEqual(tracker.rejected, 0)

    def test_stationary_and_pause_at_crossing(self):
        at = self.path(True, 1)
        tracker = PathProgress(at, 1., at(0)[0])
        for _ in range(2000):
            tracker.update(at(0)[0])
        self.assertEqual(tracker.metrics(20, 20, 1)['net_laps'], 0.)
        for t in np.linspace(0, .5, 501)[1:]:
            tracker.update(at(t)[0])
        for _ in range(2000):
            tracker.update(at(.5)[0])
        self.assertAlmostEqual(tracker.phase/tracker.n, .5, places=4)

    def test_out_and_back_cancels(self):
        at = self.path(True, 1)
        tracker = PathProgress(at, 1., at(0)[0])
        for t in np.r_[np.linspace(0,.8,801),np.linspace(.8,0,801)]:
            tracker.update(at(t)[0])
        self.assertAlmostEqual(tracker.phase/tracker.n, 0., places=4)


if __name__ == '__main__':
    unittest.main()
