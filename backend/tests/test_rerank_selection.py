import unittest

from app.rag import select_top_candidates


class TestRerankSelection(unittest.TestCase):
    def test_select_top_candidates_orders_by_score_desc(self) -> None:
        candidates = [
            {"id": "a", "score": 1.0},
            {"id": "b", "score": 9.5},
            {"id": "c", "score": 5.2},
            {"id": "d", "score": 8.8},
        ]

        selected = select_top_candidates(candidates, top_k=3)
        self.assertEqual([row["id"] for row in selected], ["b", "d", "c"])

    def test_select_top_candidates_is_stable_for_ties(self) -> None:
        candidates = [
            {"id": "x", "score": 7.0},
            {"id": "y", "score": 7.0},
            {"id": "z", "score": 7.0},
        ]

        selected = select_top_candidates(candidates, top_k=2)
        # Tie should preserve original order.
        self.assertEqual([row["id"] for row in selected], ["x", "y"])

    def test_select_top_candidates_respects_top_k(self) -> None:
        candidates = [
            {"id": "a", "score": 10.0},
            {"id": "b", "score": 9.0},
            {"id": "c", "score": 8.0},
        ]

        selected = select_top_candidates(candidates, top_k=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "a")


if __name__ == "__main__":
    unittest.main()

