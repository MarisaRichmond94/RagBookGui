import unittest

from app.timeline import extract_timeline_fields


class TestTimelineExtraction(unittest.TestCase):
    def test_extracts_structured_fields(self) -> None:
        text = (
            "Jared Gatlin arrived at Harbor Station. "
            "He discovered a hidden ledger in Boston and realized Nora was involved. "
            "Later, Nora confessed everything near City Hall."
        )
        result = extract_timeline_fields(text, pov="Jared Gatlin")

        self.assertIn("key_events", result)
        self.assertIn("locations", result)
        self.assertIn("characters_present", result)
        self.assertGreater(len(result["key_events"]), 0)
        self.assertTrue(any("Boston" in loc for loc in result["locations"]))
        self.assertTrue(any("Jared" in name for name in result["characters_present"]))


if __name__ == "__main__":
    unittest.main()

