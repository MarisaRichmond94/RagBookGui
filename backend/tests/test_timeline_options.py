import unittest

from app.timeline import _resolve_character_names


class TestTimelineOptions(unittest.TestCase):
    def test_first_name_is_upgraded_to_unique_full_name(self) -> None:
        names = ["Noah", "Noah Gatlin", "Avery", "Avery Brooks", "Nora"]
        resolved = _resolve_character_names(names, limit=100)

        self.assertIn("Noah Gatlin", resolved)
        self.assertNotIn("Noah", resolved)
        self.assertIn("Avery Brooks", resolved)
        self.assertNotIn("Avery", resolved)
        self.assertIn("Nora", resolved)

    def test_ambiguous_first_name_kept_when_multiple_full_names_exist(self) -> None:
        names = ["Noah", "Noah Gatlin", "Noah Carter"]
        resolved = _resolve_character_names(names, limit=100)

        self.assertIn("Noah", resolved)
        self.assertIn("Noah Gatlin", resolved)
        self.assertIn("Noah Carter", resolved)


if __name__ == "__main__":
    unittest.main()
