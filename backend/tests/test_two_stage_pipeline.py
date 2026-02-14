import unittest

from app.rag import _build_where_filter
from app.rag import _chapter_in_scope
from app.rag import is_global_question
from app.rag import should_use_summaries_first


class TestTwoStagePipeline(unittest.TestCase):
    def test_global_question_classifier(self) -> None:
        self.assertTrue(is_global_question("Who are the primary and secondary characters?"))
        self.assertTrue(is_global_question("What are the major themes across the book?"))
        self.assertFalse(is_global_question("What happens in chapter 7?"))

    def test_should_use_summaries_first_toggle_or_classifier(self) -> None:
        self.assertTrue(should_use_summaries_first("What are the themes?", summaries_first=False))
        self.assertTrue(should_use_summaries_first("What happens in chapter 2?", summaries_first=True))
        self.assertFalse(should_use_summaries_first("What happens in chapter 2?", summaries_first=False))

    def test_stage1_scope_filters_stage2_candidates(self) -> None:
        scope = {("Faded", "04_4.txt"), ("Faded", "05_5.txt")}

        self.assertTrue(_chapter_in_scope({"book": "Faded", "chapter_file": "04_4.txt"}, scope))
        self.assertFalse(_chapter_in_scope({"book": "Faded", "chapter_file": "09_9.txt"}, scope))
        self.assertFalse(_chapter_in_scope({"book": "Other", "chapter_file": "04_4.txt"}, scope))

    def test_where_filter_includes_chapter_scope(self) -> None:
        where = _build_where_filter(
            books=["Faded"],
            pov="Jared Gatlin",
            chapter_files=["04_4.txt", "05_5.txt"],
        )
        self.assertIsInstance(where, dict)
        self.assertIn("$and", where)


if __name__ == "__main__":
    unittest.main()

