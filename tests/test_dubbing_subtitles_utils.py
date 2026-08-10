import unittest

from dubbing_service.app.models.subtitles import SubtitleFile, SubtitleSegment, SubtitleWord
from dubbing_service.app.providers.subtitles import (
    _collect_subtitle_issues,
    _join_words_text,
    _merge_punctuation_words,
)


class TestSubtitleTextJoin(unittest.TestCase):
    def test_join_en_contraction(self) -> None:
        words = [
            SubtitleWord(index=0, text="don", start=0.0, end=0.1),
            SubtitleWord(index=1, text="'t", start=0.1, end=0.2),
        ]
        self.assertEqual(_join_words_text(words, language="en"), "don't")

    def test_join_en_punct_spacing(self) -> None:
        words = [
            SubtitleWord(index=0, text="Hello", start=0.0, end=0.1),
            SubtitleWord(index=1, text=",", start=0.1, end=0.1),
            SubtitleWord(index=2, text="world", start=0.1, end=0.2),
            SubtitleWord(index=3, text="!", start=0.2, end=0.2),
        ]
        self.assertEqual(_join_words_text(words, language="en"), "Hello, world!")

    def test_join_en_open_quote_after_space(self) -> None:
        words = [
            SubtitleWord(index=0, text="said", start=0.0, end=0.1),
            SubtitleWord(index=1, text="'", start=0.1, end=0.1),
            SubtitleWord(index=2, text="Hello", start=0.1, end=0.2),
        ]
        self.assertEqual(_join_words_text(words, language="en"), "said 'Hello")


class TestSubtitlePunctuationMerge(unittest.TestCase):
    def test_merge_open_quote_to_next(self) -> None:
        seg = SubtitleSegment(
            index=0,
            start=0.0,
            end=1.0,
            text='"Hello"',
            asr_text='"Hello"',
            words=[
                SubtitleWord(index=0, text='"', start=0.0, end=0.0),
                SubtitleWord(index=1, text="Hello", start=0.0, end=0.5),
                SubtitleWord(index=2, text='"', start=0.5, end=0.5),
            ],
        )
        _merge_punctuation_words(seg, language="en", update_asr_text=True)
        self.assertEqual([w.text for w in seg.words], ['"Hello"'])
        self.assertEqual(seg.asr_text, '"Hello"')

    def test_merge_suffix_punct_to_prev(self) -> None:
        seg = SubtitleSegment(
            index=0,
            start=0.0,
            end=1.0,
            text="Hello.",
            asr_text="Hello.",
            words=[
                SubtitleWord(index=0, text="Hello", start=0.0, end=0.4),
                SubtitleWord(index=1, text=".", start=0.4, end=0.4),
            ],
        )
        _merge_punctuation_words(seg, language="en", update_asr_text=True)
        self.assertEqual([w.text for w in seg.words], ["Hello."])
        self.assertEqual(seg.asr_text, "Hello.")


class TestSubtitleValidation(unittest.TestCase):
    def test_collect_issues_duplicate_word_index(self) -> None:
        subtitle = SubtitleFile(
            id="s1",
            project_id="p1",
            audio_rel_path="audio/final/a.wav",
            audio_filename="a.wav",
            language="en",
            segments=[
                SubtitleSegment(
                    index=0,
                    start=0.0,
                    end=1.0,
                    text="a",
                    asr_text="a",
                    words=[SubtitleWord(index=0, text="a", start=0.0, end=0.5)],
                ),
                SubtitleSegment(
                    index=1,
                    start=1.0,
                    end=2.0,
                    text="b",
                    asr_text="b",
                    words=[SubtitleWord(index=0, text="b", start=1.0, end=1.5)],
                ),
            ],
        )
        issues = _collect_subtitle_issues(subtitle, context="t")
        self.assertTrue(any("duplicate_word_index" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
