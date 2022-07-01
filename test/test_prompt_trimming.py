from lmwrapper.prompt_trimming import CharPromptTrimmer


def test_char_simple():
    trimmer = CharPromptTrimmer(3)
    result = trimmer.trim_text("abcdefghijklmnopqrstuvwxyz")
    assert result == "xyz"


def test_trim_segmented():
    trimmer = CharPromptTrimmer(7)
    result = trimmer.trim_text(["abc", "def", "ghi"])
    assert result == "defghi"


def test_trim_lines():
    trimmer = CharPromptTrimmer(10)
    result = trimmer.trim_text_line_level("abc\ndef\nghib\nabc")
    assert result == "ghib\nabc"
