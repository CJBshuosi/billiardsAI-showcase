from cueorbit_replay.rules.query import expand_query


def test_english_is_unchanged_and_chinese_adds_visible_domain_terms():
    english = "Must I call the eight-ball pocket?"
    assert expand_query(english) == english
    chinese = "\u5f00\u7403\u6253\u8fdb8\u53f7\u7403\uff0c\u9700\u8981\u6307\u888b\u5417\uff1f"
    expanded = expand_query(chinese)
    assert expanded.endswith(chinese)
    assert (
        "break shot" in expanded
        and "eight ball" in expanded
        and "call shot" in expanded
    )


def test_unmapped_text_is_not_translated_or_assigned_a_confidence():
    question = "\u6ca1\u6709\u89c4\u5219\u4e3b\u9898"
    assert expand_query(question) == question
