from billiards_ai.lab.training.identity_review_page import render_identity_review_page


def test_review_page_requires_group_and_makes_exact_number_optional() -> None:
    page = render_identity_review_page()

    for group in ("cue", "solid", "eight", "stripe", "unknown"):
        assert f'data-group="{group}"' in page
    assert 'id="exact-identity"' in page
    assert 'id="exact-evidence"' in page
    assert "球组（必选）" in page
    assert "精确号码（可选）" in page
    assert "preloadNext" in page
    assert "/api/session" in page
    assert "/api/decision" in page
    assert "保存并查看下一颗" in page


def test_review_page_can_revisit_and_overwrite_an_existing_decision() -> None:
    page = render_identity_review_page()

    assert 'id="previous"' in page
    assert 'id="next"' in page
    assert 'id="jump"' in page
    assert 'id="jump-button"' in page
    assert "goTo(index-1)" in page
    assert "goTo(index+1)" in page
    assert "goTo(requested-1)" in page
    assert "prior=session.decisions[id]" in page


def test_review_page_shows_model_error_context_without_counting_suggestion_as_reviewed() -> None:
    page = render_identity_review_page()

    assert 'id="model-context"' in page
    assert "session.decisions[id]||session.suggestions?.[id]" in page
    assert "context.truth" in page
    assert "context.predicted" in page
    assert "context.confidence" in page
