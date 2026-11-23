"""Offline tests for the Instagram date boundary (cutoff = newest collected post)."""
import pytest

from wp_tracker.instagram.browser import _boundary_at_profile_exhaustion, _date_boundary


def advance(flags, threshold=4):
    """flags: True for a non-pinned post not newer than the cutoff, False for a newer post."""
    run = 0
    history = []
    for is_old in flags:
        run, accepted = _date_boundary(run, is_old, threshold)
        history.append((run, accepted))
        if accepted:
            break
    return history, run


def test_new_posts_then_four_old_posts_stop_the_scan():
    history, run = advance([False, False] + [True] * 10)
    assert history[-1] == (4, True)
    assert len(history) == 2 + 4


def test_three_old_posts_are_not_enough():
    # Up to three pinned posts may be old and undetected at the top of the grid.
    history, run = advance([True, True, True, False, False])
    assert not any(accepted for _, accepted in history)
    assert run == 0


def test_newer_post_resets_the_counter():
    history, _ = advance([True, True, True, False, True, True, True, True])
    assert [run for run, _ in history] == [1, 2, 3, 0, 1, 2, 3, 4]
    assert history[-1][1] is True


def test_threshold_is_configurable():
    history, _ = advance([True] * 10, threshold=6)
    assert len(history) == 6 and history[-1][1]


def test_profile_end_is_a_boundary_only_after_an_old_post():
    _, run = advance([False, True])
    assert _boundary_at_profile_exhaustion(run) is True
    _, run = advance([True, False])
    assert _boundary_at_profile_exhaustion(run) is False


def test_boundary_below_pinned_limit_is_rejected(tmp_path):
    from wp_tracker.instagram.config import load_ig_settings
    path = tmp_path / 'config/settings.yaml'
    path.parent.mkdir()
    path.write_text('instagram:\n  boundary_old_posts: 3\n', encoding='utf-8')
    with pytest.raises(ValueError, match='at least 4'):
        load_ig_settings(path, 'uneceduaz')
