from types import SimpleNamespace

import pytest

from wp_tracker.instagram.browser import CollectionIncomplete, _restore_profile


PROFILE = 'https://www.instagram.com/adalat.muradov/'
POST = 'https://www.instagram.com/adalat.muradov/p/DfOgblatrrx/'


class FakeLocator:
    def __init__(self, page, kind):
        self.page = page
        self.kind = kind
        self.first = self

    def wait_for(self, state=None, timeout=None):
        self.page.waits.append((self.kind, state, timeout))
        if self.page.grid_available is False and self.kind == 'grid':
            raise RuntimeError('post grid did not load')

    def inner_text(self, timeout=None):
        return ''


class FakeKeyboard:
    def __init__(self, page):
        self.page = page

    def press(self, key, timeout=None):
        self.page.actions.append(('press', key))
        if self.page.escape_error:
            raise RuntimeError('escape failed')
        if self.page.escape_to_profile:
            self.page.url = PROFILE


class FakePage:
    def __init__(self, *, url=POST, escape_to_profile=False, history_to_profile=False,
                 history_error=None, goto_to_profile=True, goto_error=None,
                 grid_available=True, escape_error=None):
        self.url = url
        self.escape_to_profile = escape_to_profile
        self.history_to_profile = history_to_profile
        self.history_error = history_error
        self.goto_to_profile = goto_to_profile
        self.goto_error = goto_error
        self.grid_available = grid_available
        self.escape_error = escape_error
        self.actions = []
        self.waits = []
        self.scroll_values = []
        self.keyboard = FakeKeyboard(self)

    def locator(self, selector):
        return FakeLocator(self, 'body' if selector == 'body' else 'grid')

    def wait_for_timeout(self, milliseconds):
        self.actions.append(('wait', milliseconds))

    def go_back(self, **kwargs):
        self.actions.append(('go_back', kwargs))
        if self.history_error:
            raise RuntimeError(self.history_error)
        if self.history_to_profile:
            self.url = PROFILE

    def goto(self, url, **kwargs):
        self.actions.append(('goto', url, kwargs))
        if self.goto_error:
            raise RuntimeError(self.goto_error)
        if self.goto_to_profile:
            self.url = url

    def evaluate(self, expression, value):
        self.actions.append(('evaluate', expression))
        self.scroll_values.append(value)


def test_restore_profile_closes_modal_with_escape_and_restores_scroll():
    page = FakePage(escape_to_profile=True)

    _restore_profile(page, PROFILE, 1875)

    assert page.url == PROFILE
    assert page.scroll_values == [1875]
    assert not any(action[0] == 'go_back' for action in page.actions)
    assert page.waits and page.waits[-1][0] == 'grid'


def test_restore_profile_returns_from_full_page_post_using_history():
    page = FakePage(history_to_profile=True)

    _restore_profile(page, PROFILE, 920)

    assert page.url == PROFILE
    assert page.scroll_values == [920]
    assert any(action[0] == 'go_back' for action in page.actions)


def test_restore_profile_reloads_after_history_failure():
    page = FakePage(history_error='history unavailable')

    _restore_profile(page, PROFILE, 440)

    assert page.url == PROFILE
    assert page.scroll_values == [440]
    assert [action[0] for action in page.actions].count('goto') == 1


def test_restore_profile_reports_all_navigation_failures():
    page = FakePage(
        history_error='history unavailable',
        goto_to_profile=False,
        goto_error='profile reload failed',
    )

    with pytest.raises(CollectionIncomplete, match='profile reload failed'):
        _restore_profile(page, PROFILE, 10)

    assert [action[0] for action in page.actions].count('goto') == 2
    assert page.scroll_values == []
