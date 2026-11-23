"""Read a public Instagram profile through an already authenticated Playwright browser.

The scraper does not log in or defeat challenges. If authentication is rejected,
stop and let an operator refresh the session outside the server job.
"""
from __future__ import annotations
from datetime import datetime, timezone
import re
from urllib.parse import urlparse

from .schema import canonical_url, shortcode, utc_datetime


class AuthenticationRequired(RuntimeError):
    pass


class CollectionIncomplete(RuntimeError):
    pass


POST_SELECTOR = "a[href*='/p/'], a[href*='/reel/'], a[href*='/reels/']"


def _access_problem(page) -> bool:
    path = urlparse(page.url).path
    if '/accounts/login' in path or '/challenge/' in path or '/accounts/suspended' in path:
        return True
    try:
        body = page.locator('body').inner_text(timeout=1500)[:1400].casefold()
    except Exception:
        return False
    return any(term in body for term in (
        'please wait a few minutes before you try again',
        'we restrict certain activity', 'suspicious login attempt',
        'unusual activity',
    ))


def _network_dates(payload, username, output):
    """Accept dates ONLY from media explicitly attributed to the desired author."""
    if not isinstance(payload, (dict, list)):
        return
    stack = [payload]
    count = 0
    while stack and count < 50000:
        node = stack.pop(); count += 1
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            media = node.get('media') if isinstance(node.get('media'), dict) else node
            owner = media.get('user') or media.get('owner') or {}
            if (isinstance(owner, dict) and owner.get('username', '').lower() == username
                    and media.get('code') and media.get('taken_at')):
                try:
                    stamp = datetime.fromtimestamp(int(media['taken_at']), tz=timezone.utc)
                    output[str(media['code'])] = stamp
                except (ValueError, OverflowError, TypeError):
                    pass
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))


def _date_from_page(page, code: str, network_dates: dict) -> datetime | None:
    if code in network_dates:
        return network_dates[code]
    # In the opened post popup, the time element is the publication time.
    locators = [
        page.locator('div[role="dialog"] time[datetime]'),
        page.locator('article time[datetime]'),
    ]
    for locator in locators:
        try:
            for raw in locator.all()[:5]:
                iso = raw.get_attribute('datetime', timeout=1500)
                stamp = utc_datetime(iso)
                if stamp:
                    return stamp
        except Exception:
            continue
    return None


def _expand_caption(page) -> None:
    """Expand localized/accessibly-labelled caption controls when present.

    Failure to find a control is normal for a complete caption. Click failures
    are deliberately ignored here; the caller still validates the resulting
    caption and fails closed if it remains empty or truncated.
    """
    labels = {'more', 'see more', 'daha çox', 'daha fazla'}
    try:
        controls = page.locator('li:has(h1) button, li:has(h1) [role="button"]')
        for control in controls.all()[:20]:
            try:
                text = (control.inner_text(timeout=350) or '').strip().casefold()
                aria = (control.get_attribute('aria-label', timeout=350) or '').strip().casefold()
                title = (control.get_attribute('title', timeout=350) or '').strip().casefold()
                if any(value in labels for value in (text, aria, title)) and control.is_visible(timeout=350):
                    control.click(timeout=800)
            except Exception:
                continue
    except Exception:
        pass


def _caption_from_page(page, username: str) -> str:
    """Use caption h1 only when the surrounding li belongs to the target author."""
    _expand_caption(page)
    try:
        headings = page.locator('li:has(h1)')
        for li in headings.all()[:30]:
            author_names = [s.strip().lower() for s in li.locator('h2').all_inner_texts()]
            author_link = li.locator(f'a[href="/{username}/"]').count() > 0
            if username not in author_names and not author_link:
                continue
            for el in li.locator('h1').all()[:4]:
                if not el.is_visible():
                    continue
                value = el.inner_text(timeout=1500).strip()
                if value and value.casefold() != username and not re.search(
                    r'(?:…|\.{3})\s*(?:more|see more|daha çox|daha fazla)$', value, re.I
                ):
                    return value
    except Exception:
        pass
    return ''


def _profile_path(profile_url: str) -> str:
    return urlparse(profile_url).path.rstrip('/') or '/'


def _wait_for_profile(page, profile_url: str) -> None:
    """Verify both the profile URL and its post grid before resuming a scan."""
    actual_path = urlparse(page.url).path.rstrip('/') or '/'
    expected_path = _profile_path(profile_url)
    if actual_path != expected_path:
        raise CollectionIncomplete(
            f'Profile URL not restored: expected {expected_path!r}, got {actual_path!r}'
        )
    if _access_problem(page):
        raise AuthenticationRequired('Instagram profile is not accessible after post navigation')
    page.locator(POST_SELECTOR).first.wait_for(state='attached', timeout=15000)


def _grid_links(page, selector: str) -> list[dict]:
    """Return ordered grid links and best-effort pinned markers.

    Instagram does not expose a stable pinned-post API. The marker is therefore
    intentionally conservative and based on accessible/nearby DOM labels. An
    unmarked link is treated as normal grid content and must participate in the
    consecutive-history and verification rules below.
    """
    return page.locator(selector).evaluate_all("""els => els.map(a => {
        const labels = [];
        let node = a;
        for (let depth = 0; node && depth < 4; depth += 1, node = node.parentElement) {
            for (const attr of ['aria-label', 'title', 'data-testid']) {
                const value = node.getAttribute && node.getAttribute(attr);
                if (value) labels.push(value);
            }
            if (node.innerText) labels.push(node.innerText.slice(0, 240));
        }
        return {href: a.href, pinned: /\\b(pinned|sancaq)\\b/i.test(labels.join(' '))};
    })""")


def _date_boundary(old_run: int, is_old: bool, threshold: int) -> tuple[int, bool]:
    """Advance the consecutive old-post counter for one non-pinned grid post.

    A post is old when its date is not newer than the cutoff (the newest post
    already collected). The grid is newest-to-oldest, so ``threshold``
    consecutive old posts mark the boundary. Instagram allows at most three
    pinned posts, which may be old and are not always detectable, so the
    default threshold is 4. A newer post resets the counter.
    Returns ``(old_run, accepted)``.
    """
    old_run = old_run + 1 if is_old else 0
    return old_run, old_run >= threshold


def _boundary_at_profile_exhaustion(old_run: int) -> bool:
    """A profile end is a boundary only if the last ordered post was old.

    If loading stalled on newer posts, older new posts may be unseen; advancing
    the cutoff past them would lose them, so this fails closed.
    """
    return old_run > 0


def _restore_profile(page, profile_url: str, previous_y: int):
    """Return from either a post modal or full-page post with bounded recovery.

    Instagram may leave the browser on a profile-qualified post URL such as
    ``/<user>/p/<code>/`` after a click. Escape can close a modal, history can
    return from a document navigation, and a fresh profile load is the final
    recovery path. Every path validates the profile URL and post grid before
    restoring scroll, so a scan cannot continue from an unverified page.
    """
    errors = []

    def validate_and_restore(label: str) -> bool:
        try:
            _wait_for_profile(page, profile_url)
            page.evaluate('(y) => window.scrollTo(0,y)', previous_y)
            page.wait_for_timeout(400)
            return True
        except Exception as exc:
            errors.append(f'{label}: {type(exc).__name__}: {exc}')
            return False

    try:
        page.keyboard.press('Escape', timeout=3000)
        page.wait_for_timeout(250)
        if validate_and_restore('Escape/modal close'):
            return
    except Exception as exc:
        errors.append(f'Escape/modal close: {type(exc).__name__}: {exc}')

    try:
        page.go_back(wait_until='domcontentloaded', timeout=10000)
        if validate_and_restore('browser history'):
            return
    except Exception as exc:
        errors.append(f'browser history: {type(exc).__name__}: {exc}')

    for attempt in range(1, 3):
        try:
            page.goto(profile_url, wait_until='domcontentloaded', timeout=30000)
            if validate_and_restore(f'profile reload {attempt}'):
                return
        except Exception as exc:
            errors.append(f'profile reload {attempt}: {type(exc).__name__}: {exc}')

    detail = '; '.join(errors[-4:]) or 'no navigation attempt completed'
    raise CollectionIncomplete(
        f'Cannot return to profile {profile_url!r} after post; {detail}'
    )


def prepare_auth(settings):
    """One-time operator step with visible browser; NOT used during scheduled scan."""
    from playwright.sync_api import sync_playwright
    settings.auth_file.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        try:
            context = browser.new_context(locale='en-US')
            page = context.new_page()
            page.goto(f'https://www.instagram.com/{settings.username}/', wait_until='domcontentloaded', timeout=60000)
            print('Log into Instagram in the opened browser and verify the profile grid.')
            input('Press ENTER after the profile is visible: ')
            if _access_problem(page) or not page.locator("a[href*='/p/'], a[href*='/reel/']").count():
                raise AuthenticationRequired('Session not authenticated or public posts not visible')
            context.storage_state(path=str(settings.auth_file))
            try:
                settings.auth_file.chmod(0o600)
            except OSError:
                pass
            print(f'Session state saved: {settings.auth_file} (KEEP SECRET)')
        finally:
            browser.close()


def collect(settings, known_dates: dict, cutoff: datetime, on_new, *, headed=False, max_scrolls=None, on_skip=None):
    """Call on_new(post_url, caption, datetime) for posts newer than ``cutoff``.

    ``cutoff`` is the date of the newest post already collected. ``known_dates``
    maps every already known URL (snapshot, pending, skipped) to its date, or
    None if unknown; known posts are never reopened. Unknown posts are opened
    to read their date. The scan stops after ``boundary_old_posts`` consecutive
    non-pinned posts dated at or before the cutoff. Pinned posts are still
    checked for newness but never move the boundary. Raises on incomplete
    searches, challenges, or new-post parsing failures.
    """
    if not settings.auth_file.is_file():
        raise AuthenticationRequired(f'Missing auth state: {settings.auth_file}. Run instagram login first.')
    from playwright.sync_api import sync_playwright
    max_scrolls = max_scrolls or settings.max_scrolls
    threshold = settings.boundary_old_posts
    profile_url = f'https://www.instagram.com/{settings.username}/'
    found = set(); network_dates = {}; old_seen = set()
    old_run = 0
    empty_scrolls = 0
    new_count = 0; completed = False

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        try:
            context = browser.new_context(storage_state=str(settings.auth_file), locale='en-US', viewport={'width':1280,'height':900})
            page = context.new_page()

            def response_handler(response):
                if response.status != 200 or not any(x in response.url for x in (
                    '/graphql/query', '/api/v1/feed/', '/api/v1/media/')):
                    return
                try:
                    _network_dates(response.json(), settings.username, network_dates)
                except Exception:
                    return
            page.on('response', response_handler)
            page.goto(profile_url, wait_until='domcontentloaded', timeout=60000)
            if _access_problem(page):
                raise AuthenticationRequired('Instagram session expired/limited; operator login required')
            selector = POST_SELECTOR
            try:
                page.locator(selector).first.wait_for(state='attached', timeout=20000)
            except Exception as exc:
                raise CollectionIncomplete('No post grid loaded; cannot certify absence of new posts') from exc

            for scroll in range(1, max_scrolls+1):
                if _access_problem(page):
                    raise AuthenticationRequired('Instagram requested login/challenge/limited access')
                raw_links = _grid_links(page, selector)
                visible = []
                for item in raw_links:
                    try:
                        href = item['href'] if isinstance(item, dict) else item
                        url = canonical_url(href)
                        pinned = bool(item.get('pinned')) if isinstance(item, dict) else False
                        if not any(existing[0] == url for existing in visible):
                            visible.append((url, pinned))
                    except ValueError:
                        continue
                fresh = [(url, pinned) for url, pinned in visible if url not in found]
                found.update(url for url, _ in fresh)

                for url, pinned in fresh:
                    if url in known_dates:
                        stamp = known_dates[url]
                    else:
                        code = shortcode(url)
                        previous_y = page.evaluate('window.scrollY')
                        try:
                            card = page.locator(f'a[href*="/{code}"]').first
                            card.click(timeout=12000)
                            page.wait_for_timeout(900)
                            if _access_problem(page):
                                raise AuthenticationRequired('Instagram blocked access during scan')
                            if code not in urlparse(page.url).path:
                                raise CollectionIncomplete(f'Wrong post opened for {code}')
                            stamp = _date_from_page(page, code, network_dates)
                            if stamp is None:
                                raise CollectionIncomplete(f'Missing date for {code}')
                            if stamp > cutoff:
                                caption = _caption_from_page(page, settings.username)
                                if not caption:
                                    if on_skip is None:
                                        raise CollectionIncomplete(f'Missing/unexpanded caption for new post {code}')
                                    on_skip(url, 'empty_caption', stamp)
                                else:
                                    on_new(url, caption, stamp)
                                    new_count += 1
                                    print(f'[INSTAGRAM NEW] {code} {stamp.isoformat()}')
                                known_dates[url] = stamp
                        finally:
                            _restore_profile(page, profile_url, previous_y)
                        page.wait_for_timeout(500)
                    if stamp is None:
                        continue  # Known without a date (old skipped entry): no evidence either way.
                    is_old = stamp <= cutoff
                    if is_old:
                        old_seen.add(url)
                    if pinned:
                        print(f'[INSTAGRAM BOUNDARY] pinned post does not move the boundary: {url}')
                        continue
                    old_run, accepted = _date_boundary(old_run, is_old, threshold)
                    if accepted:
                        completed = True
                        print(f'[INSTAGRAM BOUNDARY] accepted: {old_run} consecutive posts not newer than {cutoff.isoformat()}')
                        break

                if completed:
                    break

                print(
                    f'[INSTAGRAM SCROLL] {scroll}: seen={len(found)} old={len(old_seen)} '
                    f'consecutive_old={old_run} new={new_count}'
                )
                if len(fresh) == 0:
                    empty_scrolls += 1
                else:
                    empty_scrolls = 0
                if empty_scrolls >= settings.boundary_quiet_scrolls:
                    if _boundary_at_profile_exhaustion(old_run):
                        completed = True
                        print(f'[INSTAGRAM BOUNDARY] accepted at profile end: consecutive_old={old_run}')
                        break
                    raise CollectionIncomplete(
                        'Instagram scroll stalled before reaching posts older than the cutoff; '
                        f'consecutive_old={old_run}; no output published'
                    )
                page.mouse.wheel(0, 1100)
                page.wait_for_timeout(1500)
            if not completed:
                raise CollectionIncomplete('Reached max_scrolls before confirmed old boundary; increase --max-scrolls')
        finally:
            browser.close()
    return {'new_found': new_count, 'known_old_seen': len(old_seen), 'links_seen': len(found),
            'boundary_reached': completed, 'cutoff': cutoff.isoformat()}
