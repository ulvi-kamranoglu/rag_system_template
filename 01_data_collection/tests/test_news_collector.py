"""Offline News tests: extraction, sitemap discovery, fixed baseline, confirmed deletions."""
import json

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from wp_tracker.news import collector as http, parsing, schema, service
from wp_tracker.news.config import NewsSettings, SITEMAP_URL, load_settings
from wp_tracker.records import dataset

BODY = "Universitetdə keçirilən tədbir haqqında ətraflı məlumat verilir."


def url(n):
    return f"https://news.unec.edu.az/xeber/1-elm/{n}-article"


def soup(html):
    return BeautifulSoup(html, "html.parser")


def article(date="2026-11-02", title="Title", content="Body"):
    return soup(f'<article><time datetime="{date}"></time><h1>{title}</h1>'
                f'<section class="article-content"><p>{content}</p></section></article>')


# --- extraction (selectors and date rules are unchanged) ---------------------

@pytest.mark.parametrize("href", [url(1) + "/?utm=x#top", "/xeber/1-elm/1-article", "http://news.unec.edu.az/xeber/1-elm/1-article"])
def test_canonical_urls(href):
    assert parsing.canonical_url(href) == url(1)


@pytest.mark.parametrize("href", ["https://example.com/xeber/1/2", "/xeber", "/xeber/foo/bar", "/xeber/1/2/3",
                                  "javascript:alert(1)", "https://user@news.unec.edu.az/xeber/1/2",
                                  "https://news.unec.edu.az:9999/xeber/1/2", "https://[broken/xeber/1/2",
                                  "https://news.unec.edu.az/xeber/1/2 bad", None])
def test_invalid_urls(href):
    assert parsing.canonical_url(href) is None


def test_original_strong_extraction_and_cleaning():
    html = soup('<article><section class="article-content"><strong>Başlıq</strong><strong>Başlıq</strong>'
                '<p>İlk <a href="/">mətn</a>.</p><p>İkinci<br>sətir</p>'
                '<script>bad</script><div class="itemSocialSharing">share</div>'
                '<div class="itemRelated"><strong>unrelated</strong></div></section></article>')
    title, content = parsing.article_title_and_content(html)
    assert title == "Başlıq"
    assert content == "İlk mətn.\nİkinci\nsətir"


def test_h1_fallback_and_no_body_date():
    html = article(content="On 2026-01-01 something happened")
    assert parsing.parse_article(html) == ("2026-11-02", "Title", "On 2026-01-01 something happened")
    html.time.decompose()
    assert parsing.article_date(html) is None


@pytest.mark.parametrize("raw,expected", [("30 Oktyabr 2026", "2026-10-30"), ("2 noy. 2026", "2026-11-02"),
                                           ("30.10.2026", "2026-10-30"), ("2026/10/30", "2026-10-30"),
                                           ("31 fevral 2026", None), ("invalid", None)])
def test_dates(raw, expected):
    assert parsing.parse_date(raw) == expected


# --- identity -----------------------------------------------------------------

def test_id_and_metadata_come_from_the_url_number():
    row = schema.article_row(url(15700), "Title", BODY, "2026-10-30")
    assert row["id"] == "news:15700" and json.loads(row["metadata"]) == {"news_id": "15700"}
    assert row["date"] == pd.Timestamp("2026-10-30")


def legacy_seed(path, numbers, **override):
    rows = []
    for index, n in enumerate(numbers, 1):
        row = schema.article_row(url(n), f"Title {n}", f"{BODY} {n}", "2026-10-01")
        rows.append({**row, "id": index, "metadata": "{}", **override})
    dataset(rows).to_parquet(path)
    return path


@pytest.mark.parametrize("override,match", [
    ({"id": "news:999"}, "conflicts with URL"),
    ({"metadata": '{"news_id":"7"}'}, "conflicts with URL"),
    ({"url": "https://news.unec.edu.az/xeber/1-elm/no-number"}, "numbered news article"),
    ({"source_type": "post"}, "not source_type=news"),
])
def test_invalid_seed_rows_are_rejected(override, match, tmp_path):
    row = {**schema.article_row(url(5), "T", BODY, "2026-10-01"), **override}
    with pytest.raises(ValueError, match=match):
        schema.seed_row(row)


# --- HTTP doubles -------------------------------------------------------------

class Response:
    def __init__(self, status=200, text="", content_type="text/html"):
        self.status_code, self.text, self.content = status, text, text.encode("utf-8")
        self.headers, self.encoding = {"Content-Type": content_type}, "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def sitemap_xml(numbers, extra=""):
    items = "".join(f"<url><loc><![CDATA[{url(n)}]]></loc><lastmod>2026-11-01T00:00:00+00:00</lastmod></url>"
                    for n in numbers)
    return ('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            '<url><loc><![CDATA[https://news.unec.edu.az/xeber]]></loc></url>'
            '<url><loc><![CDATA[https://news.unec.edu.az/xeber/68-tehsil-nazirliyi]]></loc></url>'
            f'{items}{extra}</urlset>')


class FakeSite:
    def __init__(self, listed, *, pages=None):
        self.listed, self.pages, self.calls = list(listed), dict(pages or {}), []

    def get(self, target, **kwargs):
        self.calls.append(target)
        if target == SITEMAP_URL:
            return Response(text=sitemap_xml(self.listed), content_type="text/xml")
        value = self.pages.get(target)
        if value is None:
            n = int(target.rsplit("/", 1)[1].split("-")[0])
            if n not in self.listed:
                return Response(404)
            value = str(article("2026-11-05", f"Title {n}", f"{BODY} {n}"))
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, Response) else Response(text=value)

    def article_calls(self):
        return [c for c in self.calls if c != SITEMAP_URL]


@pytest.fixture
def settings(tmp_path):
    return NewsSettings(tmp_path / "01_data_collection", delay_seconds=0)


@pytest.fixture
def initialized(settings):
    settings.seed_file.parent.mkdir(parents=True)
    legacy_seed(settings.seed_file, range(1, 41))
    service.bootstrap(settings, run=True)
    return settings


def outputs(report):
    folder = report["run_dir"]
    return (pd.read_parquet(f"{folder}/chatbot_dataset_new.parquet"),
            pd.read_parquet(f"{folder}/chatbot_dataset_deleted.parquet"))


# --- sitemap ------------------------------------------------------------------

def test_sitemap_lists_only_numbered_articles():
    assert http.sitemap_articles(FakeSite([3, 1, 2])) == {"1": url(1), "2": url(2), "3": url(3)}


def test_sitemap_index_is_followed():
    child = "https://news.unec.edu.az/sitemap-1.xml"
    index = ('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
             f'<sitemap><loc>{child}</loc></sitemap></sitemapindex>')
    site = FakeSite([], pages={})
    site.get = lambda target, **kw: Response(text=index if target == SITEMAP_URL else sitemap_xml([9]),
                                             content_type="text/xml")
    assert http.sitemap_articles(site) == {"9": url(9)}


def test_duplicate_number_keeps_the_known_url():
    other = "https://news.unec.edu.az/xeber/2-diger/5-article"
    site = FakeSite([5])
    site.get = lambda target, **kw: Response(text=sitemap_xml([5], f"<url><loc>{other}</loc></url>"),
                                             content_type="text/xml")
    assert http.sitemap_articles(site, {"5": url(5)})["5"] == url(5)


@pytest.mark.parametrize("response", [Response(500), Response(text="<html>", content_type="text/html"),
                                      Response(text=sitemap_xml([]), content_type="text/xml")])
def test_unusable_sitemap_fails_closed(response):
    site = FakeSite([])
    site.get = lambda target, **kw: response
    with pytest.raises(http.CollectionIncomplete):
        http.sitemap_articles(site)


# --- bootstrap ----------------------------------------------------------------

def test_bootstrap_rebuilds_ids_from_urls_from_the_import_folder(initialized):
    baseline = pd.read_parquet(initialized.baseline_file)
    assert baseline["id"].tolist()[:2] == ["news:1", "news:2"]
    assert json.loads(baseline["metadata"][0]) == {"news_id": "1"}
    assert pd.read_parquet(initialized.snapshot_file).equals(baseline)
    with pytest.raises(FileExistsError, match="never replaced"):
        service.bootstrap(initialized, run=True)


def test_bootstrap_requires_the_seed(settings):
    with pytest.raises(FileNotFoundError, match="data/import/news.parquet"):
        service.bootstrap(settings, run=True)


def test_bootstrap_dry_run_writes_nothing(settings):
    settings.seed_file.parent.mkdir(parents=True)
    legacy_seed(settings.seed_file, [1, 2])
    assert service.bootstrap(settings)["records"] == 2
    assert not settings.state_dir.exists()


# --- scans --------------------------------------------------------------------

def test_new_articles_are_baseline_relative_and_only_new_pages_are_opened(initialized):
    site = FakeSite(list(range(1, 41)) + [41, 42])
    report = service.scan(initialized, run=True, client=site)
    new, deleted = outputs(report)
    assert new["id"].tolist() == ["news:41", "news:42"] and deleted.empty
    assert site.article_calls() == [url(41), url(42)]
    assert json.loads(new["metadata"][0]) == {"news_id": "41"}
    # Unchanged site: WordPress semantics repeat baseline-relative NEW, no article is reopened.
    again = FakeSite(list(range(1, 43)))
    new, _ = outputs(service.scan(initialized, run=True, client=again))
    assert new["id"].tolist() == ["news:41", "news:42"] and again.article_calls() == []


def test_missing_article_is_deleted_only_after_a_404(initialized):
    site = FakeSite([n for n in range(1, 41) if n != 7])
    report = service.scan(initialized, run=True, client=site)
    _, deleted = outputs(report)
    assert deleted["id"].tolist() == ["news:7"] and report["confirmed_deleted_this_scan"] == ["news:7"]
    assert "news:7" not in pd.read_parquet(initialized.snapshot_file)["id"].tolist()


@pytest.mark.parametrize("probe,unconfirmed", [
    (Response(text=str(article("2026-10-01", "Still here", BODY))), []),   # present: sitemap glitch
    (Response(503), ["news:7"]),
    (RuntimeError("timeout"), ["news:7"]),
    (Response(301), ["news:7"]),
])
def test_uncertain_absence_never_deletes(initialized, probe, unconfirmed):
    site = FakeSite([n for n in range(1, 41) if n != 7], pages={url(7): probe})
    report = service.scan(initialized, run=True, client=site)
    _, deleted = outputs(report)
    assert deleted.empty and report["unconfirmed_missing"] == unconfirmed
    assert "news:7" in pd.read_parquet(initialized.snapshot_file)["id"].tolist()


def test_shrunken_sitemap_stops_before_probing_or_writing(initialized):
    before = initialized.snapshot_file.read_bytes()
    site = FakeSite(range(1, 30))
    with pytest.raises(RuntimeError, match="Safety stop"):
        service.scan(initialized, run=True, client=site)
    assert site.article_calls() == []
    assert initialized.snapshot_file.read_bytes() == before
    assert not list(initialized.archive_dir.glob("run_*"))


def test_large_confirmed_deletions_need_explicit_permission(initialized):
    site = FakeSite(range(1, 30))
    report = service.scan(initialized, run=True, client=site, allow_large_deletions=True)
    assert report["deleted"] == 11


def test_unparsable_new_article_is_retried_next_scan(initialized):
    site = FakeSite(list(range(1, 42)), pages={url(41): Response(text="<html><body>broken</body></html>")})
    report = service.scan(initialized, run=True, client=site)
    assert report["new"] == 0 and report["unparsed_new"][0]["url"] == url(41)
    retry = FakeSite(list(range(1, 42)))
    new, _ = outputs(service.scan(initialized, run=True, client=retry))
    assert new["id"].tolist() == ["news:41"]


def test_low_value_new_article_is_not_exported_but_kept_in_snapshot(initialized):
    site = FakeSite(list(range(1, 42)), pages={url(41): str(article("2026-11-05", "Title", "Ok"))})
    report = service.scan(initialized, run=True, client=site)
    assert report["new"] == 0 and report["filtered_new"] == 1
    assert "news:41" in pd.read_parquet(initialized.snapshot_file)["id"].tolist()


def test_dry_run_requests_nothing(initialized):
    class NoNetwork:
        def get(self, *a, **kw):
            raise AssertionError("network used")
    assert service.scan(initialized, client=NoNetwork())["dry_run"] is True


def test_publication_retry_after_a_failed_mirror(initialized, monkeypatch):
    calls = {"n": 0}
    original = service.mirror
    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk busy")
        return original(*args, **kwargs)
    monkeypatch.setattr(service, "mirror", flaky)
    with pytest.raises(RuntimeError, match="shared publish failed"):
        service.scan(initialized, run=True, client=FakeSite(range(1, 42)))
    assert service.sync_shared(initialized, run=True)["status"] == "synced"
    assert (initialized.shared_dir / "chatbot_dataset_new.parquet").is_file()


# --- configuration --------------------------------------------------------------

def test_old_listing_settings_are_rejected(tmp_path):
    path = tmp_path / "config/settings.yaml"
    path.parent.mkdir()
    path.write_text("news:\n  boundary_old_urls: 24\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only enabled and delay_seconds"):
        load_settings(path)


def test_settings_use_the_shared_import_folder(tmp_path):
    path = tmp_path / "config/settings.yaml"
    path.parent.mkdir()
    path.write_text("news:\n  delay_seconds: 0.5\n", encoding="utf-8")
    settings = load_settings(path)
    assert settings.seed_file.name == "news.parquet" and settings.seed_file.parent.name == "import"
    assert settings.delay_seconds == 0.5 and settings.max_deleted_fraction == 0.05
