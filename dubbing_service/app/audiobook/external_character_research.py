"""Keyless public-work discovery; untrusted wiki claims require local book proof.

No project name, book prose, role description, or private aliases leave Studio.
Registered title/author and one canonical character term are the only queries.
This provider never writes a project, acquires audio, or invokes a model.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dubbing_service.app.models.character import CharacterFact, StoryScope
from dubbing_service.app.models.project import Project, Segment

from .character_research import ExternalResearchUnavailable, _extract, _statements

VERSION = "mediawiki-book-validated-v1"
REGISTRY_PATH = Path(__file__).with_name("public_character_sources.json")
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
ALLOWED_APIS = frozenset({WIKIPEDIA_API, "https://supersupportive.miraheze.org/w/api.php"})
USER_AGENT = "AI-Audio-Studio-CharacterResearch/1.0 (bounded public metadata; https://github.com/tadashijlu/ai-audio-studio)"


def _normal(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(value))
    return " ".join(value.replace("’", "'").replace("‘", "'").split()).casefold()


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_public_work(metadata: Mapping, *, registry_path: Path = REGISTRY_PATH) -> dict | None:
    """Exact verified public title AND author. Unknown/private works stay offline."""
    title, author = metadata.get("source_title"), metadata.get("source_author")
    # An owner-verified per-project public binding can preserve a private/custom
    # display title. It still must resolve exactly inside the public registry.
    binding = metadata.get("external_character_research_public_work")
    if isinstance(binding, dict) and isinstance(binding.get("title"), str):
        if _normal(str(binding.get("author", ""))) != _normal(str(author or "")):
            return None
        title = binding["title"]
    if not isinstance(title, str) or not isinstance(author, str):
        return None
    # A documented chapter-range edition suffix does not change public identity.
    title = re.sub(r"\s+-\s+Chapters?\s+\d+\s*[-–]\s*\d+\s*$", "", title, flags=re.IGNORECASE)
    data = json.loads(registry_path.read_text("utf-8"))
    if data.get("schema") != "public-character-sources-v1":
        raise ValueError("Unknown public source registry schema")
    matches = [row for row in data.get("works", [])
               if _normal(title) in {_normal(v) for v in [row["title"], *row.get("title_aliases", [])]}
               and _normal(author) in {_normal(v) for v in row["authors"]}]
    if len(matches) != 1:
        return None
    work = dict(matches[0])
    if isinstance(binding, dict) and binding.get("identity_evidence") != work.get("registry_evidence"):
        return None
    if work.get("wiki_api") and work["wiki_api"] not in ALLOWED_APIS:
        raise ValueError("Public source registry endpoint is not reviewed")
    return work


_SMALL = dict(zip(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"], range(1, 20)))
_TENS = dict(zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10)))


def published_chapter(title: str) -> int | None:
    """Do not confuse Studio index 0 with published chapter 1, or 256 with 1."""
    endpoint = r"(?:[1-9]\d{0,3}|" + "|".join([*_SMALL, *_TENS]) + r")\b"
    heading = title.split(":", 1)[0]
    if re.search(r"\b(?:chapter\s+)?\d+\s*(?:[-–/]|\b(?:and|to)\b)\s*(?:chapter\s+)?" + endpoint, heading, re.IGNORECASE):
        return None
    prefix = re.split(r"\s[-–—]\s", title.split(":", 1)[0])[0].strip()
    prefix = re.sub(r"^prelude\s+and\s+chapter\s+", "Chapter ", prefix, flags=re.IGNORECASE)
    number = re.match(r"^(?:chapter\s+)?([1-9]\d{0,3})\.?$", prefix, re.IGNORECASE)
    if number:
        return int(number[1])
    words = re.sub(r"^chapter\s+", "", prefix.casefold()).replace("-", " ").split()
    units = "(?:" + "|".join(list(_SMALL)[:9]) + ")"
    small = "(?:" + "|".join(_SMALL) + ")"
    tens = "(?:" + "|".join(_TENS) + ")(?: " + units + ")?"
    tail = "(?:" + small + "|" + tens + ")"
    if not re.fullmatch("(?:" + tail + "|" + units + " hundred(?: (?:and )?" + tail + ")?)", " ".join(words)):
        return None
    total = part = 0
    for word in words:
        if word in _SMALL:
            part += _SMALL[word]
        elif word in _TENS:
            part += _TENS[word]
        elif word == "hundred" and 0 < part < 10:
            part *= 100
        elif word == "thousand" and part > 0:
            total += part * 1000
            part = 0
        elif word == "and":
            continue
        else:
            return None
    value = total + part
    return value if 0 < value <= 9999 else None


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ExternalResearchUnavailable("Redirect outside reviewed API")


@dataclass(frozen=True)
class ResearchLimits:
    max_requests: int = 12
    max_response_bytes: int = 350_000
    timeout_seconds: float = 6.0
    retries: int = 1
    max_citations: int = 50
    max_characters: int = 8

    def __post_init__(self):
        for name, low, high in (("max_requests", 1, 20), ("max_response_bytes", 1024, 1_000_000),
                                ("timeout_seconds", .1, 10), ("retries", 0, 1),
                                ("max_citations", 1, 100), ("max_characters", 1, 20)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
            if name != "timeout_seconds" and not isinstance(value, int):
                raise ValueError(f"{name} must be integral")


def fetch_json(url: str, params: Mapping, limits: ResearchLimits) -> dict:
    if url not in ALLOWED_APIS:
        raise ExternalResearchUnavailable("Endpoint outside reviewed API allowlist")
    request = Request(url + "?" + urlencode(params), headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with build_opener(_NoRedirects()).open(request, timeout=limits.timeout_seconds) as response:
        raw = response.read(limits.max_response_bytes + 1)
    if len(raw) > limits.max_response_bytes:
        raise ExternalResearchUnavailable("Public API response exceeded byte limit")
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get("error"):
        raise ExternalResearchUnavailable("Public API did not return usable JSON")
    return result


def citation_records(wikitext: str, template: str, *, limit: int = 50, chapters: set[int] | None = None) -> list[dict]:
    """Only flat, complete quote + numeric chapter records; never expand templates."""
    pattern = re.compile(r"\{\{\s*" + re.escape(template) + r"\s*\|([^{}]*?)\|\s*([1-9]\d{0,3})\s*\}\}", re.IGNORECASE)
    records = []
    for match in pattern.finditer(wikitext):
        if chapters is not None and int(match[2]) not in chapters:
            continue
        quote = match[1].strip().strip('"').strip()
        if "|" in quote or len(quote) > 1200 or len(quote) < 12:
            continue
        records.append({"quote": quote, "chapter": int(match[2]), "offset": match.start()})
        if len(records) >= limit:
            break
    return records


class ResearchDeferred(ExternalResearchUnavailable):
    research_deferred = True


class PublicWorkUnmatched(ExternalResearchUnavailable):
    pass


class UnmatchedProvider:
    provider_id = "public-work-unmatched"
    version = VERSION
    managed_extractor_prefix = VERSION
    cache_ttl_seconds = 3600
    failure_ttl_seconds = 300

    def __init__(self):
        self.last_report = {"status": "unmatched", "reason": "No exact registered public title/author; no outbound query", "network_requests": 0}

    def research(self, canonical_name, **kwargs):
        raise PublicWorkUnmatched("Public work unmatched")


class MediaWikiProvider:
    managed_extractor_prefix = VERSION
    cache_ttl_seconds = 86400
    failure_ttl_seconds = 300

    def __init__(self, work: Mapping, project: Project, segments: Sequence[Segment], *,
                 fetcher: Callable = fetch_json, limits: ResearchLimits | None = None, online: bool = False):
        self.work = dict(work)
        self.project = project
        self.segments = tuple(s for s in segments if s.lineage is None or s.lineage.retired_at is None)
        self.fetcher = fetcher
        self.limits = limits or ResearchLimits()
        self.online = online
        self.provider_id = "mediawiki:" + self.work["id"]
        # Bind cache to edition, author, mapping and canonical text without transmitting it.
        self.version = VERSION + ":" + _hash([self.work,
            [(c.id, c.index, c.title) for c in project.chapters],
            [(s.id, s.text, s.source_anchor.model_dump(mode="json") if s.source_anchor else None) for s in self.segments]])[:24]
        self.requests = 0
        self.characters = 0
        self._memo = {}
        self.last_report = {}

    def _get(self, endpoint: str, params: Mapping) -> dict:
        if not self.online:
            raise ResearchDeferred("Network research deferred to explicit metadata refresh")
        if endpoint not in ALLOWED_APIS:
            raise ExternalResearchUnavailable("Endpoint outside reviewed API allowlist")
        key = _hash([endpoint, params])
        if key in self._memo:
            return self._memo[key]
        for attempt in range(self.limits.retries + 1):
            if self.requests >= self.limits.max_requests:
                raise ExternalResearchUnavailable("Public research request budget exhausted")
            self.requests += 1
            try:
                result = self.fetcher(endpoint, dict(params), self.limits)
                self._memo[key] = result
                return result
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.limits.retries:
                    raise ExternalResearchUnavailable("Public API HTTP failure") from None
            except (URLError, TimeoutError, OSError):
                if attempt == self.limits.retries:
                    raise ExternalResearchUnavailable("Public API temporarily unavailable") from None
            time.sleep(.1)
        raise ExternalResearchUnavailable("Public API unavailable")

    def _search(self, endpoint: str, query: str) -> list[dict]:
        data = self._get(endpoint, {"action": "query", "list": "search", "srsearch": query,
                                  "srlimit": 5, "format": "json", "formatversion": 2, "maxlag": 5})
        return data.get("query", {}).get("search", [])

    def _revision(self, endpoint: str, pageid: int) -> dict:
        data = self._get(endpoint, {"action": "query", "pageids": pageid, "prop": "info|revisions", "inprop": "url",
                                  "rvprop": "ids|timestamp|content", "rvslots": "main", "format": "json", "formatversion": 2})
        pages = data.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        valid = [p for p in pages if p.get("pageid") == pageid and "missing" not in p]
        if len(valid) != 1:
            raise ExternalResearchUnavailable("Public page identity unresolved")
        return valid[0]

    def _chapter_urls(self, endpoint: str) -> dict[int, str]:
        data = self._get(endpoint, {"action": "query", "titles": self.work["chapter_template"], "prop": "revisions",
            "rvprop": "ids|content", "rvslots": "main", "format": "json", "formatversion": 2})
        pages = data.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        text = self._text(pages[0]) if pages else ""
        self._chapter_template_revision = ((pages[0].get("revisions") or [{}])[0].get("revid")
                                           if pages else None)
        result = {}
        for number, url in re.findall(r"\|\s*([1-9]\d{0,3})\s*=\s*(https://[^\s|<]+)", text):
            parsed = urlsplit(url)
            if parsed.netloc == self.work["chapter_source_host"] and parsed.path.startswith(self.work["chapter_source_path"]) and not parsed.query and not parsed.fragment:
                result[int(number)] = url
        return result

    @staticmethod
    def _text(page: dict) -> str:
        revisions = page.get("revisions") or []
        if not revisions:
            return ""
        slot = revisions[0].get("slots", {}).get("main", {})
        return slot.get("content", slot.get("*", ""))

    def _corroborate(self, record: Mapping, names: Sequence[str], chapter):
        quote = record["quote"].replace("’", "'").replace("‘", "'")
        if re.search(r"\b(?:if|unless|might|could|would|not|never|imagined|claimed|rumored|hypothetical)\b", quote, re.IGNORECASE):
            return
        external_traits = {field: value for tail, _ in _statements(quote, names)
                           for field, value in _extract(tail)}
        needle = _normal(quote).strip('"')
        for segment in self.segments:
            if segment.chapter_id != chapter.id or needle not in _normal(segment.text):
                continue
            # The local full sentence decides identity/context. A shortened wiki
            # quote cannot turn dialogue, another person, or a conditional into
            # a narrator assertion. Page titles never become unproven aliases.
            local = segment.text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            masked = re.sub(r'"[^"]*"', lambda m: " " * len(m[0]), local)
            for tail, excerpt in _statements(local, names):
                start = masked.find(excerpt)
                if start < 0:
                    continue
                prefix = re.split(r"[.!?;\n]", masked[:start])[-1]
                if not re.fullmatch(r"\s*(?:(?:But|Yet|And|Then)\s+|After all,\s+|I said before that\s+)?", prefix, re.IGNORECASE):
                    continue
                if re.search(r"\b(?:if|unless|might|could|would|not|never|imagined|claimed|rumored|hypothetical)\b", tail, re.IGNORECASE):
                    continue
                for field, value in _extract(tail):
                    if field not in {"scene_delivery", "portrayal_changes"} and external_traits.get(field) == value:
                        yield field, value, excerpt, segment

    def research(self, canonical_name: str, *, aliases: Sequence[str], source_title: str | None,
                 max_chapter: int | None) -> Sequence[CharacterFact]:
        self.last_report = {"status": "researching", "work_id": self.work["id"], "network_requests": 0, "reviews": [],
                            "reference_sources": self.work.get("research_sources", [])}
        if not isinstance(canonical_name, str) or not re.fullmatch(r"[\w][\w .'-]{0,79}", canonical_name):
            raise PublicWorkUnmatched("Character query is not a bounded public term")
        if not any(re.search(r"(?<!\w)" + re.escape(canonical_name) + r"(?!\w)", s.text, re.IGNORECASE) for s in self.segments):
            self.last_report.update(status="unmatched", reason="Character name not present in current canonical book text")
            raise PublicWorkUnmatched("Character not corroborated locally")
        if self.characters >= self.limits.max_characters:
            self.last_report.update(status="held", reason="Bounded character budget exhausted")
            raise ExternalResearchUnavailable("Character budget exhausted")
        self.characters += 1
        reviews = self.last_report["reviews"]
        endpoint = self.work.get("wiki_api")
        page = None
        if endpoint:
            try:
                hits = self._search(endpoint, canonical_name)
                # Exact name or exact first+surname expansion only; multiple matches abstain.
                exact = [h for h in hits if _normal(h.get("title", "")) == _normal(canonical_name)]
                if not exact:
                    exact = [h for h in hits if _normal(h.get("title", "")).startswith(_normal(canonical_name) + " ")
                             and len(h.get("title", "").split()) <= 3]
                if len(exact) == 1 and isinstance(exact[0].get("pageid"), int):
                    page = self._revision(endpoint, exact[0]["pageid"])
                    # Resolve only one same-wiki wikitext redirect, never arbitrary HTTP URLs.
                    redirect = re.fullmatch(r"\s*#redirect\s*\[\[([^\]|#]+)\]\]\s*", self._text(page), re.IGNORECASE)
                    if redirect:
                        target = redirect[1].replace("_", " ")
                        if not re.fullmatch(r"[\w][\w .'-]{0,79}", target):
                            raise ExternalResearchUnavailable("Unbounded public redirect target")
                        matches = [h for h in self._search(endpoint, target) if _normal(h.get("title", "")) == _normal(target)]
                        page = self._revision(endpoint, matches[0]["pageid"]) if len(matches) == 1 else None
            except ExternalResearchUnavailable:
                reviews.append({"reason": "Dedicated wiki unavailable within request limits"})
        if page is None:
            # Only registered public identity and the canonical term leave the app.
            hits = self._search(WIKIPEDIA_API, f'{self.work["title"]} {self.work["authors"][0]} {canonical_name}')
            self.last_report.update(status="review_only" if hits else "unmatched", provider="wikipedia_fallback",
                network_requests=self.requests, matches=[{"title": h.get("title"), "pageid": h.get("pageid")} for h in hits[:5]])
            reviews.append({"reason": "Wikipedia discovery has no verified edition/chapter-scoped quote; no casting facts admitted"})
            if not hits:
                raise PublicWorkUnmatched("No matching public character source")
            return []
        text = self._text(page)
        revision = (page.get("revisions") or [{}])[0]
        revid = revision.get("revid")
        if not isinstance(revid, int) or not isinstance(page.get("pageid"), int):
            raise ExternalResearchUnavailable("Public page revision is missing")
        urls = self._chapter_urls(endpoint)
        mapped = {}
        for chapter in self.project.chapters:
            number = published_chapter(chapter.title)
            if number is not None and (max_chapter is None or chapter.index <= max_chapter):
                mapped.setdefault(number, []).append(chapter)
        records = citation_records(text, self.work["citation_template"], limit=self.limits.max_citations, chapters=set(mapped))
        names = [canonical_name]  # page titles/private aliases never establish character identity
        facts = []
        host = urlsplit(endpoint).netloc
        revision_url = f'https://{host}/w/index.php?oldid={revid}'
        for record in records:
            chapters = mapped.get(record["chapter"], [])
            if len(chapters) != 1 or record["chapter"] not in urls:
                reviews.append({"chapter": record["chapter"], "reason": "Citation outside unambiguous loaded chapter mapping or lacks a verified chapter URL"})
                continue
            chapter = chapters[0]
            matches = list(self._corroborate(record, names, chapter))
            if not matches:
                reviews.append({"chapter": record["chapter"], "reason": "Quote or explicit named trait not corroborated in the supplied edition"})
                continue
            for field, value, excerpt, segment in matches:
                anchor = segment.source_anchor
                local_provenance = (f";local-source-revision:{anchor.source_revision_id};book-span:{anchor.start_offset}-{anchor.end_offset}"
                                    if anchor else ";local-source-revision:unanchored")
                identity = [self.version, self.work["id"], page["pageid"], revid, record["offset"], field, value, segment.id]
                facts.append(CharacterFact(id="external-" + _hash(identity)[:24], field=field, value=value,
                    level="SUPPORTED", confidence=.95, source_kind="reference", source=revision_url,
                    source_revision_id=f'mediawiki:{host}:{page["pageid"]}:{revid}', chapter_id=chapter.id,
                    location=f'wiki-offset:{record["offset"]};published-chapter:{record["chapter"]};segment:{segment.id};chapter-url:{urls[record["chapter"]]}{local_provenance}',
                    excerpt=excerpt[:400], scope=StoryScope(chapter_start=chapter.index, chapter_end=chapter.index if field in {"age", "age_range", "apparent_age"} else None),
                    available_after_chapter=chapter.index, spoiler_safe=False, extractor=self.version))
        reviews.append({"reason": "Unscoped biography/infobox claims were not promoted to casting facts"})
        self.last_report.update(status="ok" if facts else "review_only", provider="dedicated_mediawiki",
            network_requests=self.requests, page_id=page["pageid"], page_title=page["title"], revision_id=revid,
            revision_url=revision_url, wiki_license=self.work.get("wiki_license"),
            chapter_template_revision_id=getattr(self, "_chapter_template_revision", None),
            citation_records=len(records), accepted_facts=len(facts))
        return list({fact.id: fact for fact in facts}.values())


ELF_CHAPTER_ONE = "https://www.royalroad.com/fiction/104434/the-elf-who-would-become-a-dragon-story-complete/chapter/2034542/prelude-and-chapter-1-the-frog-and-the-toad"
ALLOWED_OFFICIAL_CHAPTERS = frozenset({ELF_CHAPTER_ONE})


class _ChapterHTML(HTMLParser):
    """Read only the chapter body, not comments, glossary or navigation."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.bodies = 0
        self.skip = 0
        self.parts = []
        self.heading = []
        self.in_heading = False
        self.heading_done = False
        self.canonical = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")
        if tag == "h1" and not self.heading_done:
            self.in_heading = True
        if tag == "div":
            if self.depth:
                self.depth += 1
            elif "chapter-content" in attrs.get("class", "").split():
                self.depth = 1
                self.bodies += 1
        if self.depth:
            if tag in {"script", "style"}:
                self.skip += 1
            elif tag in {"p", "br"}:
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "h1" and self.in_heading:
            self.in_heading = False
            self.heading_done = True
        if self.depth:
            if tag in {"script", "style"} and self.skip:
                self.skip -= 1
            elif tag == "p":
                self.parts.append("\n")
            if tag == "div":
                self.depth -= 1

    def handle_data(self, data):
        if self.in_heading:
            self.heading.append(data)
        if self.depth and not self.skip:
            self.parts.append(data)


def fetch_official_html(url: str, limits: ResearchLimits) -> str:
    if url not in ALLOWED_OFFICIAL_CHAPTERS:
        raise ExternalResearchUnavailable("Official chapter URL outside reviewed allowlist")
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with build_opener(_NoRedirects()).open(request, timeout=limits.timeout_seconds) as response:
        raw = response.read(limits.max_response_bytes + 1)
    if len(raw) > limits.max_response_bytes:
        raise ExternalResearchUnavailable("Official chapter exceeded byte limit")
    return raw.decode("utf-8")


class OfficialChapterProvider(MediaWikiProvider):
    """A fixed author-verified chapter source; no link traversal or crawler."""
    def __init__(self, *args, html_fetcher=fetch_official_html, **kwargs):
        super().__init__(*args, **kwargs)
        self.html_fetcher = html_fetcher
        self.provider_id = "official-chapter:" + self.work["id"]

    def _chapter(self, url):
        if not self.online:
            raise ResearchDeferred("Network research deferred to explicit metadata refresh")
        if url not in ALLOWED_OFFICIAL_CHAPTERS:
            raise ExternalResearchUnavailable("Official chapter URL outside reviewed allowlist")
        if url in self._memo:
            return self._memo[url]
        for attempt in range(self.limits.retries + 1):
            if self.requests >= self.limits.max_requests:
                raise ExternalResearchUnavailable("Public research request budget exhausted")
            self.requests += 1
            try:
                text = self.html_fetcher(url, self.limits)
                if not isinstance(text, str) or len(text.encode("utf-8")) > self.limits.max_response_bytes:
                    raise ExternalResearchUnavailable("Official chapter exceeded byte limit")
                self._memo[url] = text
                return text
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.limits.retries:
                    raise ExternalResearchUnavailable("Official chapter HTTP failure") from None
            except (URLError, TimeoutError, OSError):
                if attempt == self.limits.retries:
                    raise ExternalResearchUnavailable("Official chapter temporarily unavailable") from None
            time.sleep(.1)
        raise ExternalResearchUnavailable("Official chapter unavailable")

    def research(self, canonical_name, *, aliases, source_title, max_chapter):
        self.last_report = {"status": "researching", "work_id": self.work["id"], "provider": "official_chapter",
                            "network_requests": self.requests, "reviews": [],
                            "reference_sources": self.work.get("research_sources", [])}
        if not isinstance(canonical_name, str) or not re.fullmatch(r"[\w][\w .'-]{0,79}", canonical_name):
            raise PublicWorkUnmatched("Character query is not a bounded public term")
        if not any(re.search(r"(?<!\w)" + re.escape(canonical_name) + r"(?!\w)", s.text, re.IGNORECASE) for s in self.segments):
            raise PublicWorkUnmatched("Character not corroborated locally")
        if self.characters >= self.limits.max_characters:
            raise ExternalResearchUnavailable("Character budget exhausted")
        self.characters += 1
        facts, sources = [], []
        for seed in self.work.get("official_chapters", []):
            mapped = [chapter for chapter in self.project.chapters
                      if published_chapter(chapter.title) == seed["number"]
                      and (max_chapter is None or chapter.index <= max_chapter)]
            if len(mapped) != 1:
                continue
            chapter = mapped[0]
            text = self._chapter(seed["url"])
            parser = _ChapterHTML()
            parser.feed(text)
            if (parser.bodies != 1 or parser.canonical != seed["url"]
                    or published_chapter(" ".join(parser.heading)) != seed["number"]):
                raise ExternalResearchUnavailable("Official page identity or chapter body unresolved")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            sources.append({"url": seed["url"], "published_chapter": seed["number"], "content_sha256": content_hash})
            paragraphs = [" ".join(part.split()) for part in "".join(parser.parts).split("\n") if part.strip()]
            eligible = [(index, paragraph) for index, paragraph in enumerate(paragraphs)
                        if 12 <= len(paragraph) <= 1200 and any(_extract(tail) for tail, _ in _statements(paragraph, [canonical_name]))]
            for index, quote in eligible[:self.limits.max_citations]:
                for field, value, excerpt, segment in self._corroborate({"quote": quote}, [canonical_name], chapter):
                    anchor = segment.source_anchor
                    local_provenance = (f";local-source-revision:{anchor.source_revision_id};book-span:{anchor.start_offset}-{anchor.end_offset}"
                                        if anchor else ";local-source-revision:unanchored")
                    identity = [self.version, seed["url"], content_hash, index, field, value, segment.id]
                    facts.append(CharacterFact(id="external-" + _hash(identity)[:24], field=field, value=value,
                        level="SUPPORTED", confidence=.95, source_kind="official", source=seed["url"],
                        source_revision_id="official-sha256:" + content_hash, chapter_id=chapter.id,
                        location=f'paragraph:{index};published-chapter:{seed["number"]};segment:{segment.id}{local_provenance}',
                        excerpt=excerpt[:400], scope=StoryScope(chapter_start=chapter.index, chapter_end=chapter.index if field in {"age", "age_range", "apparent_age"} else None),
                        available_after_chapter=chapter.index, spoiler_safe=False, extractor=self.version))
        self.last_report.update(status="ok" if facts else "review_only", network_requests=self.requests,
                                verified_sources=sources, accepted_facts=len(facts))
        return list({fact.id: fact for fact in facts}.values())


def provider_for_project(project: Project, segments: Sequence[Segment], **kwargs):
    work = resolve_public_work(project.metadata or {}, registry_path=kwargs.pop("registry_path", REGISTRY_PATH))
    if not work:
        return UnmatchedProvider()
    selected = OfficialChapterProvider if work.get("official_chapters") and not work.get("wiki_api") else MediaWikiProvider
    return selected(work, project, segments, **kwargs)
