"""Synthetic metadata/edition fixtures; no actor audio, credentials or network."""
import json
from contextlib import contextmanager
from urllib.error import HTTPError, URLError

import pytest

from dubbing_service.app.audiobook import character_casting as casting
from dubbing_service.app.audiobook import character_research as research
from dubbing_service.app.audiobook import external_character_research as external
from dubbing_service.app.models.audiobook import SourceAnchor
from dubbing_service.app.models.project import Chapter, Project, Role, Segment


def fixture(text="Alice's voice was deep and clear.", chapter_number=1, index=0):
    project = Project(id="public-fixture", name="PRIVATE PROJECT LABEL",
        metadata={"source_title": "Super Supportive", "source_author": "Sleyca"},
        roles=[Role(id="alice", name="Alice")],
        chapters=[Chapter(id="chapter", index=index, title=f"Chapter {chapter_number}")])
    project.character_cast.policy.external_research = True
    segments = [Segment(id="segment", chapter_id="chapter", index=0, role_id="alice", text=text,
        source_anchor=SourceAnchor(source_revision_id="source1", chapter_id="chapter",
                                   occurrence_index=0, start_offset=0, end_offset=len(text)))]
    return project, segments


class API:
    def __init__(self, content="{{soup|Alice's voice was deep and clear.|1}}", hits=None):
        self.content = content
        self.hits = hits if hits is not None else [{"title": "Alice", "pageid": 7}]
        self.calls = []

    def __call__(self, endpoint, params, limits):
        self.calls.append((endpoint, params))
        if params.get("list") == "search":
            return {"query": {"search": self.hits}}
        if params.get("titles") == "Template:Soup/chapter":
            text = "| 1 = https://www.royalroad.com/fiction/63759/super-supportive/chapter/111/one\n| 5 = https://www.royalroad.com/fiction/63759/super-supportive/chapter/555/five\n| 256 = https://www.royalroad.com/fiction/63759/super-supportive/chapter/256256/two-hundred-fifty-six"
            return {"query": {"pages": [{"pageid": 9, "title": "Template:Soup/chapter", "revisions": [
                {"revid": 80, "slots": {"main": {"content": text}}}]}]}}
        return {"query": {"pages": [{"pageid": 7, "title": "Alice", "revisions": [
            {"revid": 42, "timestamp": "2026-10-01T00:00:00Z", "slots": {"main": {"content": self.content}}}]}]}}


def provider(project, segments, api, **kwargs):
    return external.provider_for_project(project, segments, fetcher=api, online=True, **kwargs)


@pytest.mark.parametrize("title,expected", [("Chapter 1",1),("ONE: A beginning",1),
    ("TWO HUNDRED FIFTY-SIX: A Pointy Thing",256),("CHAPTER ELEVEN: Last",11),
    ("Information",None),("Section 1",None),("0",None),("Chapter 5",5)])
def test_published_chapter_is_not_local_index(title, expected):
    assert external.published_chapter(title) == expected


@pytest.mark.parametrize("metadata", [{}, {"source_title":"Super Supportive"},
    {"source_title":"PRIVATE PROJECT LABEL","source_author":"Sleyca"},
    {"source_title":"Super Supportive","source_author":"Wrong author"}])
def test_unknown_private_or_wrong_author_work_never_queries(metadata):
    project, segments = fixture()
    project.metadata = metadata
    api = API()
    selected = provider(project, segments, api)
    assert isinstance(selected, external.UnmatchedProvider)
    with pytest.raises(external.PublicWorkUnmatched):
        selected.research("Alice", aliases=["PRIVATE ALIAS"], source_title=project.name, max_chapter=0)
    assert not api.calls


@pytest.mark.parametrize("title,author,expected", [
    ("Super Supportive - Chapters 256-260","Sleyca","super-supportive"),
    ("Pride and Prejudice","Jane Austen","pride-and-prejudice"),
    ("The Adventures of Sherlock Holmes","Arthur Conan Doyle","adventures-sherlock-holmes")])
def test_extensible_registered_public_works(title,author,expected):
    assert external.resolve_public_work({"source_title":title,"source_author":author})["id"] == expected


def test_quote_scope_provenance_privacy_and_stable_identity():
    project, segments = fixture()
    api = API()
    selected = provider(project, segments, api)
    facts = selected.research("Alice", aliases=["PRIVATE ALIAS"], source_title="PRIVATE LABEL", max_chapter=0)
    assert len(facts) == 1
    fact = facts[0]
    assert fact.field == "voice_description" and fact.value == "deep and clear"
    assert fact.source_kind == "reference" and fact.source_revision_id.endswith(":7:42")
    assert fact.chapter_id == "chapter" and fact.available_after_chapter == 0
    assert "segment:segment" in fact.location and "published-chapter:1" in fact.location
    assert "oldid=42" in fact.source and "chapter-url:https://www.royalroad.com/" in fact.location
    queries = json.dumps(api.calls)
    assert "PRIVATE" not in queries and "deep and clear" not in queries
    assert selected.research("Alice", aliases=[], source_title=None, max_chapter=0)[0].id == fact.id
    assert len(api.calls) == 3  # repeated query uses bounded per-refresh memo


@pytest.mark.parametrize("content", [
    "Alice has a low voice; no chapter citation.",
    "{{soup|Alice's voice was bright and clear.|1}}",  # supplied edition differs
    "{{soup|Alice's voice was deep and clear.|122}}",  # future/unloaded
    "{{soup|She had a deep and clear voice.|1}}",      # pronoun is not identity proof
    "{{soup|If Alice's voice was deep and clear, we would leave.|1}}",
    "{{soup|Alice's voice was not deep and clear.|1}}",
    "{{soup|Alice's voice was {{bad}} deep and clear.|1}}",
])
def test_unscoped_disputed_future_and_untrusted_claims_do_not_enter_facts(content):
    project, segments = fixture()
    selected = provider(project, segments, API(content))
    assert selected.research("Alice", aliases=[], source_title=None, max_chapter=0) == []
    assert selected.last_report["status"] == "review_only"
    assert "quote" not in json.dumps(selected.last_report)


def test_canonical_character_term_must_exist_in_current_book():
    project, segments = fixture()
    api = API()
    selected = provider(project, segments, api)
    with pytest.raises(external.PublicWorkUnmatched):
        selected.research("Private Actor Nickname",aliases=[],source_title=None,max_chapter=0)
    assert not api.calls


def test_information_chapter_and_256_are_mapped_explicitly():
    project, segments = fixture(chapter_number=256,index=1)
    project.chapters.insert(0, Chapter(id="information",index=0,title="Information"))
    selected = provider(project,segments,API("{{soup|Alice's voice was deep and clear.|256}}"))
    facts = selected.research("Alice",aliases=[],source_title=None,max_chapter=1)
    assert facts[0].available_after_chapter == 1
    assert "published-chapter:256" in facts[0].location
    assert selected.research("Alice",aliases=[],source_title=None,max_chapter=0) == []


def test_duplicate_chapter_numbers_abstain():
    project,segments = fixture()
    project.chapters.append(Chapter(id="other",index=1,title="Chapter 1"))
    selected=provider(project,segments,API())
    assert not selected.research("Alice",aliases=[],source_title=None,max_chapter=1)


def test_wikipedia_fallback_is_discovery_only_and_uses_registered_identity():
    project,segments=fixture()
    project.metadata={"source_title":"Pride and Prejudice","source_author":"Jane Austen"}
    api=API(hits=[{"title":"Alice, unrelated person","pageid":9}])
    selected=provider(project,segments,api)
    assert selected.research("Alice",aliases=["SECRET"],source_title=project.name,max_chapter=0)==[]
    assert api.calls[0][0]==external.WIKIPEDIA_API
    assert api.calls[0][1]["srsearch"]=="Pride and Prejudice Jane Austen Alice"
    assert selected.last_report["status"]=="review_only"


def test_no_search_match_is_not_success():
    project,segments=fixture()
    selected=provider(project,segments,API(hits=[]))
    with pytest.raises(external.PublicWorkUnmatched):
        selected.research("Alice",aliases=[],source_title=None,max_chapter=0)
    assert selected.last_report["status"]=="unmatched"


def test_offline_ingest_performs_zero_requests():
    project,segments=fixture()
    api=API()
    selected=external.provider_for_project(project,segments,fetcher=api)
    with pytest.raises(external.ExternalResearchUnavailable):
        selected.research("Alice",aliases=[],source_title=None,max_chapter=0)
    assert not api.calls


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1/", "https://localhost/w/api.php",
    "https://en.wikipedia.org.evil.test/w/api.php", "https://en.wikipedia.org/w/api.php?secret=x"])
def test_fixed_allowlist_rejects_arbitrary_endpoints(endpoint):
    with pytest.raises(external.ExternalResearchUnavailable):
        external.fetch_json(endpoint,{},external.ResearchLimits())


def test_redirects_are_not_followed():
    with pytest.raises(external.ExternalResearchUnavailable):
        external._NoRedirects().redirect_request(None,None,302,"",{},"http://127.0.0.1/")


def test_response_byte_limit(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,amount): return b'x'*amount
    class Opener:
        def open(self,*args,**kwargs): return Response()
    monkeypatch.setattr(external,"build_opener",lambda *args:Opener())
    with pytest.raises(external.ExternalResearchUnavailable):
        external.fetch_json(external.WIKIPEDIA_API,{},external.ResearchLimits(max_response_bytes=1024))


def test_retry_is_bounded_by_global_request_budget():
    project,segments=fixture()
    calls=[]
    def failing(*args):
        calls.append(1)
        raise URLError("SECRET must not be persisted")
    selected=provider(project,segments,failing,limits=external.ResearchLimits(max_requests=2,retries=1))
    with pytest.raises(external.ExternalResearchUnavailable):
        selected.research("Alice",aliases=[],source_title=None,max_chapter=0)
    assert len(calls)==2


def test_nonretryable_http_failure_is_sanitized():
    project,segments=fixture()
    def failing(*args): raise HTTPError("secret",403,"PRIVATE_TOKEN",{},None)
    selected=provider(project,segments,failing)
    with pytest.raises(external.ExternalResearchUnavailable) as error:
        selected._get(external.WIKIPEDIA_API,{})
    assert "PRIVATE" not in str(error.value) and selected.requests==1


@pytest.mark.parametrize("kwargs", [{"max_requests":0},{"max_response_bytes":1000001},
    {"retries":2},{"timeout_seconds":float('nan')},{"max_characters":1.5}])
def test_request_limits_are_validated(kwargs):
    with pytest.raises(ValueError):external.ResearchLimits(**kwargs)


@pytest.mark.parametrize("fail",[False,True])
def test_success_and_failure_cache_expire_without_secret_exception_text(fail):
    project,segments=fixture()
    class Cached:
        provider_id="ttl-fixture"
        version="1"
        cache_ttl_seconds=86400
        failure_ttl_seconds=300
        calls=0
        def __init__(self):
            self.last_report={"status":"held","reason":"synthetic review only"}
        def research(self,*args,**kwargs):
            self.calls+=1
            if fail:raise OSError("PRIVATE_TOKEN")
            return []
    selected=Cached()
    research.research_project(project,segments,external_provider=selected)
    research.research_project(project,segments,external_provider=selected)
    assert selected.calls==1
    entry=next(iter(project.character_cast.external_cache.values()))
    assert entry["expires_at"]>0 and "report" in entry
    assert "PRIVATE_TOKEN" not in project.character_cast.model_dump_json()
    entry["expires_at"]=0
    research.research_project(project,segments,external_provider=selected)
    assert selected.calls==2


def test_book_priority_and_quarantine_through_existing_research_contract():
    project,segments=fixture()
    selected=provider(project,segments,API("{{soup|Alice's voice was deep and clear.|1}}{{soup|Secret future.|122}}"))
    research.research_project(project,segments,external_provider=selected)
    resolved=research.resolve_fact(project.character_cast.dossiers['alice'],'voice_description',project.character_cast.policy)
    assert resolved.source_kind=='book'
    cached=next(iter(project.character_cast.external_cache.values()))
    assert cached['report']['accepted_facts']==1
    assert all(f['available_after_chapter']==0 for f in cached['facts'])
    assert 'Secret future' not in project.character_cast.model_dump_json()


def test_refresh_does_http_outside_lock_and_preserves_live_rating_update(monkeypatch):
    original,segments=fixture()
    latest=original.model_copy(deep=True)
    held=False
    @contextmanager
    def lock(project_id):
        nonlocal held
        assert not held
        held=True
        try:yield
        finally:held=False
    monkeypatch.setattr(casting,'project_meta_lock',lock)
    loads=iter([original,latest])
    monkeypatch.setattr(casting,'load_project',lambda _:next(loads))
    monkeypatch.setattr(casting,'load_segments',lambda _:segments)
    saved=[]
    monkeypatch.setattr(casting,'save_project',lambda p:saved.append(p.model_copy(deep=True)))
    class Selected:
        provider_id='outside-lock'
        version='1'
        def research(self,*args,**kwargs):
            assert not held
            latest.voice_lab.casting={'rating_during_research':5}
            return []
    result=casting.refresh_cast(original.id,external_provider=Selected())
    assert saved and result.voice_lab.casting=={'rating_during_research':5}
    assert result.roles==original.roles


def test_refresh_refuses_stale_book_without_writing(monkeypatch):
    original,segments=fixture()
    changed=original.model_copy(deep=True)
    changed.roles[0].description='Changed during HTTP research'
    loads=iter([original,changed])
    monkeypatch.setattr(casting,'load_project',lambda _:next(loads))
    monkeypatch.setattr(casting,'load_segments',lambda _:segments)
    saved=[]
    monkeypatch.setattr(casting,'save_project',lambda p:saved.append(p))
    monkeypatch.setattr(casting,'research_and_profile',lambda *args,**kwargs:None)
    with pytest.raises(ValueError,match='Research inputs changed'):
        casting.refresh_cast(original.id)
    assert not saved


def test_source_hash_and_chapter_mapping_invalidate_provider_identity():
    project,segments=fixture()
    first=provider(project,segments,API()).version
    segments[0].text+=' Edition change.'
    second=provider(project,segments,API()).version
    project.chapters[0].title='Chapter 5'
    third=provider(project,segments,API()).version
    assert len({first,second,third})==3


@pytest.mark.parametrize("local", [
    'He said, "Alice\'s voice was deep and clear."',
    "If Alice's voice was deep and clear, he would know.",
    "Bob claimed Alice's voice was deep and clear.",
])
def test_shortened_wiki_quote_cannot_remove_local_dialogue_or_conditional_context(local):
    project,segments=fixture(local)
    selected=provider(project,segments,API("{{soup|Alice's voice was deep and clear|1}}"))
    assert selected.research("Alice",aliases=[],source_title=None,max_chapter=0)==[]


def test_expanded_page_title_is_not_a_book_character_alias():
    project,segments=fixture("Alice watched. Alice Smith's voice was deep and clear.")
    api=API("{{soup|Alice Smith's voice was deep and clear.|1}}",hits=[{"title":"Alice Smith","pageid":7}])
    selected=provider(project,segments,api)
    assert selected.research("Alice",aliases=[],source_title=None,max_chapter=0)==[]


def test_offline_deferral_does_not_poison_immediate_online_refresh():
    project,segments=fixture()
    offline_api=API()
    offline=external.provider_for_project(project,segments,fetcher=offline_api,online=False)
    research.research_project(project,segments,external_provider=offline)
    assert not project.character_cast.external_cache and not offline_api.calls
    online_api=API()
    online=provider(project,segments,online_api)
    research.research_project(project,segments,external_provider=online)
    assert len(online_api.calls)==3
    assert any(f.source_kind=='reference' for f in project.character_cast.dossiers['alice'].facts)


def test_sparse_late_chapter_quotes_are_selected_before_record_cap():
    project,segments=fixture(chapter_number=256,index=1)
    content="{{soup|Alice's voice was deep and clear.|1}}"*60
    content+="{{soup|Alice's voice was deep and clear.|256}}"
    selected=provider(project,segments,API(content),limits=external.ResearchLimits(max_citations=1))
    facts=selected.research("Alice",aliases=[],source_title=None,max_chapter=1)
    assert len(facts)==1 and facts[0].available_after_chapter==1


def test_expired_removed_quote_retires_only_owned_evidence_and_preserves_history():
    project,segments=fixture()
    selected=provider(project,segments,API())
    research.research_project(project,segments,external_provider=selected)
    dossier=project.character_cast.dossiers['alice']
    previous=next(f for f in dossier.facts if f.source_kind=='reference')
    prior_revision=dossier.revision
    for entry in project.character_cast.external_cache.values():
        if 'expires_at' in entry:entry['expires_at']=0
    research.research_project(project,segments,external_provider=provider(project,segments,API('No source citations remain.')))
    assert not any(f.id==previous.id for f in dossier.facts)
    history=project.character_cast.external_cache['external-evidence-history:alice']
    assert previous.model_dump(mode='json') in history['retired_facts']
    assert dossier.revision!=prior_revision
    assert any(f.source_kind=='book' for f in dossier.facts)


def test_changed_edition_retires_old_provider_facts_even_if_offline():
    project,segments=fixture()
    research.research_project(project,segments,external_provider=provider(project,segments,API()))
    previous=next(f for f in project.character_cast.dossiers['alice'].facts if f.source_kind=='reference')
    segments[0].text="Alice's voice was bright and light."
    selected=external.provider_for_project(project,segments,fetcher=API(),online=False)
    research.research_project(project,segments,external_provider=selected)
    assert not any(f.id==previous.id for f in project.character_cast.dossiers['alice'].facts)
    assert any(raw['id']==previous.id for raw in project.character_cast.external_cache['external-evidence-history:alice']['retired_facts'])


def test_durable_voice_trait_scope_continues_after_citation_chapter():
    project,segments=fixture()
    selected=provider(project,segments,API())
    facts=selected.research('Alice',aliases=[],source_title=None,max_chapter=0)
    assert facts[0].scope.chapter_start==0 and facts[0].scope.chapter_end is None


@pytest.mark.parametrize("title", ["One one", "Chapter one and two", "Two twenty", "Chapter 1 / Chapter 2",
    "Chapter 1 to 2", "Chapter 1 and Chapter Two", "Chapter 1 / Chapter Two", "Chapter 1 - Two"])
def test_ambiguous_or_invalid_number_headings_abstain(title):
    assert external.published_chapter(title) is None


def test_verified_elf_prelude_mapping_is_one_not_zero():
    assert external.published_chapter("PRELUDE AND CHAPTER 1 – The Frog and the Toad") == 1


def test_custom_project_label_needs_explicit_verified_public_binding():
    meta = {"source_title":"Elf_1_10 AutoCast Acceptance","source_author":"ljamberfantasy"}
    assert external.resolve_public_work(meta) is None
    meta["external_character_research_public_work"] = {
        "title":"The Elf Who Would Become A Dragon", "author":"ljamberfantasy",
        "identity_evidence":"https://www.royalroad.com/fiction/104434/the-elf-who-would-become-a-dragon-story-complete"}
    assert external.resolve_public_work(meta)["id"] == "elf-who-would-become-a-dragon"
    meta["external_character_research_public_work"]["author"]="Private author"
    assert external.resolve_public_work(meta) is None


def test_disabling_queries_still_retires_stale_edition_evidence_locally():
    project,segments=fixture()
    casting.research_and_profile(project,segments,external_provider=provider(project,segments,API()))
    previous=next(f for f in project.character_cast.dossiers['alice'].facts if f.source_kind=='reference')
    project.character_cast.policy.external_research=False
    segments[0].text="Alice's voice was bright and light."
    casting.research_and_profile(project,segments)
    assert not any(f.id==previous.id for f in project.character_cast.dossiers['alice'].facts)
    assert any(raw['id']==previous.id for raw in project.character_cast.external_cache['external-evidence-history:alice']['retired_facts'])


def official_fixture(text="Alice's voice was deep and clear."):
    project, segments = fixture(text)
    project.metadata = {"source_title": "The Elf Who Would Become A Dragon", "source_author": "ljamberfantasy"}
    return project, segments


def chapter_html(body="Alice's voice was deep and clear.", *, heading="Chapter 1", canonical=None, before="", after=""):
    url = external.ELF_CHAPTER_ONE if canonical is None else canonical
    return f'<html><link rel="canonical" href="{url}"><h1>{heading}</h1>{before}<div class="chapter-content"><p>{body}</p></div>{after}</html>'


def test_official_factory_exact_provenance_memo_and_book_priority():
    project, segments = official_fixture()
    calls = []
    def fetch(url, limits):
        calls.append(url)
        return chapter_html()
    selected = external.provider_for_project(project, segments, online=True, html_fetcher=fetch)
    assert isinstance(selected, external.OfficialChapterProvider)
    first = selected.research("Alice", aliases=["PRIVATE"], source_title="PRIVATE", max_chapter=0)
    assert len(first) == 1
    fact = first[0]
    assert fact.source_kind == "official" and fact.source == external.ELF_CHAPTER_ONE
    assert fact.source_revision_id.startswith("official-sha256:")
    assert "local-source-revision:source1;book-span:" in fact.location
    assert selected.research("Alice", aliases=[], source_title=None, max_chapter=0)[0].id == fact.id
    assert calls == [external.ELF_CHAPTER_ONE]
    research.research_project(project, segments, external_provider=selected)
    effective = research.effective_facts(project.character_cast.dossiers["alice"], project.character_cast.policy)
    assert next(f for f in effective if f.field == "voice_description").source_kind == "book"


@pytest.mark.parametrize("body", ["Alice's voice was bright and clear.", '"Alice\'s voice was deep and clear."',
    "If Alice's voice was deep and clear, we would leave.", "Alice Smith's voice was deep and clear."])
def test_official_wrong_edition_dialogue_conditional_and_alias_abstain(body):
    project, segments = official_fixture()
    selected = external.provider_for_project(project, segments, online=True, html_fetcher=lambda *args: chapter_html(body))
    assert selected.research("Alice", aliases=[], source_title=None, max_chapter=0) == []


@pytest.mark.parametrize("html", [chapter_html(heading="Chapter 2"), chapter_html(canonical="https://evil.invalid/chapter"),
    chapter_html().replace('class="chapter-content"', 'class="comments"'),
    chapter_html()+chapter_html()])
def test_official_page_identity_and_single_body_required(html):
    project, segments = official_fixture()
    selected = external.provider_for_project(project, segments, online=True, html_fetcher=lambda *args: html)
    with pytest.raises(research.ExternalResearchUnavailable):
        selected.research("Alice", aliases=[], source_title=None, max_chapter=0)


def test_official_comments_and_scripts_are_not_character_evidence():
    project, segments = official_fixture()
    html = chapter_html("No named traits here.", before="<p>Alice's voice was deep and clear.</p>", after="<p>Alice's voice was deep and clear.</p>")
    html = html.replace("No named traits here.", "<script>Alice's voice was deep and clear.</script><style>Alice's voice was deep and clear.</style>")
    selected = external.provider_for_project(project, segments, online=True, html_fetcher=lambda *args: html)
    assert selected.research("Alice", aliases=[], source_title=None, max_chapter=0) == []


def test_official_unloaded_duplicate_and_policy_hidden_chapters_never_fetch():
    for chapters, maximum in [([Chapter(id="chapter", index=0, title="Chapter 2")], None),
                              ([Chapter(id="chapter", index=1, title="Chapter 1")], 0),
                              ([Chapter(id="chapter", index=0, title="Chapter 1"), Chapter(id="other", index=1, title="Chapter 1")], None)]:
        project, segments = official_fixture()
        project.chapters = chapters
        calls = []
        selected = external.provider_for_project(project, segments, online=True, html_fetcher=lambda *args, calls=calls: calls.append(args))
        assert selected.research("Alice", aliases=[], source_title=None, max_chapter=maximum) == []
        assert not calls


def test_official_offline_and_url_allowlist_are_enforced_before_fetch():
    project, segments = official_fixture()
    selected = external.provider_for_project(project, segments, html_fetcher=lambda *args: pytest.fail("offline fetch"))
    with pytest.raises(external.ResearchDeferred):
        selected.research("Alice", aliases=[], source_title=None, max_chapter=0)
    selected.online = True
    selected.work["official_chapters"][0]["url"] = "http://127.0.0.1/private"
    with pytest.raises(research.ExternalResearchUnavailable):
        selected.research("Alice", aliases=[], source_title=None, max_chapter=0)


def test_official_owned_claim_retirement_retains_history():
    project, segments = official_fixture()
    selected = external.provider_for_project(project, segments, online=True, html_fetcher=lambda *args: chapter_html())
    research.research_project(project, segments, external_provider=selected)
    original = next(f for f in project.character_cast.dossiers["alice"].facts if f.source_kind == "official")
    project.character_cast.external_cache = {}
    refreshed = external.provider_for_project(project, segments, online=True, html_fetcher=lambda *args: chapter_html("No named traits remain."))
    research.research_project(project, segments, external_provider=refreshed)
    assert not any(f.id == original.id for f in project.character_cast.dossiers["alice"].facts)
    assert any(f["id"] == original.id for f in project.character_cast.external_cache["external-evidence-history:alice"]["retired_facts"])


def test_external_flag_constructs_official_factory_without_caller_injection(monkeypatch):
    project, segments = official_fixture()
    saved = []
    monkeypatch.setattr(casting, "project_meta_lock", lambda *args: contextmanager(lambda: (yield))())
    monkeypatch.setattr(casting, "load_project", lambda *args: project.model_copy(deep=True))
    monkeypatch.setattr(casting, "load_segments", lambda *args: segments)
    monkeypatch.setattr(casting, "save_project", lambda value: saved.append(value))
    calls = []
    def chapter(self, url):
        calls.append((self.online, url))
        return chapter_html()
    monkeypatch.setattr(external.OfficialChapterProvider, "_chapter", chapter)
    casting.refresh_cast(project.id)
    assert calls == [(True, external.ELF_CHAPTER_ONE)]
    assert any(f.source_kind == "official" for f in saved[0].character_cast.dossiers["alice"].facts)
