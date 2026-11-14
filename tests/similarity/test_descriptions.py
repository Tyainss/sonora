from sonora.similarity.descriptions import clean_artist_description


def test_removes_lastfm_attribution_and_markup():
    raw = (
        'A <b>jazz</b> and <a href="https://example.org">folk</a> artist.<br>'
        '<a href="https://www.last.fm/music/Example">Read more on Last.fm</a>. '
        'User-contributed text is available under the Creative Commons '
        'By-SA License; additional terms may apply.'
    )
    assert clean_artist_description(raw) == "A jazz and folk artist."


def test_preserves_unicode_and_normalizes_whitespace():
    raw = "  Niños del Cerro\n  hacen música  latinoamericana.  &amp; rock  "
    assert clean_artist_description(raw) == (
        "Niños del Cerro hacen música latinoamericana. & rock"
    )


def test_removes_plain_urls_without_erasing_other_text():
    assert clean_artist_description("Listen to dream pop: https://example.com/music") == (
        "Listen to dream pop"
    )


def test_empty_and_link_only_descriptions_are_missing():
    assert clean_artist_description(None) is None
    assert clean_artist_description("") is None
    assert clean_artist_description("  123 ; ") is None
    assert clean_artist_description(
        'https://example.org <a href="https://www.last.fm">Read more on Last.fm</a>'
    ) is None


def test_cleaning_is_idempotent():
    raw = "<p>Experimental &amp; electronic</p> Read more on Last.fm"
    cleaned = clean_artist_description(raw)
    assert cleaned == "Experimental & electronic"
    assert clean_artist_description(cleaned) == cleaned
