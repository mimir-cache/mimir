from mimir_cache.normalizer import context_hash, exact_match_key, normalize_prompt


def test_should_lowercase_and_collapse_whitespace():
    assert normalize_prompt("  Hello   WORLD  ") == "hello world"


def test_should_strip_filler_phrases():
    assert normalize_prompt("Can you explain semantic caching?") == "semantic caching?"


def test_should_remove_disallowed_symbols():
    assert normalize_prompt("what's #semantic @caching!") == "whats semantic caching"


def test_equivalent_phrasings_normalize_identically():
    a = normalize_prompt("Please tell me about vector databases")
    b = normalize_prompt("tell me about Vector Databases")
    assert a == b == "vector databases"


def test_exact_match_key_is_tenant_scoped():
    key_a = exact_match_key("same prompt", "tenant_a")
    key_b = exact_match_key("same prompt", "tenant_b")
    assert key_a != key_b
    assert key_a.startswith("tenant:tenant_a:exact:")


def test_exact_match_key_varies_with_context():
    ctx = context_hash(["turn one", "turn two"])
    assert exact_match_key("p", "t", "") != exact_match_key("p", "t", ctx)


def test_context_hash_empty_for_no_context():
    assert context_hash(None) == ""
    assert context_hash([]) == ""
