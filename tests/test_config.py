def test_all_configs_load(reg):
    assert {"faithfulness", "helpfulness", "safety", "format_adherence"} <= set(reg.rubric_versions)
    assert {"kimi-k3", "glm-5.3"} <= set(reg.judges)
    for s in reg.systems.values():
        assert s.model in reg.models, s.id
        assert s.prompt_text(), s.id
    for j in reg.judges.values():
        assert j.model in reg.models
    for m in reg.models.values():
        assert m.provider in reg.providers


def test_rubric_versioning(reg):
    r = reg.rubric("faithfulness")
    assert r.key == "faithfulness@1" and reg.rubric("faithfulness@1").hash == r.hash
    assert set(r.anchors) == {1, 2, 3, 4, 5}


def test_dataset_integrity(reg):
    ds = reg.dataset("support_v1")
    assert len(ds.items) >= 40
    assert len({i.id for i in ds.items}) == len(ds.items), "duplicate item ids"
    assert set(ds.sections) == {f"S{i}" for i in range(1, 13)}
    for it in ds.items:
        assert it.key_facts, it.id
        assert all(s in ds.sections for s in it.context_sections), it.id
    ids = {i.id for i in ds.items}
    assert all(p["item_id"] in ids for p in ds.planted)
    assert "120 words" in ds.style_guide


def test_config_hash_changes_with_rubric(reg):
    kw = dict(dataset="support_v1", systems=["mock-good"], judges=["mock-judge-strict"])
    assert reg.config_hash(**kw, rubrics=["faithfulness"]) != reg.config_hash(**kw, rubrics=["helpfulness"])
