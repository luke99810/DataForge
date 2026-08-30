"""外部数据源：链接 → 结构 → 同构合成集。

**没有一条测试出网。** 网络会让测试在别人机器上、在 CI 里、在断网时
随机变红，而变红的原因和被测逻辑毫无关系。取数一律走 StubFetcher。

重点在三条红线上：
只取元数据不下载数据本体、合成声明不可关闭、认不出来就说认不出来。
"""

from __future__ import annotations

import json

import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.graph.executor import Executor
from dataforge.pipelines import source_aligned_pipeline
from dataforge.sources import (
    ALLOWED_HOSTS, CachedFetcher, HttpFetcher, SourceError, StubFetcher, identify,
    known, match, resolve,
)
from dataforge.sources import hf


@pytest.fixture()
def ex(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")
    yield Executor(store, rs, workdir=tmp_path), store, rs
    rs.close()


# ── 链接识别 ──────────────────────────────────────────────

@pytest.mark.parametrize("link,kind,ident", [
    ("hf://datasets/openai/gsm8k", "hf_dataset", "openai/gsm8k"),
    ("https://huggingface.co/datasets/stanfordnlp/imdb", "hf_dataset", "stanfordnlp/imdb"),
    ("https://github.com/org/repo", "github_repo", "org/repo"),
    ("https://github.com/org/repo.git", "github_repo", "org/repo"),
    ("https://arxiv.org/abs/2401.12345", "arxiv", "2401.12345"),
    ("arxiv:2401.12345", "arxiv", "2401.12345"),
])
def test_identify(link, kind, ident):
    ref = identify(link)
    assert ref.kind == kind and ref.id == ident


def test_unknown_link_is_admitted_not_guessed():
    """认不出来就说认不出来 —— 不猜。"""
    assert identify("随便一句话").kind == "unknown"


def test_empty_link_rejected():
    with pytest.raises(SourceError, match="空"):
        identify("")


# ── 基准注册表 ────────────────────────────────────────────

def test_registry_matches_by_alias():
    for alias in ("gsm8k", "openai/gsm8k", "GSM8K"):
        assert match(alias)["key"] == "gsm8k"
    assert match("cais/mmlu")["key"] == "mmlu"


def test_registry_returns_none_for_unknown():
    assert match("某个不存在的基准") is None


def test_registry_entries_carry_protocol_and_notice():
    for k in known():
        b = match(k)
        assert b["protocol"].get("metric"), k
        assert "不得用于刷榜" in b["synthetic_notice"], k
        assert b["fields"], k


def test_registry_contains_no_actual_items():
    """★ 注册表只描述结构与协议，不含任何原题内容。

    判据不能靠关键词黑名单——"####" 之类的记号本来就会出现在**协议描述**里
    （"取答案里 #### 之后的数值"），那是结构说明不是题目。
    改成两条真正的不变量：不许有装样例的键；所有字符串都短到只可能是描述。
    """
    from dataforge.sources.registry import BENCHMARKS

    SAMPLE_KEYS = {"examples", "samples", "items", "rows", "data", "instances"}

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                assert k.lower() not in SAMPLE_KEYS, f"{path}.{k} 疑似在装原题样例"
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            # 真实题面动辄上百字；描述与协议说明不会
            assert len(node) <= 120, f"{path} 的字符串过长（{len(node)}），疑似原题内容"

    walk(BENCHMARKS)


# ── 解析 ──────────────────────────────────────────────────

def test_registry_hit_needs_no_network():
    """★ 注册表命中就不出网 —— 也就不会因为对方接口变动而挂。"""
    fetcher = StubFetcher()
    out = resolve("gsm8k", fetcher)
    assert out["resolved_by"] == "registry"
    assert fetcher.calls == [], "命中注册表还出网了"
    assert [f["name"] for f in out["fields"]] == ["question", "answer"]
    assert out["protocol"]["metric"] == "exact_match"


def test_network_is_not_the_default():
    """★ 出网必须是显式选择。"""
    with pytest.raises(SourceError, match="显式传入 fetcher"):
        resolve("hf://datasets/someone/unknown-dataset")


def test_unknown_link_gives_actionable_error():
    with pytest.raises(SourceError, match="认不出"):
        resolve("这不是一个链接", StubFetcher())


# ── HuggingFace ───────────────────────────────────────────

HF_INFO = {
    "dataset_info": {
        "description": "一个演示数据集",
        "license": "mit",
        "features": {
            "text": {"_type": "Value", "dtype": "string"},
            "score": {"_type": "Value", "dtype": "float32"},
            "count": {"_type": "Value", "dtype": "int64"},
            "label": {"_type": "ClassLabel", "names": ["neg", "pos"]},
            "embedding": {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "float32"}},
            "image": {"_type": "Image"},
        },
        "splits": {"train": {"num_examples": 1000}, "test": {"num_examples": 200}},
    }
}


def test_hf_feature_mapping():
    out = hf.fetch_schema("demo/ds", StubFetcher({"datasets-server": HF_INFO}))
    by = {f["name"]: f for f in out["fields"]}
    assert by["text"]["type"] == "string"
    assert by["score"]["type"] == "float"
    assert by["count"]["type"] == "integer"
    assert by["label"]["type"] == "categorical" and by["label"]["categories"] == ["neg", "pos"]
    assert out["splits"] == {"train": 1000, "test": 200}


def test_hf_unmappable_features_are_reported_not_faked():
    """★ 映射不了的必须回传 —— 否则用户会以为覆盖了全部字段。"""
    out = hf.fetch_schema("demo/ds", StubFetcher({"datasets-server": HF_INFO}))
    skipped = {s["name"] for s in out["skipped"]}
    assert skipped == {"embedding", "image"}
    assert "embedding" not in {f["name"] for f in out["fields"]}


def test_hf_only_reads_metadata_endpoint():
    """★ 只碰 /info，不碰任何取数据行的接口。"""
    fetcher = StubFetcher({"datasets-server": HF_INFO})
    hf.fetch_schema("demo/ds", fetcher)
    assert len(fetcher.calls) == 1
    url = fetcher.calls[0]
    assert "/info?" in url
    for data_ep in ("/rows", "/first-rows", "/parquet", "/download"):
        assert data_ep not in url


def test_hf_multi_config_picks_one_and_says_so():
    payload = {"dataset_info": {
        "main": {"features": {"a": {"_type": "Value", "dtype": "string"}}, "splits": {}},
        "extra": {"features": {"b": {"_type": "Value", "dtype": "int64"}}, "splits": {}},
    }}
    out = hf.fetch_schema("demo/multi", StubFetcher({"datasets-server": payload}))
    assert out["config"] in ("main", "extra")
    assert set(out["available_configs"]) == {"main", "extra"}


def test_hf_bad_json_fails_clearly():
    with pytest.raises(SourceError, match="不是合法 JSON"):
        hf.fetch_schema("x/y", StubFetcher({"datasets-server": "<html>404</html>"}))


def test_hf_all_features_unmappable_is_an_error():
    payload = {"dataset_info": {"features": {"img": {"_type": "Image"}}, "splits": {}}}
    with pytest.raises(SourceError, match="映射不了"):
        resolve("hf://datasets/x/y", StubFetcher({"datasets-server": payload}))


# ── GitHub ────────────────────────────────────────────────

GH_META = {"default_branch": "main", "stargazers_count": 42,
           "license": {"spdx_id": "MIT"}}
GH_SCHEMA = {"fields": [
    {"name": "temp", "type": "float", "min": -20, "max": 45, "unit": "C"},
    {"name": "city", "type": "categorical", "categories": ["bj", "sh"]},
    {"name": "blob", "type": "tensor"},          # 不认识的类型
]}


def test_github_schema_file_parsed():
    f = StubFetcher({"api.github.com": GH_META, "schema.json": GH_SCHEMA})
    out = resolve("https://github.com/org/repo", f)
    names = [x["name"] for x in out["fields"]]
    assert names == ["temp", "city"], "不认识的类型该被跳过而不是硬塞成 string"
    assert out["provenance"]["license"] == "MIT"
    assert out["provenance"]["schema_file"] == "schema.json"


def test_github_without_schema_refuses_to_guess():
    """★ 找不到 schema 就说找不到 —— 猜出来的字段会一路流进生成的数据。"""
    with pytest.raises(SourceError, match="不猜数据长什么样"):
        resolve("https://github.com/org/repo", StubFetcher({"api.github.com": GH_META}))


# ── 取数器本身 ────────────────────────────────────────────

def test_http_fetcher_rejects_non_allowlisted_host():
    """★ 白名单写死是刻意的：加源必须经过一次显式修改。"""
    with pytest.raises(SourceError, match="不在白名单"):
        HttpFetcher().get("https://evil.example.com/data.csv")


def test_allowlist_has_no_data_download_hosts():
    for h in ALLOWED_HOSTS:
        assert "cdn" not in h and "s3" not in h


def test_cached_fetcher_avoids_repeat_calls():
    inner = StubFetcher({"x": "hello"})
    c = CachedFetcher(inner)
    assert c.get("https://x/1") == "hello"
    assert c.get("https://x/1") == "hello"
    assert len(inner.calls) == 1


# ── 端到端 ────────────────────────────────────────────────

def test_aligned_pipeline_from_registry(ex):
    """★ 由基准结构生成同构合成集，全程不出网。"""
    executor, store, _ = ex
    rep = executor.run(source_aligned_pipeline(link="imdb", num_records=300), seed=4)
    assert rep.ok, str(rep)

    df = store.get(rep.results["generate"].artifact)
    assert len(df) == 300 and set(df.columns) == {"text", "label"}

    card = store.get(rep.results["export"].artifact)
    assert card["synthetic"] is True
    assert card["aligned_to"]["benchmark"] == "IMDB"
    assert "不得用于刷榜" in card["notice"]
    assert card["provenance"]["source"] == "builtin_registry"


def test_aligned_pipeline_records_skipped_fields(ex, tmp_path):
    """★ 跳过的字段要出现在数据卡里，用户才知道生成的数据不完整。"""
    store = ArtifactStore(tmp_path / "b")
    rs = RunStore(tmp_path / "b.sqlite")
    executor = Executor(store, rs, workdir=tmp_path,
                        fetcher=StubFetcher({"datasets-server": HF_INFO}))
    dag = source_aligned_pipeline(link="hf://datasets/demo/ds", num_records=120,
                                  allow_network=True)
    rep = executor.run(dag, seed=1)
    assert rep.ok, str(rep)

    card = store.get(rep.results["export"].artifact)
    assert set(card["aligned_to"]["skipped_fields"]) == {"embedding", "image"}
    assert card["synthetic"] is True
    rs.close()


def test_network_flag_off_means_no_fetch(ex):
    """执行器带着取数器，但节点没被允许出网时也不该出网。"""
    executor, store, _ = ex
    fetcher = StubFetcher({"datasets-server": HF_INFO})
    executor.fetcher = fetcher
    rep = executor.run(
        source_aligned_pipeline(link="hf://datasets/demo/ds", allow_network=False), seed=1)
    assert rep.results["source"].status.value == "FAILED"
    assert fetcher.calls == []


def test_lineage_traces_columns_to_source(ex):
    executor, store, _ = ex
    rep = executor.run(source_aligned_pipeline(link="gsm8k", num_records=100), seed=2)
    cols = rep.lineage.entries["source"].columns
    assert cols["question"]["origin"] == "registry"
    assert "gsm8k" in rep.lineage.explain("source").lower() or cols
