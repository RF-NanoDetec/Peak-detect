from service.storage import InMemoryStore


def test_in_memory_store_evicts_lru_result_and_matching_timing(monkeypatch):
    monkeypatch.setenv("PEAK_TOOL_MAX_RESULTS", "2")
    monkeypatch.setenv("PEAK_TOOL_RESULT_TTL_SECONDS", "600")

    clock = {"now": 0.0}
    monkeypatch.setattr("service.storage.time.monotonic", lambda: clock["now"])

    store = InMemoryStore()

    first = store.save_result({"value": 1})
    store.save_timing(first, {"name": "first"})

    clock["now"] = 1.0
    second = store.save_result({"value": 2})
    store.save_timing(second, {"name": "second"})

    clock["now"] = 2.0
    assert store.get_result(first) == {"value": 1}

    clock["now"] = 3.0
    third = store.save_result({"value": 3})
    store.save_timing(third, {"name": "third"})

    assert store.get_result(first) == {"value": 1}
    assert store.get_result(second) is None
    assert store.get_result(third) == {"value": 3}
    assert store.get_timing(second) is None
    assert set(store.get_all_timings()) == {first, third}


def test_in_memory_store_expires_results_and_timings(monkeypatch):
    monkeypatch.setenv("PEAK_TOOL_MAX_RESULTS", "5")
    monkeypatch.setenv("PEAK_TOOL_RESULT_TTL_SECONDS", "60")

    clock = {"now": 100.0}
    monkeypatch.setattr("service.storage.time.monotonic", lambda: clock["now"])

    store = InMemoryStore()
    result_id = store.save_result({"value": "cached"})
    store.save_timing(result_id, {"seconds": 1.2})

    clock["now"] = 161.0

    assert store.get_result(result_id) is None
    assert store.get_timing(result_id) is None
    assert store.get_all_timings() == {}
