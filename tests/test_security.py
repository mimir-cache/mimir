from mimir_cache.security import ProbeDetector, SecurityMonitor


def test_probe_detector_fires_at_threshold():
    detector = ProbeDetector(window_seconds=60, threshold=3)

    assert not detector.record_and_check("t1", "same prompt")
    assert not detector.record_and_check("t1", "same prompt")
    assert detector.record_and_check("t1", "same prompt")


def test_probe_detector_is_tenant_and_prompt_scoped():
    detector = ProbeDetector(window_seconds=60, threshold=2)

    assert not detector.record_and_check("t1", "prompt")
    assert not detector.record_and_check("t2", "prompt")  # other tenant
    assert not detector.record_and_check("t1", "different prompt")


def test_monitor_records_collision_alerts():
    monitor = SecurityMonitor()
    monitor.record_collision_suspect("t1", "attack prompt", "target prompt", 0.9995)

    alerts = monitor.recent_alerts("t1")

    assert len(alerts) == 1
    assert alerts[0].alert_type == "collision_suspect"
    assert alerts[0].details["score"] == 0.9995


def test_monitor_probe_alert_and_tenant_filter():
    monitor = SecurityMonitor(probe_window_seconds=60, probe_threshold=2)
    monitor.check_probe("t1", "p")
    alert = monitor.check_probe("t1", "p")

    assert alert is not None
    assert alert.alert_type == "probe_pattern"
    assert monitor.recent_alerts("t2") == []
