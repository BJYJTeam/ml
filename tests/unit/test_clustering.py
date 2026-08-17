from ai_app.evaluation import summarize_clusters


def test_summarize_clusters_reports_noise_and_cluster_size_distribution():
    report = summarize_clusters([0, 0, 1, -1, -1])

    assert report == {
        "n_samples": 5,
        "n_clusters": 2,
        "noise_count": 2,
        "noise_ratio": 0.4,
        "cluster_sizes": {"0": 2, "1": 1},
        "largest_cluster_ratio": 0.4,
    }


def test_summarize_clusters_handles_all_noise_without_division_error():
    report = summarize_clusters([-1, -1])

    assert report["n_clusters"] == 0
    assert report["noise_ratio"] == 1.0
    assert report["cluster_sizes"] == {}
    assert report["largest_cluster_ratio"] == 0.0


def test_summarize_clusters_handles_an_empty_input():
    report = summarize_clusters([])

    assert report["n_samples"] == 0
    assert report["noise_ratio"] == 0.0
    assert report["largest_cluster_ratio"] == 0.0
