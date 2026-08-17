from ai_app.application.clustering_benchmark import (
    score_cluster_diagnostics,
    select_best_cluster_candidate,
)


def test_cluster_score_rewards_cohesion_coverage_and_non_dominance_not_cluster_count():
    diagnostics = {
        "n_samples": 20,
        "n_clusters": 2,
        "noise_ratio": 0.2,
        "largest_cluster_ratio": 0.5,
        "mean_cluster_cohesion": 0.8,
        "mean_keyword_jaccard": 0.7,
    }
    same_quality_with_different_cluster_count = {**diagnostics, "n_clusters": 8}

    assert score_cluster_diagnostics(diagnostics) == score_cluster_diagnostics(
        same_quality_with_different_cluster_count
    )
    assert score_cluster_diagnostics(diagnostics) > score_cluster_diagnostics(
        {
            **diagnostics,
            "noise_ratio": 0.8,
            "largest_cluster_ratio": 0.9,
            "mean_keyword_jaccard": 0.1,
        }
    )


def test_select_best_cluster_candidate_uses_quality_score_without_target_count():
    candidates = [
        {"name": "large-dominant", "selection_score": 0.45},
        {"name": "coherent-balanced", "selection_score": 0.72},
    ]

    assert select_best_cluster_candidate(candidates)["name"] == "coherent-balanced"
