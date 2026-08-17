import json
from unittest.mock import patch

import pandas as pd
from django.core.management import call_command

from ai_app.management.commands.benchmark_faq_clustering import (
    DEFAULT_CLUSTERING_CANDIDATES,
    SELECTED_PRODUCTION_CANDIDATE,
)


def test_default_clustering_candidates_compare_representation_and_hdbscan_selection_methods():
    representations = {candidate.representation for candidate in DEFAULT_CLUSTERING_CANDIDATES}
    selection_methods = {
        candidate.config.cluster_selection_method for candidate in DEFAULT_CLUSTERING_CANDIDATES
    }

    assert {"legacy_keyword_content", "title_only", "title_keywords"} <= representations
    assert selection_methods == {"eom", "leaf"}
    assert all(
        not hasattr(candidate.config, "target_clusters_per_category")
        for candidate in DEFAULT_CLUSTERING_CANDIDATES
    )
    assert SELECTED_PRODUCTION_CANDIDATE == (
        "multilingual-e5-small-title_keywords-leaf-mcs3-ms2"
    )


def test_clustering_command_writes_the_documented_selection_without_initializing_a_model(tmp_path):
    output_path = tmp_path / "clustering.json"
    questions = pd.DataFrame(
        [
            {
                "id": "1",
                "tag": "[증상] 허리 통증",
                "title": "허리 통증",
                "content": "본문",
            }
        ]
    )
    selected_report = {
        "name": SELECTED_PRODUCTION_CANDIDATE,
        "model_name": "intfloat/multilingual-e5-small",
        "representation": "title_keywords",
        "config": {"min_cluster_size": 3, "min_samples": 2, "cluster_selection_method": "leaf"},
        "categories": {},
        "selection_score": 0.5,
    }

    with patch(
        "ai_app.management.commands.benchmark_faq_clustering.load_raw_qa_dataframe",
        return_value=questions,
    ), patch(
        "ai_app.management.commands.benchmark_faq_clustering.run_clustering_benchmark",
        return_value=[selected_report],
    ):
        call_command(
            "benchmark_faq_clustering",
            "--candidate",
            SELECTED_PRODUCTION_CANDIDATE,
            "--output",
            str(output_path),
        )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["selected"]["name"] == SELECTED_PRODUCTION_CANDIDATE
    assert report["selection_decision"]["automated_best_candidate"] == SELECTED_PRODUCTION_CANDIDATE
