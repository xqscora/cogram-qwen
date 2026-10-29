"""Generic hub activation must not swamp query-specific cited evidence."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import memory_api


class RetrievalRankingTests(unittest.TestCase):
    def recall(self, n_docs=100, query="dataclass shell", shell_df=None):
        graph = {
            "nodes": {
                "dataclass": {"df": 1, "pointers": [[0, 1]], "parents": []},
                "shell": {"df": n_docs if shell_df is None else shell_df, "pointers": [[1 if n_docs > 1 else 0, 2]], "parents": []},
            },
            "edges": [],
            "meta": {"tick": 1, "files": [f"public_{i}.md" for i in range(n_docs)]},
        }
        with patch.object(memory_api, "load_graph", return_value=graph), \
             patch.object(memory_api, "_persist_pending_gaps"), \
             patch.object(memory_api, "spread_activation_traced", return_value=({"dataclass": 1.0, "shell": 5.0}, {})), \
             patch.object(memory_api, "_resolve_line", side_effect=lambda fid, line, files: (files[fid], "dataclass default_factory fix" if line == 1 else "generic shell prompt")):
            return memory_api.recommend(query, budget_tokens=100)

    def test_rare_evidence_outweighs_high_activation_boilerplate(self):
        result = self.recall()
        self.assertEqual(result["recommendations"][0]["line_idx"], 1)
        self.assertIn("default_factory", result["recommendations"][0]["text"])
        self.assertIn("dataclass", result["recommendations"][0]["path"][0])

    def test_single_document_remains_retrievable(self):
        result = self.recall(n_docs=1)
        self.assertGreater(len(result["recommendations"]), 0)
        self.assertTrue(all(row["strength"] > 0 for row in result["recommendations"]))

    def test_direct_query_evidence_precedes_rare_but_unrelated_hops(self):
        result = self.recall(query="dataclass", shell_df=1)
        self.assertEqual(result["recommendations"][0]["line_idx"], 1)
        self.assertEqual(result["recommendations"][0]["path"][0], "query_term 'dataclass'")

    def test_budget_and_citations_are_preserved(self):
        result = self.recall()
        self.assertLessEqual(result["stats"]["tokens_used"], 100)
        self.assertTrue(all(row["session"].startswith("public_") for row in result["recommendations"]))
        self.assertEqual({row["line_idx"] for row in result["recommendations"]}, {1, 2})


if __name__ == "__main__":
    unittest.main()
