"""Integration and Statistical Tests for Drosophila In-Silico Lesion Battery."""

import os
import sys
import tempfile
import unittest
import numpy as np

# Adjust module path
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from experiments.lesion_study import (
    CONDITIONS,
    run_single_trial,
    run_battery,
    compute_one_way_anova,
    compute_welch_t_test,
    compute_group_statistics,
    f_distribution_p_value,
    t_distribution_p_value,
    export_csv,
    export_json,
    generate_markdown_report
)


class TestLesionStudyBattery(unittest.TestCase):
    def test_single_trial_all_conditions(self):
        """Verifies that all 6 knockout cohorts execute single trials with valid metrics."""
        for cond_key in CONDITIONS.keys():
            res = run_single_trial(
                condition_key=cond_key,
                trial_id=1,
                seed=42,
                max_steps=150
            )

            # Check expected keys
            self.assertEqual(res['condition'], cond_key)
            self.assertIn(res['success'], [0, 1])
            self.assertIn(res['killed'], [0, 1])
            self.assertTrue(1 <= res['ttf'] <= 150)
            self.assertGreater(res['distance'], 0.0)
            self.assertTrue(0.0 <= res['pte'] <= 1.0)
            self.assertGreaterEqual(res['tortuosity'], 1.0)
            self.assertTrue(-1.0 <= res['upwind_fidelity'] <= 1.0)
            self.assertTrue(0.0 <= res['satiety_auc'] <= 1.0)
            self.assertGreaterEqual(res['escapes_count'], 0)

    def test_statistical_distribution_functions(self):
        """Verifies precision of pure-Python Fisher F and Student/Welch t survival functions."""
        # F(1, 100) = 3.936 gives p ~ 0.05
        p_f = f_distribution_p_value(1, 100, 3.936)
        self.assertAlmostEqual(p_f, 0.05, delta=0.005)

        # t(100) = 1.984 gives two-tailed p ~ 0.05
        p_t = t_distribution_p_value(100, 1.984)
        self.assertAlmostEqual(p_t, 0.05, delta=0.005)

        # High significance
        p_extreme = f_distribution_p_value(5, 500, 50.0)
        self.assertLess(p_extreme, 1e-15)

    def test_welch_t_and_cohens_d(self):
        """Verifies Welch's unequal variance t-test and Cohen's d effect size calculations."""
        # Group 1 (Treatment / Lesion): lower mean
        g1 = [10.0, 12.0, 11.0, 10.5, 11.5, 9.5]
        # Group 2 (WT Control): higher mean
        g2 = [20.0, 22.0, 21.0, 20.5, 21.5, 19.5]

        res = compute_welch_t_test(g1, g2)
        self.assertAlmostEqual(res['delta_mean'], -10.0, places=2)
        self.assertLess(res['p_val'], 0.001)
        self.assertEqual(res['effect_magnitude'], 'Large')
        self.assertLess(res['cohens_d'], -5.0)

    def test_one_way_anova(self):
        """Verifies One-Way ANOVA computation across distinct synthetic groups."""
        groups = {
            'A': [1.0, 2.0, 1.5, 1.8],
            'B': [5.0, 5.5, 4.8, 5.2],
            'C': [10.0, 10.2, 9.8, 10.5]
        }
        anova = compute_one_way_anova(groups)
        self.assertEqual(anova['df_between'], 2)
        self.assertEqual(anova['df_within'], 9)
        self.assertGreater(anova['f_stat'], 50.0)
        self.assertLess(anova['p_val'], 1e-5)

    def test_mini_battery_and_exporters(self):
        """Verifies end-to-end execution of a mini-battery with CSV, JSON, and report generation."""
        trials, summary = run_battery(
            trials_per_condition=2,
            base_seed=123,
            max_steps=50,
            verbose=False
        )

        self.assertEqual(len(trials), 12)  # 6 conditions * 2 trials
        self.assertIn('anova', summary)
        self.assertIn('pairwise_vs_wt', summary)
        self.assertIn('metrics', summary)

        # Test exporters with temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_trials.csv')
            json_path = os.path.join(tmpdir, 'test_summary.json')
            report_path = os.path.join(tmpdir, 'test_report.md')

            export_csv(trials, csv_path)
            export_json(summary, json_path)
            generate_markdown_report(summary, report_path)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(report_path))

            # Verify CSV row count
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 13)  # Header + 12 trials

            # Verify report contains markdown table
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('## 2. Summary Results Table', content)
                self.assertIn('## 3. One-Way ANOVA Hypothesis Testing', content)


if __name__ == '__main__':
    unittest.main()
