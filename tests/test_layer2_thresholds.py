"""CPU-only checks for CLI thresholds, including their effect on saved labels.

Run from the repository root:
    python -m unittest discover -s tests -p test_layer2_thresholds.py
"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Load the actual Layer 2 modules without the top-level package's inference
# imports. These checks do not load models or require GPU libraries.
package_name = '_hmuttms_layer2_threshold_tests'
package = types.ModuleType(package_name)
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'src/hmuttms/layer2_validation')]
sys.modules[package_name] = package
cli = importlib.import_module(package_name + '.run_layer2')
decision = importlib.import_module(package_name + '.decision_matrix')


class ThresholdTests(unittest.TestCase):
    def test_cli_defaults_match_direct_api(self):
        args = cli._parse_args([])
        config = decision.DecisionConfig()
        for name in ('a_fail_below', 'b_fail_below', 'b_pass_above'):
            self.assertEqual(getattr(args, name), getattr(config, name))
        self.assertEqual(args.b_pass_above, 0.60)

    def test_two_of_three_and_explicit_historical_override(self):
        self.assertEqual(decision.classify(0.2, 2 / 3), decision.Classification.SUBSTITUTABLE)
        args = cli._parse_args(['--b-pass-above', '0.70'])
        config = decision.DecisionConfig(b_pass_above=args.b_pass_above)
        self.assertEqual(decision.classify(0.2, 2 / 3, config), decision.Classification.REVIEW)

    def test_strict_boundary_and_missing_score(self):
        self.assertEqual(decision.classify(0.2, 0.60), decision.Classification.REVIEW)
        self.assertEqual(decision.classify(0.3, 2 / 3), decision.Classification.FALSE_POSITIVE)
        self.assertEqual(decision.classify(0.2, None), decision.Classification.REVIEW)

    def test_cli_writes_labels_using_effective_threshold(self):
        class FixedProbeBackend:
            name = 'test-probe'

            def probe(self, token_id, token_str):
                return cli.TokenProbe(token_id, token_str, 2 / 3, 'handled', self.name)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / 'input.jsonl'
            output_path = Path(directory) / 'output.jsonl'
            input_path.write_text(json.dumps({'token_id': 1, 'token_str': 'example', 'magikarp_score': 0.2}) + '\n', encoding='utf-8')
            for overrides, expected, threshold in (
                ([], 'SUBSTITUTABLE', 0.60),
                (['--b-pass-above', '0.70'], 'REVIEW', 0.70),
            ):
                with self.subTest(threshold=threshold):
                    stdout = io.StringIO()
                    with patch.object(cli, '_build_backends', return_value=[FixedProbeBackend()]), contextlib.redirect_stdout(stdout):
                        result = cli.main(['--input', str(input_path), '--output', str(output_path)] + overrides)
                    self.assertEqual(result, 0)
                    self.assertEqual(json.loads(output_path.read_text(encoding='utf-8'))['classification'], expected)
                    self.assertEqual(json.loads(stdout.getvalue())['thresholds']['b_pass_above'], threshold)


if __name__ == '__main__':
    unittest.main()
