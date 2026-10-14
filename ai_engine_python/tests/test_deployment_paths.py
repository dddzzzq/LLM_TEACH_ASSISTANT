import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.paths import model_path


class DeploymentPathTests(unittest.TestCase):
    def test_sibling_models_do_not_depend_on_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            try:
                root = Path(directory)
                repository = root / "repository"
                model = root / "models/bert-base-chinese"
                repository.mkdir()
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}", encoding="utf-8")
                os.chdir(root)
                with patch("app.core.paths.PROJECT_ROOT", repository):
                    self.assertEqual(Path(model_path("bert-base-chinese")), model)
            finally:
                os.chdir(previous)

    def test_missing_model_configuration_fails_before_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TEACH_MODELS_DIR": directory}):
            with self.assertRaisesRegex(FileNotFoundError, "TEACH_MODELS_DIR"):
                model_path("missing")


if __name__ == "__main__":
    unittest.main()
