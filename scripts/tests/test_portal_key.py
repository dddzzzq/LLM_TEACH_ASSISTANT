import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("portal_key", Path(__file__).resolve().parents[1] / "init-portal-key.py")
key_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(key_module)


@unittest.skipUnless(os.name == "posix", "portal credentials run inside the WSL/Linux service account")
class PortalKeyTests(unittest.TestCase):
    def test_provision_is_idempotent_and_never_replaces_invalid_existing_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime" / ".portal-credential-key"
            self.assertTrue(key_module.provision(path))
            value = path.read_bytes()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(key_module.provision(path))
            self.assertEqual(value, path.read_bytes())
            path.write_bytes(b"invalid existing key")
            with self.assertRaises(SystemExit):
                key_module.provision(path)
            self.assertEqual(path.read_bytes(), b"invalid existing key")

    def test_existing_symlink_and_public_key_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "key"
            key_module.provision(path)
            link = Path(folder) / "linked-key"
            link.symlink_to(path)
            with self.assertRaises(SystemExit):
                key_module.provision(link)
            path.chmod(0o644)
            with self.assertRaises(SystemExit):
                key_module.provision(path)


if __name__ == "__main__":
    unittest.main()
