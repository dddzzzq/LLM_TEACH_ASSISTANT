"""Provision a portal credential key once, outside the source and database."""

import argparse
import base64
import os
from pathlib import Path
import secrets
import stat


def provision(path):
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise SystemExit("Existing key must be a regular file with mode 0600.")
        try:
            value = base64.b64decode(path.read_bytes().strip(), validate=True)
        except ValueError:
            raise SystemExit("Existing key is invalid; it was not replaced.") from None
        if len(value) != 32:
            raise SystemExit("Existing key has the wrong size; it was not replaced.")
        return False
    with os.fdopen(fd, "wb") as output:
        output.write(base64.b64encode(secrets.token_bytes(32)) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = Path(os.environ.get("RUNTIME_DIR", Path(__file__).resolve().parents[2] / "teaching-runtime"))
    parser.add_argument("--path", default=os.environ.get("PORTAL_CREDENTIAL_KEY_FILE", str(runtime / ".portal-credential-key")))
    args = parser.parse_args()
    created = provision(args.path)
    print("Portal credential key provisioned." if created else "Existing portal key preserved.")
