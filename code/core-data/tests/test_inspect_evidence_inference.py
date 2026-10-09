import io
import sys
import tarfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspect_evidence_inference import read_member  # noqa: E402


class EvidenceInferenceInspectionTests(unittest.TestCase):
    def archive(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            for name, content in {
                "annotations/splits/ev2_train_article_ids.txt": b"1\n",
                "annotations/splits/ev2_test_article_ids.txt": b"2\n",
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        payload.seek(0)
        return tarfile.open(fileobj=payload, mode="r:gz")

    def test_development_split_is_readable(self):
        with self.archive() as archive:
            self.assertEqual(
                read_member(archive, "annotations/splits/ev2_train_article_ids.txt"), b"1\n"
            )

    def test_test_split_fails_closed(self):
        with self.archive() as archive, self.assertRaisesRegex(ValueError, "sealed"):
            read_member(archive, "annotations/splits/ev2_test_article_ids.txt")


if __name__ == "__main__":
    unittest.main()
