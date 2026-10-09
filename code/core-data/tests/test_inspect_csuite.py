import io
import sys
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspect_csuite import read_development_member  # noqa: E402


class CSuiteInspectionTests(unittest.TestCase):
    def archive(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("train.csv", "1,2\n")
            archive.writestr("test.csv", "3,4\n")
            archive.writestr("interventions.json", "{}")
        payload.seek(0)
        return zipfile.ZipFile(payload)

    def test_development_member_is_readable(self):
        with self.archive() as archive:
            self.assertEqual(read_development_member(archive, "train.csv"), b"1,2\n")

    def test_test_members_fail_closed(self):
        with self.archive() as archive:
            for member in ("test.csv", "interventions.json", "counterfactuals.json"):
                with self.subTest(member=member), self.assertRaisesRegex(ValueError, "sealed"):
                    read_development_member(archive, member)


if __name__ == "__main__":
    unittest.main()
