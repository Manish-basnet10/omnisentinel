import pytest
import shutil
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from ml.serving.inference_server import app
from ml.serving.services.pcap_validator import validate_pcap, PCAPValidationError

client = TestClient(app)

# Dummy test PCAP file content (a few bytes that look like a pcap)
DUMMY_PCAP_MAGIC = b"\xd4\xc3\xb2\xa1\x02\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

@pytest.fixture
def dummy_pcap():
    tmp_dir = Path(tempfile.mkdtemp())
    pcap_path = tmp_dir / "test.pcap"
    with open(pcap_path, "wb") as f:
        f.write(DUMMY_PCAP_MAGIC)
    yield pcap_path
    shutil.rmtree(tmp_dir, ignore_errors=True)

def test_pcap_validator(dummy_pcap):
    # Should pass
    valid, msg = validate_pcap(dummy_pcap)
    assert valid

def test_pcap_validator_bad_ext(dummy_pcap):
    # Rename to .txt
    bad_file = dummy_pcap.with_suffix(".txt")
    dummy_pcap.rename(bad_file)
    with pytest.raises(PCAPValidationError):
        validate_pcap(bad_file)

def test_pcap_validator_empty():
    tmp = Path(tempfile.mktemp(suffix=".pcap"))
    tmp.touch()
    try:
        with pytest.raises(PCAPValidationError):
            validate_pcap(tmp)
    finally:
        tmp.unlink(missing_ok=True)

def test_zeek_status_endpoint():
    response = client.get("/api/zeek-status")
    assert response.status_code == 200
    data = response.json()
    assert "zeek_available" in data

from unittest.mock import patch

def test_analyze_pcap_endpoint_bad_file():
    # Mock Path.exists to return False for xgb_binary.pkl to avoid segfault during testing
    original_exists = Path.exists
    def mock_exists(self):
        if self.name == "xgb_binary.pkl":
            return False
        return original_exists(self)
    
    with patch.object(Path, 'exists', new=mock_exists):
        # Use context manager to trigger app startup events so _model is loaded
        with TestClient(app) as client:
            # Test uploading a file that fails validation
            response = client.post(
                "/api/analyze-pcap",
                files={"file": ("test.txt", b"not a pcap", "text/plain")}
            )
            assert response.status_code == 422
            assert "Invalid PCAP" in response.json()["detail"]

