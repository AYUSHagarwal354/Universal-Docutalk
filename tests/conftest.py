"""
Shared pytest configuration.

Patches all heavy external dependencies (IBM IAM, Elasticsearch, Redis)
at the sys.modules level before any app module is imported, preventing real
network/auth calls during the test session.
"""

import os
import sys
from unittest.mock import MagicMock

# ── 1. Add app directory to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# ── 2. IBM Cloud SDK ──────────────────────────────────────────────────────────
# token_manager.get_token() must return a str so "Bearer " + token works
mock_iam_instance = MagicMock()
mock_iam_instance.token_manager.get_token.return_value = "fake-test-token"
mock_iam_class = MagicMock(return_value=mock_iam_instance)

mock_ibm_auth_module = MagicMock()
mock_ibm_auth_module.IAMAuthenticator = mock_iam_class

sys.modules["ibm_cloud_sdk_core"] = MagicMock()
sys.modules["ibm_cloud_sdk_core.authenticators"] = mock_ibm_auth_module

# ── 3. Elasticsearch ──────────────────────────────────────────────────────────
sys.modules["elasticsearch"] = MagicMock()

# ── 4. Redis ──────────────────────────────────────────────────────────────────
sys.modules["redis"] = MagicMock()

# httpx stays real: fastapi's TestClient subclasses it, and test_main.py cannot even be collected with a mock here
