import os

import pytest

from rag_bencher.providers.aws.auth import is_installed as aws_installed
from rag_bencher.providers.azure.auth import is_installed as az_installed
from rag_bencher.providers.gcp.auth import is_installed as gcp_installed

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_provider_detection_env_driven() -> None:
    expect_gcp = os.environ.get("EXPECT_GCP_INSTALLED")
    expect_aws = os.environ.get("EXPECT_AWS_INSTALLED")
    expect_az = os.environ.get("EXPECT_AZURE_INSTALLED")
    if expect_gcp is not None:
        assert gcp_installed() == (expect_gcp.lower() == "true")
    if expect_aws is not None:
        assert aws_installed() == (expect_aws.lower() == "true")
    if expect_az is not None:
        assert az_installed() == (expect_az.lower() == "true")
