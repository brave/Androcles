import pytest


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--model-path",
        action="store",
        default="test_model",
        help="Path to the model directory"
    )


@pytest.fixture
def model_path(request):
    """Fixture that returns the model path from command line."""
    return request.config.getoption("--model-path")
