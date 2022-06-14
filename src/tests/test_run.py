"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path
import sys
import pytest
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession


@pytest.fixture
def project_context():
    with KedroSession.create(
        package_name="skunkworks_synthetic_data", project_path=Path.cwd()
    ) as session:
        session.run()
    return KedroContext(
        package_name="skunkworks_synthetic_data", project_path=Path.cwd()
    )
