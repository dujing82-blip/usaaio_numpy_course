import pytest
from usaaio_check.exercise_tests import evaluate

@pytest.mark.parametrize("id", ['3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '3.10', '3.11', '3.12', '3.13'])
def test_exercise(id,solution_namespaces):
    assert evaluate(solution_namespaces[id],id)
