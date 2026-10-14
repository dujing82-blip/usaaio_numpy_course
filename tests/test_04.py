import pytest
from usaaio_check.exercise_tests import evaluate

@pytest.mark.parametrize("id", ['4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9', '4.10', '4.11', '4.12'])
def test_exercise(id,solution_namespaces):
    assert evaluate(solution_namespaces[id],id)
