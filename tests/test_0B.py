import pytest
from usaaio_check.exercise_tests import evaluate

@pytest.mark.parametrize("id", ['B.1', 'B.2', 'B.3', 'B.4', 'B.5'])
def test_exercise(id,solution_namespaces):
    assert evaluate(solution_namespaces[id],id)
