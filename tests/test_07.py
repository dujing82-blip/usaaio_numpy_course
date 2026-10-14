import pytest
from usaaio_check.exercise_tests import evaluate

@pytest.mark.parametrize("id", ['7.1', '7.2', '7.3', '7.4', '7.5', '7.6'])
def test_exercise(id,solution_namespaces):
    assert evaluate(solution_namespaces[id],id)
