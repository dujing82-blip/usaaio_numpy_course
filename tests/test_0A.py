import pytest
from usaaio_check.exercise_tests import evaluate

@pytest.mark.parametrize("id", ['A.1', 'A.2', 'A.3', 'A.4', 'A.5', 'A.6'])
def test_exercise(id,solution_namespaces):
    assert evaluate(solution_namespaces[id],id)
