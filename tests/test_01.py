import pytest
from usaaio_check.exercise_tests import evaluate

@pytest.mark.parametrize("id", ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '1.10', '1.11', '1.12', '1.13', '1.14', '1.15'])
def test_exercise(id,solution_namespaces):
    assert evaluate(solution_namespaces[id],id)
