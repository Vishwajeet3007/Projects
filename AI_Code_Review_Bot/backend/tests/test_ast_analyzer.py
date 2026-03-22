"""Tests for static analysis helpers."""

from app.analysis.ast_analyzer import StaticAnalysisService



def test_detects_mutable_defaults_and_eval_usage() -> None:
    service = StaticAnalysisService()
    result = service.analyze(
        [
            {
                "path": "sample.py",
                "language": "python",
                "content": "def bad(x=[]):\n    return eval('1+1')\n",
            }
        ]
    )

    issues = result["python"]["issues"]
    titles = {issue["title"] for issue in issues}
    assert "Mutable default argument" in titles
    assert "Use of eval" in titles
