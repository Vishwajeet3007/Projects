"""Tests for schema normalization."""

from app.schemas.review import ReviewReport



def test_review_report_normalizes_object_optimizations() -> None:
    report = ReviewReport.model_validate(
        {
            "bugs": [],
            "security_issues": [],
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "optimizations": [
                {
                    "type": "performance",
                    "description": "Avoid repeated work inside loops.",
                    "recommendation": "Cache invariant computations.",
                },
                {
                    "type": "maintainability",
                    "issue": "analysis and IDE support",
                    "suggestion": "Add type hints to public functions.",
                },
            ],
            "refactored_code": "print('ok')",
            "documentation": "doc",
            "unit_tests": "def test_ok(): pass",
            "code_quality_score": "8.5",
            "final_summary": "Looks solid overall.",
        }
    )

    assert len(report.optimizations) == 2
    assert "Type: performance" in report.optimizations[0]
    assert "Recommendation: Cache invariant computations." in report.optimizations[0]
    assert report.code_quality_score == 8.5
