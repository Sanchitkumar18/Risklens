"""Pydantic schemas describing data-validation results.

A ``ValidationReport`` is returned by the validation stage and surfaced in the
ingestion summary / upload API so callers see exactly how many rows were processed,
accepted, and rejected, and why. Bad data is quarantined and reported — never
silently dropped.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """A single validation finding (row-level or dataset-level)."""

    category: str = Field(..., description="Machine-readable issue category.")
    severity: str = Field(..., description="'ERROR' (row rejected) or 'WARNING' (row kept, flagged).")
    message: str = Field(..., description="Human-readable description.")
    ticker: str | None = Field(default=None, description="Affected ticker, if row-level.")
    date: str | None = Field(default=None, description="Affected date (ISO), if row-level.")
    row_index: int | None = Field(default=None, description="Source row index, if row-level.")


class ValidationReport(BaseModel):
    """Aggregate outcome of validating a batch of market-data rows."""

    rows_processed: int = Field(..., description="Total rows examined.")
    rows_accepted: int = Field(..., description="Rows with no ERROR-level issue.")
    rows_rejected: int = Field(..., description="Rows dropped due to at least one ERROR.")
    errors_by_category: dict[str, int] = Field(
        default_factory=dict, description="Count of ERROR issues per category."
    )
    warnings_by_category: dict[str, int] = Field(
        default_factory=dict, description="Count of WARNING issues per category."
    )
    sample_issues: list[ValidationIssue] = Field(
        default_factory=list, description="A capped sample of concrete issues for inspection."
    )

    @property
    def is_valid(self) -> bool:
        """True when no rows were rejected (warnings are allowed)."""
        return self.rows_rejected == 0
