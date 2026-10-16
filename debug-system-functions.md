# Debug Session: system-functions

- **Status**: [OPEN]
- **Issue**: The Flask system needs correct multi-slice upload/viewing, meaningful RADAR diagnosis explainability, working report generation, and effective settings.
- **Hypotheses**:
  - A: The upload API accepts only one file, so a DICOM series cannot be assembled as one study.
  - B: The current diagnosis path uses a mock/limited result schema rather than RADAR inference and explainability outputs.
  - C: PDF generation fails because the frontend calls the wrong endpoint or the generator receives an incomplete result structure/dependency.
  - D: Settings are only UI placeholders and are not persisted or applied by the backend.

## Evidence
Pending runtime and static verification.

## Fixes
Pending.
