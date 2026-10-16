# Debug Session: verify-dicom-ohif

- **Status**: [OPEN]
- **Issue**: Validate whether `data/test_ct_dicom_series` can be opened by the configured OHIF viewer.
- **Debug Server**: pending startup
- **Log File**: `.dbg/trae-debug-log-verify-dicom-ohif.ndjson`

## Reproduction Steps
1. Check the DICOM sequence files and Study/Series/Instance identifiers.
2. Check if Flask (5000) and OHIF (3000) are responding.
3. Query the Flask DICOMWeb endpoints and OHIF viewer route using the test study UID.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | The generated sequence is malformed or has inconsistent Study/Series UIDs | Medium | Low | Pending |
| B | Flask DICOMWeb endpoints cannot discover/serve this folder | High | Low | Pending |
| C | OHIF route or data-source config does not match the running app | High | Low | Pending |
| D | One of the services is not running or unavailable | Medium | Low | Pending |

## Log Evidence
- Flask homepage: HTTP 200.
- OHIF homepage: HTTP 200.
- OHIF config contains `defaultDataSourceName: 'local5000'` and `http://localhost:5000/dicomweb`.
- Querying Flask with the generated sequence UID from `data/test_ct_dicom_series/series_info.txt` returns: `study=0 series=0 instances=0`.
- The active OHIF server returned HTTP 404 for `/viewer/local5000?StudyInstanceUIDs=<sequence UID>`.
- Previous route tests also showed the Flask DICOMWeb service returns studies created by its upload conversion, but the generated folder's original Study UID is not registered under Flask's DICOMWeb storage directory.

## Hypothesis Results
| ID | Result | Evidence |
|----|--------|----------|
| A | Rejected | The folder contains 128 DICOM files with shared Study and Series UIDs (verified in the previous sequence-generation check). |
| B | Confirmed | The existing Flask DICOMWeb source returns zero studies/series/instances for this folder's Study UID. The files are not in Flask's served `uploads/ohif/<study_uid>` location. |
| C | Confirmed | Active OHIF dev server returns 404 for the assumed explicit data-source viewer route. The running app's route is not serving that route as requested. |
| D | Rejected | Both service homepages responded HTTP 200. |

## Verification Conclusion
The DICOM sequence is structurally valid, but it is **not currently importable by this OHIF setup**. Flask does not index the directory as a DICOMWeb study, and the active OHIF route request returns 404. This is an integration/configuration failure, not a malformed DICOM sequence.
