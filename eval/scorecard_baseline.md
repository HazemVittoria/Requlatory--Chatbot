# Baseline Scorecard

Generated: 2026-02-18T20:28:36.987137+00:00

## Overall metrics
- total: 50
- strong_correct: 28 (56.0%)
- partial: 21 (42.0%)
- incorrect: 0 (0.0%)
- correct_refusal: 1 (2.0%)
- unknown: 0 (0.0%)
- strong+partial % (target >= 85%): 98.0%
- strong % (target >= 75%): 56.0%
- hallucinations count (target = 0): 0

## By-domain table

| domain | total | strong | partial | incorrect | correct_refusal | strong% | strong+partial% | refusal% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apr | 5 | 1 | 3 | 0 | 1 | 20.0% | 80.0% | 20.0% |
| computerized_systems | 10 | 6 | 4 | 0 | 0 | 60.0% | 100.0% | 0.0% |
| data_integrity | 10 | 5 | 5 | 0 | 0 | 50.0% | 100.0% | 0.0% |
| deviations | 10 | 9 | 1 | 0 | 0 | 90.0% | 100.0% | 0.0% |
| equipment_qualification | 5 | 4 | 1 | 0 | 0 | 80.0% | 100.0% | 0.0% |
| validation | 10 | 3 | 7 | 0 | 0 | 30.0% | 100.0% | 0.0% |

## Worst 10

| id | domain | question | cited_sources | domain_filter_used | empty_scope_retry_used |
|---|---|---|---|---|---|
| ev001 | validation | What is process validation and what are its stages? | Process-Validation--General-Principles-and-Practices.pdf:6 | False | False |
| ev002 | validation | What is required in a validation master plan (VMP)? | 2015-10_annex15.pdf:2;pe-009-17-gmp-guide-xannexes.pdf:220;PI 011-3 Recommendation on Computerised Systems.pdf:13 | False | True |
| ev003 | validation | What documentation is required to support process validation? | Process-Validation--General-Principles-and-Practices.pdf:20 | False | False |
| ev004 | validation | When is revalidation required? | ICH_Q2(R2)_Guideline_2023_1130_ErrorCorrection_2025.pdf:8;guidance-computer-software-assurance-production-quality-system.pdf:10 | False | True |
| ev006 | validation | How should cleaning validation be conducted? | pe-009-17-gmp-guide-xannexes.pdf:228;2015-10_annex15.pdf:14 | False | True |
| ev007 | validation | What is the lifecycle approach to process validation? | Process-Validation--General-Principles-and-Practices.pdf:10 | False | False |
| ev009 | validation | What is required before commercial distribution after validation? | Process-Validation--General-Principles-and-Practices.pdf:13 | False | False |
| ev013 | deviations | What is an out-of-specification (OOS) result and how should it be investigated? | 19287685_L2-OOS.pdf:15 | False | False |
| ev021 | computerized_systems | What controls are required for computerized systems in GMP environments? | vol4-chap1_2013-01_en.pdf:2 | False | False |
| ev026 | computerized_systems | What documentation is required for system validation? | Process-Validation--General-Principles-and-Practices.pdf:20;guidance-computer-software-assurance-production-quality-system.pdf:25 | False | False |
