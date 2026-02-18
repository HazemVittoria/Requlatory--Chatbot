from __future__ import annotations

TOPICS: dict[str, dict[str, list[str]]] = {
    "computerized_systems": {
        "keywords": [
            "annex 11",
            "part 11",
            "computerised",
            "computerized",
            "electronic records",
            "audit trail",
        ],
        "paths": [
            "eu_gmp/",
            "fda/21-cfr-part-11",
            "pic_s/pi 011",
        ],
    },
    "process_validation": {
        "keywords": [
            "validation",
            "process validation",
            "continued process verification",
            "stage 1",
            "stage 2",
            "stage 3",
            "continuous manufacturing",
            "qualification",
            "dq",
            "iq",
            "oq",
            "pq",
            "validation master plan",
            "vmp",
            "annex 15",
            "qualification and validation",
        ],
        "paths": [
            "eu_gmp/",
            "eu_gmp/annex15",
            "fda/",
            "ich/",
        ],
    },
    "oos": {
        "keywords": [
            "oos",
            "out-of-specification",
            "out of specification",
            "retest",
            "retesting",
            "resample",
            "investigation",
        ],
        "paths": [
            "fda/",
            "pic_s/",
            "who/",
        ],
    },
    "sterile": {
        "keywords": [
            "aseptic",
            "annex 1",
            "sterile",
            "environmental monitoring",
        ],
        "paths": [
            "eu_gmp/",
            "fda/",
            "pic_s/",
        ],
    },
    "data_integrity": {
        "keywords": [
            "alcoa",
            "data integrity",
            "audit trail review",
            "data governance",
        ],
        "paths": [
            "fda/",
            "pic_s/",
            "who/",
            "others/mhra",
        ],
    },
    "api_gmp": {
        "keywords": [
            "api",
            "ich q7",
            "drug substance",
        ],
        "paths": [
            "ich/",
            "eu_gmp/",
        ],
    },
}
