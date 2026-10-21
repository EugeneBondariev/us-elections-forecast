from src.constants import GENERIC_BALLOT, PRES_PARTY, REDISTRICTING_YEARS, STATE_FIPS_TO_PO

_ALL_STATE_PO = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}

ELECTION_YEARS = set(range(1976, 2025, 2))


class TestStateFipsToPoMapping:
    def test_exactly_50_states(self) -> None:
        assert len(STATE_FIPS_TO_PO) == 50

    def test_all_50_postal_codes_present(self) -> None:
        assert set(STATE_FIPS_TO_PO.values()) == _ALL_STATE_PO

    def test_no_duplicate_postal_codes(self) -> None:
        values = list(STATE_FIPS_TO_PO.values())
        assert len(values) == len(set(values))

    def test_fips_are_zero_padded_two_digit_strings(self) -> None:
        for fips in STATE_FIPS_TO_PO:
            assert len(fips) == 2 and fips.isdigit(), f"Bad FIPS: {fips!r}"

    def test_known_spot_checks(self) -> None:
        spot = {"06": "CA", "48": "TX", "36": "NY", "12": "FL", "17": "IL"}
        for fips, po in spot.items():
            assert STATE_FIPS_TO_PO[fips] == po, (
                f"FIPS {fips}: expected {po}, got {STATE_FIPS_TO_PO[fips]}"
            )

    def test_territories_excluded(self) -> None:
        # DC=11, PR=72, VI=78, GU=66, AS=60, MP=69
        for fips in ("11", "72", "78", "66", "60", "69"):
            assert fips not in STATE_FIPS_TO_PO, f"Territory FIPS {fips} should not be in mapping"


class TestConstants:
    def test_generic_ballot_covers_all_election_years(self):
        assert ELECTION_YEARS.issubset(GENERIC_BALLOT.keys())

    def test_generic_ballot_2026_dem_favorable(self):
        assert GENERIC_BALLOT[2026] > 0

    def test_pres_party_covers_all_election_years(self):
        assert ELECTION_YEARS.issubset(PRES_PARTY.keys())

    def test_redistricting_years(self):
        assert REDISTRICTING_YEARS == {1982, 1992, 2002, 2012, 2022}
