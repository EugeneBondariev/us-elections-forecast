from src.constants import GENERIC_BALLOT, PRES_PARTY, REDISTRICTING_YEARS

ELECTION_YEARS = set(range(1976, 2025, 2))


class TestConstants:
    def test_generic_ballot_covers_all_election_years(self):
        assert ELECTION_YEARS.issubset(GENERIC_BALLOT.keys())

    def test_generic_ballot_2026_dem_favorable(self):
        assert GENERIC_BALLOT[2026] > 0

    def test_pres_party_covers_all_election_years(self):
        assert ELECTION_YEARS.issubset(PRES_PARTY.keys())

    def test_redistricting_years(self):
        assert REDISTRICTING_YEARS == {1982, 1992, 2002, 2012, 2022}
