import unittest

from .fuzzy_alignment import (
    FloraEntityAligner,
    _event_description_support,
    _identity_context_sufficient,
    _temporal_merge_supported,
)
from .schema import clean_entity_info


class EntityTypeNormalizationTests(unittest.TestCase):
    def test_explicit_vietnamese_name_prefixes_override_inconsistent_types(self):
        organization = clean_entity_info(
            "Nha Bình dân học vụ", "event", "A historical institution"
        )
        event = clean_entity_info(
            "Phong trào Bình dân học vụ", "organization", "A historical campaign"
        )

        self.assertIsNotNone(organization)
        self.assertIsNotNone(event)
        self.assertEqual(organization[1], "ORGANIZATION")
        self.assertEqual(event[1], "EVENT")


class LocalClusterTypeGuardTests(unittest.TestCase):
    def setUp(self):
        self.aligner = FloraEntityAligner({})

    def cluster(self, entity_types):
        nodes = {
            name: [{"entity_type": entity_type, "alternate_names": []}]
            for name, entity_type in entity_types.items()
        }
        clustered, _ = self.aligner._cluster_local_kg(nodes, {})
        return clustered

    def test_conflicting_known_types_are_not_merged(self):
        clustered = self.cluster(
            {
                "BÌNH DÂN HỌC VỤ": "EVENT",
                "NHA BÌNH DÂN HỌC VỤ": "ORGANIZATION",
            }
        )

        self.assertEqual(len(clustered), 2)

    def test_unknown_node_cannot_bridge_different_known_types(self):
        clustered = self.cluster(
            {
                "BÌNH DÂN HỌC VỤ": "EVENT",
                "NHA BÌNH DÂN HỌC VỤ": "UNKNOWN",
                "NHA BÌNH DÂN HỌC VỤ PHONG TRÀO": "ORGANIZATION",
            }
        )

        self.assertEqual(len(clustered), 2)
        cluster_types = [
            {node["entity_type"] for node in node_lists}
            for node_lists in clustered.values()
        ]
        self.assertFalse(
            any({"EVENT", "ORGANIZATION"}.issubset(types) for types in cluster_types)
        )

    def test_same_known_types_still_merge_by_identity(self):
        clustered = self.cluster(
            {
                "BÌNH DÂN HỌC VỤ": "EVENT",
                "NHA BÌNH DÂN HỌC VỤ": "EVENT",
            }
        )

        self.assertEqual(len(clustered), 1)


class OrganizationIdentityContextTests(unittest.TestCase):
    def test_partial_organization_name_requires_shared_neighbors(self):
        self.assertFalse(
            _identity_context_sufficient(
                "Mặt trận Việt Minh",
                "ORGANIZATION",
                "Việt Minh",
                "ORGANIZATION",
                shared_neighbor_count=1,
                min_shared_neighbors=2,
            )
        )
        self.assertTrue(
            _identity_context_sufficient(
                "Mặt trận Việt Minh",
                "ORGANIZATION",
                "Việt Minh",
                "ORGANIZATION",
                shared_neighbor_count=2,
                min_shared_neighbors=2,
            )
        )

    def test_identical_organization_names_do_not_need_extra_context(self):
        self.assertTrue(
            _identity_context_sufficient(
                "Việt Minh",
                "ORGANIZATION",
                " VIỆT MINH ",
                "ORGANIZATION",
                shared_neighbor_count=0,
                min_shared_neighbors=2,
            )
        )


class EventTemporalAliasTests(unittest.TestCase):
    def test_description_support_recognizes_same_event_across_periods(self):
        early_description = (
            "A nationwide literacy campaign launched by the Provisional Government "
            "of the Democratic Republic of Vietnam on September 8, 1945, to eradicate "
            "illiteracy among the entire population."
        )
        later_description = (
            "A mass literacy campaign launched in Vietnam to eliminate illiteracy, "
            "particularly in the South."
        )

        self.assertTrue(
            _event_description_support(early_description, later_description)
        )
        self.assertTrue(
            _temporal_merge_supported(
                "EVENT",
                "EVENT",
                "mismatch",
                early_description,
                later_description,
                identity_similarity=0.9,
                identity_threshold=0.8,
            )
        )

    def test_unrelated_event_descriptions_do_not_override_temporal_conflict(self):
        self.assertFalse(
            _event_description_support(
                "A literacy campaign to eradicate illiteracy in Vietnam.",
                "A military campaign to capture a northern fortress.",
            )
        )
        self.assertFalse(
            _temporal_merge_supported(
                "EVENT",
                "EVENT",
                "mismatch",
                "A literacy campaign to eradicate illiteracy in Vietnam.",
                "A military campaign to capture a northern fortress.",
                identity_similarity=0.9,
                identity_threshold=0.8,
            )
        )

    def test_neighborhood_evidence_alone_cannot_override_time_mismatch(self):
        self.assertFalse(
            _temporal_merge_supported(
                "PERSON",
                "PERSON",
                "mismatch",
                "A historical person",
                "Another historical person",
                identity_similarity=0.9,
                identity_threshold=0.8,
            )
        )

    def test_event_description_cannot_override_time_without_identity_support(self):
        self.assertFalse(
            _temporal_merge_supported(
                "EVENT",
                "EVENT",
                "mismatch",
                "A literacy campaign to eradicate illiteracy in Vietnam.",
                "A literacy campaign to eradicate illiteracy in Vietnam.",
                identity_similarity=0.6,
                identity_threshold=0.8,
            )
        )


if __name__ == "__main__":
    unittest.main()
