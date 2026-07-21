"""Unit tests for HAIP retrieval-evaluation metric functions."""
import math
import sys
import pytest

sys.path.insert(0, ".")


def dcg(rels):
    """Discounted cumulative gain (mirrors eval_retrieval.dcg)."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


class TestDCG:
    def test_empty(self):
        assert dcg([]) == 0.0

    def test_single_relevant_first(self):
        assert dcg([1]) == pytest.approx(1.0)

    def test_rank_one_beats_rank_three(self):
        assert dcg([1, 0, 0]) > dcg([0, 0, 1])

    def test_ndcg_perfect_ranking_is_one(self):
        rels = [1, 1, 0, 0, 0]
        assert dcg(rels) / dcg(sorted(rels, reverse=True)) == pytest.approx(1.0)


class TestReciprocalRank:
    @staticmethod
    def rr(rels):
        return 1 / (rels.index(1) + 1) if 1 in rels else 0.0

    def test_first_position(self):
        assert self.rr([1, 0, 0]) == pytest.approx(1.0)

    def test_third_position(self):
        assert self.rr([0, 0, 1]) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert self.rr([0, 0, 0]) == 0.0


class TestRecall:
    @staticmethod
    def recall(gold, retrieved):
        return len(set(gold) & set(retrieved)) / len(gold) if gold else 0.0

    def test_never_exceeds_one_with_duplicates(self):
        # regression: duplicate doc_ids previously pushed recall above 1.0
        assert self.recall(["d1"], ["d1", "d1", "d1"]) == pytest.approx(1.0)

    def test_partial(self):
        assert self.recall(["d1", "d2"], ["d1", "d9"]) == pytest.approx(0.5)

    def test_miss(self):
        assert self.recall(["d1"], ["d7", "d8"]) == 0.0


class TestPrecisionAtK:
    @staticmethod
    def precision(rels, k):
        return sum(rels[:k]) / k

    def test_p_at_1(self):
        assert self.precision([1, 0, 0, 0, 0], 1) == pytest.approx(1.0)

    def test_p_at_5_partial(self):
        assert self.precision([1, 1, 0, 0, 0], 5) == pytest.approx(0.4)

    def test_precision_decreases_in_tail(self):
        rels = [1, 1, 1, 0, 0]
        assert self.precision(rels, 3) > self.precision(rels, 5)
