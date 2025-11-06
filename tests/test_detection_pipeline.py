from detection_pipeline import SignDetectionPipeline
from tests.test_letters_conditions import _hand_letter_a


def test_pipeline_smoothing_requires_consensus():
    pipeline = SignDetectionPipeline(history_size=3, min_consensus=2, enable_ml=False)
    hand = _hand_letter_a()

    first = pipeline.process(hand)
    assert first.letter is None
    assert first.raw_letter == "A"

    second = pipeline.process(hand)
    assert second.letter == "A"
    assert second.source == "rules"


def test_pipeline_clears_history_after_misses():
    pipeline = SignDetectionPipeline(history_size=3, min_consensus=2, max_misses=2, enable_ml=False)
    hand = _hand_letter_a()

    # Seed pipeline with a stable letter
    pipeline.process(hand)
    pipeline.process(hand)

    pipeline.process(None)
    result = pipeline.process(None)

    assert result.letter is None
    assert len(pipeline.history) == 0
