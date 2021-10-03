# import pytest
import pycrires


class TestPipline:
    def setup_class(self) -> None:
        self.limit = 1e-8

    @staticmethod
    def test_pipeline() -> None:
        pipeline = pycrires.Pipeline("./")
        assert isinstance(pipeline, pycrires.pipeline.Pipeline)
