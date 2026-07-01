"""
Ragas evaluation — runs automatically on every query response.
Measures: faithfulness, context_precision, context_recall, answer_relevancy.
Results are logged to MLflow for tracking over time.
"""

from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)


class RagasEvaluator:
    """
    Wraps Ragas evaluation metrics.
    Called after every response — adds ~500ms but gives continuous quality signal.
    Can be disabled in production for latency-sensitive deployments.
    """

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, Any]:
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)

            result = evaluate(dataset, metrics=metrics)
            scores = result.to_pandas().iloc[0].to_dict()

            logger.info(
                "ragas_evaluation_complete",
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
                context_precision=scores.get("context_precision"),
            )
            return {
                "faithfulness": scores.get("faithfulness"),
                "context_precision": scores.get("context_precision"),
                "context_recall": scores.get("context_recall"),
                "answer_relevancy": scores.get("answer_relevancy"),
            }

        except ImportError:
            logger.warning("ragas_not_installed_skipping_evaluation")
            return {}
        except Exception as exc:
            logger.warning("ragas_evaluation_error", error=str(exc))
            return {}
