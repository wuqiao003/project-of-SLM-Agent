"""
Metrics for evaluating tool-use model performance.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class ToolUseMetrics:
    """Metrics for tool-use evaluation."""
    
    # Accuracy metrics
    tool_selection_accuracy: float = 0.0  # Correct tool selected
    argument_accuracy: float = 0.0         # Correct arguments provided
    exact_match_accuracy: float = 0.0      # Both tool and arguments correct
    
    # Format metrics
    json_validity_rate: float = 0.0        # Valid JSON output
    schema_compliance_rate: float = 0.0    # Matches expected schema
    
    # Latency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Sample counts
    total_samples: int = 0
    successful_samples: int = 0


def compute_metrics(
    predictions: list[dict],
    references: list[dict],
    latencies: Optional[list[float]] = None,
) -> ToolUseMetrics:
    """
    Compute tool-use metrics from predictions and references.
    
    Args:
        predictions: List of predicted tool calls
        references: List of ground truth tool calls
        latencies: Optional list of latencies in milliseconds
    
    Returns:
        ToolUseMetrics object
    """
    metrics = ToolUseMetrics()
    metrics.total_samples = len(predictions)
    
    if not predictions:
        return metrics
    
    # Counters
    correct_tool = 0
    correct_args = 0
    exact_match = 0
    valid_json = 0
    schema_valid = 0
    
    for pred, ref in zip(predictions, references):
        # Check JSON validity
        if isinstance(pred, dict):
            valid_json += 1
            
            # Check schema compliance
            if "name" in pred and "arguments" in pred:
                schema_valid += 1
                
                # Check tool selection
                if pred.get("name") == ref.get("name"):
                    correct_tool += 1
                    
                    # Check arguments
                    pred_args = pred.get("arguments", {})
                    ref_args = ref.get("arguments", {})
                    
                    if _compare_arguments(pred_args, ref_args):
                        correct_args += 1
                        exact_match += 1
    
    # Calculate rates
    n = metrics.total_samples
    metrics.tool_selection_accuracy = correct_tool / n
    metrics.argument_accuracy = correct_args / n
    metrics.exact_match_accuracy = exact_match / n
    metrics.json_validity_rate = valid_json / n
    metrics.schema_compliance_rate = schema_valid / n
    metrics.successful_samples = exact_match
    metrics.error_rate = 1 - (valid_json / n)
    
    # Latency metrics
    if latencies:
        sorted_latencies = sorted(latencies)
        metrics.avg_latency_ms = sum(latencies) / len(latencies)
        metrics.p50_latency_ms = _percentile(sorted_latencies, 50)
        metrics.p95_latency_ms = _percentile(sorted_latencies, 95)
        metrics.p99_latency_ms = _percentile(sorted_latencies, 99)
    
    return metrics


def _compare_arguments(pred: dict, ref: dict, tolerance: float = 0.01) -> bool:
    """Compare predicted and reference arguments."""
    if set(pred.keys()) != set(ref.keys()):
        return False
    
    for key in ref:
        pred_val = pred.get(key)
        ref_val = ref[key]
        
        # Type mismatch
        if type(pred_val) != type(ref_val):
            # Allow string/number flexibility
            try:
                if isinstance(ref_val, (int, float)):
                    if abs(float(pred_val) - float(ref_val)) > tolerance:
                        return False
                    continue
            except (ValueError, TypeError):
                return False
        
        # Compare values
        if pred_val != ref_val:
            return False
    
    return True


def _percentile(sorted_data: list, p: float) -> float:
    """Calculate percentile from sorted data."""
    if not sorted_data:
        return 0.0
    
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    
    if f == c:
        return sorted_data[f]
    
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def format_metrics_report(metrics: ToolUseMetrics) -> str:
    """Format metrics as a human-readable report."""
    report = """
╔══════════════════════════════════════════════════════════════╗
║                    Tool-Use Evaluation Report                 ║
╠══════════════════════════════════════════════════════════════╣
║ ACCURACY METRICS                                              ║
║   Tool Selection Accuracy:    {tool_acc:>6.1%}                       ║
║   Argument Accuracy:          {arg_acc:>6.1%}                       ║
║   Exact Match Accuracy:       {exact_acc:>6.1%}                       ║
╠══════════════════════════════════════════════════════════════╣
║ FORMAT METRICS                                                ║
║   JSON Validity Rate:         {json_rate:>6.1%}                       ║
║   Schema Compliance Rate:     {schema_rate:>6.1%}                       ║
╠══════════════════════════════════════════════════════════════╣
║ LATENCY METRICS                                               ║
║   Average Latency:            {avg_lat:>6.1f} ms                      ║
║   P50 Latency:                {p50_lat:>6.1f} ms                      ║
║   P95 Latency:                {p95_lat:>6.1f} ms                      ║
║   P99 Latency:                {p99_lat:>6.1f} ms                      ║
╠══════════════════════════════════════════════════════════════╣
║ SUMMARY                                                       ║
║   Total Samples:              {total:>6d}                            ║
║   Successful Samples:         {success:>6d}                            ║
║   Error Rate:                 {error_rate:>6.1%}                       ║
╚══════════════════════════════════════════════════════════════╝
""".format(
        tool_acc=metrics.tool_selection_accuracy,
        arg_acc=metrics.argument_accuracy,
        exact_acc=metrics.exact_match_accuracy,
        json_rate=metrics.json_validity_rate,
        schema_rate=metrics.schema_compliance_rate,
        avg_lat=metrics.avg_latency_ms,
        p50_lat=metrics.p50_latency_ms,
        p95_lat=metrics.p95_latency_ms,
        p99_lat=metrics.p99_latency_ms,
        total=metrics.total_samples,
        success=metrics.successful_samples,
        error_rate=metrics.error_rate,
    )
    
    return report
