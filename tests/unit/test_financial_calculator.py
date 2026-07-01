"""Unit tests for financial calculator tools."""

import pytest
from agent.tools.financial_calculator import (
    calculate_cagr, calculate_yoy_growth, calculate_gross_margin,
    calculate_roe, calculate_debt_to_equity, calculate_free_cash_flow,
)


class TestCAGR:
    def test_basic_cagr(self):
        result = calculate_cagr(100, 200, 5)
        assert abs(result["cagr"] - 0.148698) < 0.0001

    def test_cagr_zero_years(self):
        result = calculate_cagr(100, 200, 0)
        assert "error" in result

    def test_cagr_negative_beginning(self):
        result = calculate_cagr(-100, 200, 5)
        assert "error" in result


class TestYoYGrowth:
    def test_positive_growth(self):
        result = calculate_yoy_growth(120, 100)
        assert result["yoy_growth"] == 0.2
        assert result["yoy_growth_pct"] == "20.00%"

    def test_negative_growth(self):
        result = calculate_yoy_growth(80, 100)
        assert result["yoy_growth"] == pytest.approx(-0.2)

    def test_zero_previous(self):
        result = calculate_yoy_growth(100, 0)
        assert "error" in result


class TestGrossMargin:
    def test_apple_like_margin(self):
        result = calculate_gross_margin(383_000, 214_000)
        assert 0.44 < result["gross_margin"] < 0.45

    def test_zero_revenue(self):
        result = calculate_gross_margin(0, 100)
        assert "error" in result


class TestROE:
    def test_basic_roe(self):
        result = calculate_roe(net_income=50, shareholders_equity=200)
        assert result["roe"] == 0.25
        assert result["roe_pct"] == "25.00%"


class TestDebtToEquity:
    def test_high_leverage(self):
        result = calculate_debt_to_equity(300, 100)
        assert result["debt_to_equity"] == 3.0
        assert result["interpretation"] == "high leverage"

    def test_low_leverage(self):
        result = calculate_debt_to_equity(50, 200)
        assert result["interpretation"] == "low leverage"


class TestFreeCashFlow:
    def test_positive_fcf(self):
        result = calculate_free_cash_flow(100, 30)
        assert result["free_cash_flow"] == 70
