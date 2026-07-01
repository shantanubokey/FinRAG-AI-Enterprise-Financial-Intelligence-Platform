"""
Financial calculator tools.
The LLM calls these instead of doing arithmetic in natural language,
which is notoriously unreliable. Tool calling = reliable computation.
"""

from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)


def calculate_cagr(beginning_value: float, ending_value: float, years: float) -> dict[str, Any]:
    """
    Compound Annual Growth Rate.
    CAGR = (ending / beginning) ^ (1/years) - 1
    """
    if beginning_value <= 0 or years <= 0:
        return {"error": "beginning_value and years must be positive"}
    cagr = (ending_value / beginning_value) ** (1 / years) - 1
    return {
        "cagr": round(cagr, 6),
        "cagr_pct": f"{cagr * 100:.2f}%",
        "beginning_value": beginning_value,
        "ending_value": ending_value,
        "years": years,
    }


def calculate_yoy_growth(current: float, previous: float) -> dict[str, Any]:
    """Year-over-Year growth rate."""
    if previous == 0:
        return {"error": "previous value cannot be zero"}
    growth = (current - previous) / abs(previous)
    return {
        "yoy_growth": round(growth, 6),
        "yoy_growth_pct": f"{growth * 100:.2f}%",
        "current": current,
        "previous": previous,
        "absolute_change": current - previous,
    }


def calculate_ebitda(
    net_income: float,
    interest: float,
    taxes: float,
    depreciation: float,
    amortization: float,
) -> dict[str, Any]:
    """EBITDA = Net Income + Interest + Taxes + D&A"""
    ebitda = net_income + interest + taxes + depreciation + amortization
    return {
        "ebitda": round(ebitda, 2),
        "components": {
            "net_income": net_income,
            "interest": interest,
            "taxes": taxes,
            "depreciation": depreciation,
            "amortization": amortization,
        },
    }


def calculate_roe(net_income: float, shareholders_equity: float) -> dict[str, Any]:
    """Return on Equity = Net Income / Shareholders' Equity"""
    if shareholders_equity == 0:
        return {"error": "shareholders_equity cannot be zero"}
    roe = net_income / shareholders_equity
    return {"roe": round(roe, 6), "roe_pct": f"{roe * 100:.2f}%"}


def calculate_roa(net_income: float, total_assets: float) -> dict[str, Any]:
    """Return on Assets = Net Income / Total Assets"""
    if total_assets == 0:
        return {"error": "total_assets cannot be zero"}
    roa = net_income / total_assets
    return {"roa": round(roa, 6), "roa_pct": f"{roa * 100:.2f}%"}


def calculate_debt_to_equity(total_debt: float, shareholders_equity: float) -> dict[str, Any]:
    """D/E Ratio = Total Debt / Shareholders' Equity"""
    if shareholders_equity == 0:
        return {"error": "shareholders_equity cannot be zero"}
    de = total_debt / shareholders_equity
    return {
        "debt_to_equity": round(de, 4),
        "interpretation": "high leverage" if de > 2 else "moderate leverage" if de > 1 else "low leverage",
    }


def calculate_gross_margin(revenue: float, cogs: float) -> dict[str, Any]:
    """Gross Margin = (Revenue - COGS) / Revenue"""
    if revenue == 0:
        return {"error": "revenue cannot be zero"}
    gross_profit = revenue - cogs
    margin = gross_profit / revenue
    return {
        "gross_margin": round(margin, 6),
        "gross_margin_pct": f"{margin * 100:.2f}%",
        "gross_profit": round(gross_profit, 2),
    }


def calculate_operating_margin(operating_income: float, revenue: float) -> dict[str, Any]:
    """Operating Margin = Operating Income / Revenue"""
    if revenue == 0:
        return {"error": "revenue cannot be zero"}
    margin = operating_income / revenue
    return {
        "operating_margin": round(margin, 6),
        "operating_margin_pct": f"{margin * 100:.2f}%",
    }


def calculate_free_cash_flow(
    operating_cash_flow: float, capital_expenditures: float
) -> dict[str, Any]:
    """FCF = Operating Cash Flow - CapEx"""
    fcf = operating_cash_flow - capital_expenditures
    return {
        "free_cash_flow": round(fcf, 2),
        "operating_cash_flow": operating_cash_flow,
        "capital_expenditures": capital_expenditures,
        "fcf_yield_note": "Divide by market cap for FCF yield",
    }


def calculate_eps_growth(
    current_eps: float, previous_eps: float
) -> dict[str, Any]:
    """EPS Growth Rate."""
    return calculate_yoy_growth(current_eps, previous_eps)


# Registry for the agent's tool dispatcher
FINANCIAL_TOOLS = {
    "calculate_cagr": calculate_cagr,
    "calculate_yoy_growth": calculate_yoy_growth,
    "calculate_ebitda": calculate_ebitda,
    "calculate_roe": calculate_roe,
    "calculate_roa": calculate_roa,
    "calculate_debt_to_equity": calculate_debt_to_equity,
    "calculate_gross_margin": calculate_gross_margin,
    "calculate_operating_margin": calculate_operating_margin,
    "calculate_free_cash_flow": calculate_free_cash_flow,
    "calculate_eps_growth": calculate_eps_growth,
}
