"""Read-only data tools for the advisor research loop.

Only historical prices are retrieved. These tools intentionally do not offer
broker connectivity, trade execution, or automated investment allocation.
"""

from datetime import datetime
import math
import re
from typing import Any

from .engine import Tool

_SYMBOL = re.compile(r"^[A-Z0-9.^=\-]{1,20}$")
_PERIODS = frozenset({"1mo", "3mo", "6mo", "1y", "2y"})


def _valid_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError("invalid ticker symbol")
    return symbol


def _history(symbol: str, period: str):
    if period not in _PERIODS:
        raise ValueError("unsupported period")
    symbol = _valid_symbol(symbol)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to use market tools") from exc
    history = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history:
        raise ValueError("no adjusted close prices returned")
    close = history["Close"].dropna()
    if close.empty:
        raise ValueError("no valid adjusted close prices")
    return symbol, close


def _as_of(index: Any) -> str:
    last = index[-1]
    if isinstance(last, datetime):
        return last.date().isoformat()
    if hasattr(last, "date"):
        return last.date().isoformat()
    return str(last)[:10]


def price_snapshot(symbol: str) -> dict[str, Any]:
    """Return the last adjusted closing price and date (not a live quote)."""
    symbol, close = _history(symbol, "1mo")
    value = float(close.iloc[-1])
    if not math.isfinite(value):
        raise ValueError("invalid price")
    return {
        "symbol": symbol,
        "as_of": _as_of(close.index),
        "adjusted_close": round(value, 4),
        "note": "Historical adjusted close; not a live or executable quote.",
    }


def volatility_snapshot(symbol: str, period: str) -> dict[str, Any]:
    """Annualized sample standard deviation of daily close-to-close returns.

    Assumes 252 trading sessions per year. This is realized backward-looking
    volatility, not a prediction and not a comprehensive risk score.
    """
    symbol, close = _history(symbol, period)
    returns = close.pct_change().dropna()
    if len(returns) < 2:
        raise ValueError("at least three valid closes are required")
    value = float(returns.std(ddof=1)) * math.sqrt(252)
    if not math.isfinite(value):
        raise ValueError("invalid volatility")
    return {
        "symbol": symbol,
        "as_of": _as_of(close.index),
        "period": period,
        "observations": int(len(returns)),
        "annualized_realized_volatility": round(value, 6),
        "method": "std(sample daily adjusted-close returns) * sqrt(252)",
        "note": "Historical estimate; does not predict future risk or return.",
    }


def market_tools() -> tuple[Tool, ...]:
    return (
        Tool(
            name="price_snapshot",
            description="Get a stock's last available historical adjusted close and as-of date.",
            handler=price_snapshot,
            required_arguments=frozenset({"symbol"}),
        ),
        Tool(
            name="volatility_snapshot",
            description=(
                "Calculate backward-looking annualized daily-return volatility. "
                "Allowed periods: 1mo, 3mo, 6mo, 1y, 2y."
            ),
            handler=volatility_snapshot,
            required_arguments=frozenset({"symbol", "period"}),
        ),
    )
