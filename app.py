import warnings
warnings.filterwarnings('ignore')

from datetime import date, timedelta
from math import sqrt
import concurrent.futures as cf

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# ALPHA — DYNAMIC PORTFOLIO INTELLIGENCE & STRATEGY
# ============================================================

st.set_page_config(
    page_title="ALPHA — Dynamic Portfolio Intelligence",
    page_icon="α",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

INVESTMENT_DATE = pd.Timestamp("2026-09-01")
TARGET_DATE = pd.Timestamp("2029-09-01")

TOTAL_CAPITAL = 1_00_00_000

DEFAULT_BENCHMARK = "^NSEI"

NETWORK_TIMEOUT = 8

BROKERAGE_RATE = 0.0025


# ============================================================
# DEFAULT 15-STOCK UNIVERSE
# ============================================================

DEFAULT_COMPANIES = [
    "BAJAJ-AUTO.NS",
    "MARUTI.NS",
    "M&M.NS",
    "HAL.NS",
    "POLYCAB.NS",
    "SIEMENS.NS",
    "PIDILITIND.NS",
    "SOLARINDS.NS",
    "PIIND.NS",
    "BRITANNIA.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "AXISBANK.NS",
    "BSE.NS",
    "ANGELONE.NS"
]


# ============================================================
# COMPANY METADATA
# ============================================================

META = {

    "BAJAJ-AUTO.NS":
        ("Bajaj Auto Ltd", "Automobiles"),

    "MARUTI.NS":
        ("Maruti Suzuki India Ltd", "Automobiles"),

    "M&M.NS":
        ("Mahindra & Mahindra Ltd", "Automobiles"),

    "HAL.NS":
        ("Hindustan Aeronautics Ltd", "Defense / Capital Goods"),

    "POLYCAB.NS":
        ("Polycab India Ltd", "Capital Goods / Manufacturing"),

    "SIEMENS.NS":
        ("Siemens Ltd", "Capital Goods"),

    "PIDILITIND.NS":
        ("Pidilite Industries Ltd", "Chemicals"),

    "SOLARINDS.NS":
        ("Solar Industries India Ltd", "Chemicals / Defense"),

    "PIIND.NS":
        ("PI Industries Ltd", "Chemicals"),

    "BRITANNIA.NS":
        ("Britannia Industries Ltd", "FMCG"),

    "HINDUNILVR.NS":
        ("Hindustan Unilever Ltd", "FMCG"),

    "ITC.NS":
        ("ITC Ltd", "FMCG"),

    "AXISBANK.NS":
        ("Axis Bank Ltd", "Financials"),

    "BSE.NS":
        ("BSE Ltd", "Capital Markets"),

    "ANGELONE.NS":
        ("Angel One Ltd", "Financial Services"),
}


# ============================================================
# PEER GROUPS
# ============================================================
#
# These peer groups are used for the relative P/E calculation:
#
# Relative P/E =
# Company P/E / Median P/E of Peer Group
#
# ============================================================

PEER_GROUP = {

    "BAJAJ-AUTO.NS": "Automobiles",
    "MARUTI.NS": "Automobiles",
    "M&M.NS": "Automobiles",

    "HAL.NS": "Capital Goods & Defence",
    "POLYCAB.NS": "Capital Goods & Defence",
    "SIEMENS.NS": "Capital Goods & Defence",

    "PIDILITIND.NS": "Chemicals",
    "SOLARINDS.NS": "Chemicals",
    "PIIND.NS": "Chemicals",

    "BRITANNIA.NS": "FMCG",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",

    "AXISBANK.NS": "Financials",
    "BSE.NS": "Financials",
    "ANGELONE.NS": "Financials",
}


# ============================================================
# INVESTOR PERSONAS
# ============================================================

PERSONAS = {

    "Conservative": {

        "valuation": 0.40,
        "risk": 0.40,
        "momentum": 0.10,
        "quality": 0.10,

        "max_holdings": 8,
        "max_weight": 0.18,
    },

    "Balanced": {

        "valuation": 0.30,
        "risk": 0.25,
        "momentum": 0.25,
        "quality": 0.20,

        "max_holdings": 10,
        "max_weight": 0.15,
    },

    "Aggressive": {

        "valuation": 0.25,
        "risk": 0.15,
        "momentum": 0.40,
        "quality": 0.20,

        "max_holdings": 12,
        "max_weight": 0.13,
    },
}


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault(
    "companies",
    DEFAULT_COMPANIES.copy()
)

st.session_state.setdefault(
    "mode",
    "live"
)

st.session_state.setdefault(
    "report_date",
    None
)

st.session_state.setdefault(
    "benchmark",
    DEFAULT_BENCHMARK
)

st.session_state.setdefault(
    "persona",
    "Balanced"
)

st.session_state.setdefault(
    "watchlist",
    []
)

st.session_state.setdefault(
    "target_prices",
    {}
)

st.session_state.setdefault(
    "portfolio_name",
    "ALPHA Portfolio"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def inr(x):

    if x is None or pd.isna(x):
        return "-"

    x = float(x)

    negative = x < 0

    x = abs(x)

    s = f"{x:,.2f}"

    a, b = s.split(".")

    if len(a) <= 3:

        grouped = a

    else:

        last3 = a[-3:]
        rest = a[:-3]

        parts = []

        while len(rest) > 2:

            parts.insert(0, rest[-2:])
            rest = rest[:-2]

        if rest:
            parts.insert(0, rest)

        grouped = ",".join(parts) + "," + last3

    result = "₹" + grouped + "." + b

    if negative:
        result = "-" + result

    return result


def pct(x):

    if x is None or pd.isna(x):
        return "-"

    return f"{float(x):+.2f}%"


def name(ticker):

    return META.get(
        ticker,
        (ticker, "Other")
    )[0]


def sector(ticker):

    return META.get(
        ticker,
        (ticker, "Other")
    )[1]


def peer(ticker):

    return PEER_GROUP.get(
        ticker,
        sector(ticker)
    )


# ============================================================
# NETWORK TIMEOUT
# ============================================================

def bounded(fn, timeout=NETWORK_TIMEOUT):

    try:

        with cf.ThreadPoolExecutor(max_workers=1) as executor:

            return executor.submit(fn).result(
                timeout=timeout
            )

    except Exception:

        return None


# ============================================================
# YAHOO FINANCE HISTORY
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def history(ticker):

    def call():

        h = yf.Ticker(ticker).history(

            start=(
                INVESTMENT_DATE -
                timedelta(days=400)
            ).strftime("%Y-%m-%d"),

            end=(
                pd.Timestamp.today() +
                timedelta(days=1)
            ).strftime("%Y-%m-%d"),

            auto_adjust=False,

            timeout=NETWORK_TIMEOUT
        )

        if isinstance(h, pd.DataFrame):

            return h

        return pd.DataFrame()

    result = bounded(call)

    if isinstance(result, pd.DataFrame):

        return result

    return pd.DataFrame()


# ============================================================
# YAHOO FINANCE FUNDAMENTALS
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def info(ticker):

    result = bounded(
        lambda: yf.Ticker(ticker).info,
        6
    )

    if isinstance(result, dict):

        return result

    return {}


# ============================================================
# BENCHMARK DATA
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def benchmark_history(ticker):

    return history(ticker)


# ============================================================
# CLEAN PRICE SERIES
# ============================================================

def clean_series(
    ticker,
    column="Close"
):

    h = history(ticker)

    if h.empty:

        return pd.Series(dtype=float)

    if column not in h:

        return pd.Series(dtype=float)

    s = h[column].dropna()

    s.index = pd.DatetimeIndex(
        pd.to_datetime(s.index)
    ).tz_localize(None)

    return s


# ============================================================
# PRICE HELPERS
# ============================================================

def price_on_or_before(
    ticker,
    when
):

    s = clean_series(ticker)

    if s.empty:

        return np.nan

    s = s[
        s.index <= pd.Timestamp(when)
    ]

    if len(s) == 0:

        return np.nan

    return float(s.iloc[-1])


def entry_price(ticker):

    s = clean_series(ticker)

    s = s[
        s.index >= INVESTMENT_DATE
    ]

    if len(s) == 0:

        return np.nan

    return float(s.iloc[0])


def latest_price(ticker):

    s = clean_series(ticker)

    if len(s) == 0:

        return np.nan

    return float(s.iloc[-1])


def selected_price(
    ticker,
    selected_date
):

    if selected_date is None:

        return latest_price(ticker)

    return price_on_or_before(
        ticker,
        selected_date
    )


# ============================================================
# WEEKLY REPORT DATES
# ============================================================

def report_dates():

    s = clean_series(
        st.session_state.benchmark
    )

    s = s[
        s.index >= INVESTMENT_DATE
    ]

    if s.empty:

        return []

    grouped = pd.Series(
        s.index,
        index=s.index
    ).groupby(
        s.index.to_period("W-FRI")
    ).max()

    return list(grouped.values)


# ============================================================
# BETA CALCULATION
# ============================================================

def beta_at(
    ticker,
    asof=None,
    window=252
):

    stock = clean_series(ticker)

    benchmark = clean_series(
        st.session_state.benchmark
    )

    if asof is not None:

        stock = stock[
            stock.index <= pd.Timestamp(asof)
        ]

        benchmark = benchmark[
            benchmark.index <= pd.Timestamp(asof)
        ]

    stock_returns = stock.pct_change()

    benchmark_returns = benchmark.pct_change()

    combined = pd.concat(
        [
            stock_returns,
            benchmark_returns
        ],
        axis=1,
        join="inner"
    ).dropna()

    combined = combined.tail(window)

    if len(combined) < 30:

        return np.nan

    benchmark_variance = (
        combined.iloc[:, 1].var()
    )

    if benchmark_variance == 0:

        return np.nan

    return float(
        combined.iloc[:, 0].cov(
            combined.iloc[:, 1]
        ) /
        benchmark_variance
    )


# ============================================================
# MOMENTUM
# ============================================================

def return_at(
    ticker,
    asof=None,
    days=126
):

    s = clean_series(ticker)

    if asof is not None:

        s = s[
            s.index <= pd.Timestamp(asof)
        ]

    if len(s) <= days:

        return np.nan

    return float(
        s.iloc[-1] /
        s.iloc[-days-1] -
        1
    )


# ============================================================
# PEER-RELATIVE P/E
# ============================================================

def valuation_data(companies):

    raw = []

    for ticker in companies:

        inf = info(ticker)

        pe = inf.get(
            "trailingPE"
        )

        try:

            pe = (
                float(pe)
                if pe is not None
                else np.nan
            )

        except Exception:

            pe = np.nan

        raw.append(
            (
                ticker,
                pe
            )
        )

    rows = []

    for ticker, pe in raw:

        peer_values = [

            value

            for other_ticker, value
            in raw

            if (
                other_ticker != ticker
                and
                peer(other_ticker) == peer(ticker)
                and
                pd.notna(value)
                and
                0 < value < 200
            )

        ]

        if peer_values:

            peer_median = float(
                np.median(peer_values)
            )

        else:

            peer_median = np.nan

        if (
            pd.notna(pe)
            and
            pd.notna(peer_median)
            and
            peer_median > 0
        ):

            relative_pe = (
                pe / peer_median
            )

        else:

            relative_pe = np.nan

        rows.append({

            "Ticker": ticker,

            "P/E": pe,

            "Peer P/E": peer_median,

            "Relative P/E":
                relative_pe
        })

    return pd.DataFrame(rows)


# ============================================================
# NORMALIZATION
# ============================================================

def minmax(
    series,
    higher_is_better=True
):

    s = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = s.dropna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    if len(valid) == 0:

        return result

    low = valid.min()
    high = valid.max()

    if high == low:

        result.loc[
            valid.index
        ] = 0.5

    elif higher_is_better:

        result.loc[
            valid.index
        ] = (
            valid - low
        ) / (
            high - low
        )

    else:

        result.loc[
            valid.index
        ] = (
            high - valid
        ) / (
            high - low
        )

    return result


# ============================================================
# ALPHA STRATEGY ENGINE
# ============================================================

def strategy_snapshot(
    companies,
    persona="Balanced",
    asof=None
):

    config = PERSONAS[persona]

    valuation = valuation_data(
        companies
    )

    rows = []

    for _, row in valuation.iterrows():

        ticker = row["Ticker"]

        beta = beta_at(
            ticker,
            asof
        )

        momentum = return_at(
            ticker,
            asof,
            126
        )

        inf = info(ticker)

        roe = inf.get(
            "returnOnEquity"
        )

        try:

            roe = (
                float(roe)
                if roe is not None
                else np.nan
            )

        except Exception:

            roe = np.nan

        rows.append({

            "Ticker":
                ticker,

            "Company":
                name(ticker),

            "Sector":
                sector(ticker),

            "Peer Group":
                peer(ticker),

            "P/E":
                row["P/E"],

            "Peer P/E":
                row["Peer P/E"],

            "Relative P/E":
                row["Relative P/E"],

            "Beta":
                beta,

            "6M Return":
                momentum,

            "ROE":
                roe,
        })

    data = pd.DataFrame(rows)

    if data.empty:

        return data

    # --------------------------------------------------------
    # FACTOR SCORES
    # --------------------------------------------------------

    data["Valuation Score"] = minmax(
        data["Relative P/E"],
        higher_is_better=False
    )

    data["Risk Score"] = minmax(
        data["Beta"],
        higher_is_better=False
    )

    data["Momentum Score"] = minmax(
        data["6M Return"],
        higher_is_better=True
    )

    data["Quality Score"] = minmax(
        data["ROE"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # PERSONA WEIGHTS
    # --------------------------------------------------------

    data["Strategy Score"] = (

        data["Valuation Score"].fillna(0.5)
        * config["valuation"]

        +

        data["Risk Score"].fillna(0.5)
        * config["risk"]

        +

        data["Momentum Score"].fillna(0.5)
        * config["momentum"]

        +

        data["Quality Score"].fillna(0.5)
        * config["quality"]

    ) * 100

    # --------------------------------------------------------
    # RANK STOCKS
    # --------------------------------------------------------

    data = data.sort_values(
        "Strategy Score",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SELECT STOCKS
    # --------------------------------------------------------

    selected = data.head(
        min(
            config["max_holdings"],
            len(data)
        )
    ).copy()

    # --------------------------------------------------------
    # SCORE-PROPORTIONAL WEIGHTS
    # --------------------------------------------------------

    positive_scores = np.maximum(
        selected[
            "Strategy Score"
        ].fillna(0).values,
        0.01
    )

    weights = (
        positive_scores /
        positive_scores.sum()
    )

    max_weight = config[
        "max_weight"
    ]

    # Apply position cap
    for _ in range(20):

        over_limit = (
            weights > max_weight
        )

        if not over_limit.any():

            break

        excess = (
            weights[over_limit]
            - max_weight
        ).sum()

        weights[
            over_limit
        ] = max_weight

        under_limit = ~over_limit

        if under_limit.any():

            weights[
                under_limit
            ] += (
                excess *
                weights[under_limit] /
                weights[under_limit].sum()
            )

        else:

            break

    weights = (
        weights /
        weights.sum()
    )

    selected["New Weight"] = weights

    data["Selected"] = (
        data["Ticker"].isin(
            selected["Ticker"]
        )
    )

    # --------------------------------------------------------
    # MERGE WEIGHTS
    # --------------------------------------------------------

    data = data.merge(

        selected[
            [
                "Ticker",
                "New Weight"
            ]
        ],

        on="Ticker",

        how="left"
    )

    data["New Weight"] = (
        data["New Weight"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    data["Action"] = np.where(

        data["Selected"],

        "SELECT / HOLD",

        "EXCLUDE"
    )

    # --------------------------------------------------------
    # EXPLAINABLE REASON
    # --------------------------------------------------------

    def reason(row):

        relative_pe = row[
            "Relative P/E"
        ]

        beta = row[
            "Beta"
        ]

        if (
            pd.notna(relative_pe)
            and
            pd.notna(beta)
        ):

            return (
                f"Relative P/E "
                f"{relative_pe:.2f}x; "
                f"Beta {beta:.2f}"
            )

        return (
            "Insufficient "
            "peer/factor data"
        )

    data["Reason"] = data.apply(
        reason,
        axis=1
    )

    return data


# ============================================================
# WEEKLY STRATEGY BACKTEST
# ============================================================

def weekly_strategy_history(
    companies,
    persona
):

    dates = report_dates()

    output = []

    for i, current_date in enumerate(
        dates
    ):

        if i >= len(dates) - 1:

            break

        next_date = dates[i + 1]

        snapshot = strategy_snapshot(
            companies,
            persona,
            current_date
        )

        if snapshot.empty:

            continue

        selected = snapshot[
            snapshot["Selected"]
        ]

        weekly_return = 0

        valid = False

        for _, row in selected.iterrows():

            ticker = row["Ticker"]

            price_start = (
                price_on_or_before(
                    ticker,
                    current_date
                )
            )

            price_end = (
                price_on_or_before(
                    ticker,
                    next_date
                )
            )

            if (
                pd.notna(price_start)
                and
                pd.notna(price_end)
            ):

                stock_return = (
                    price_end /
                    price_start -
                    1
                )

                weekly_return += (
                    stock_return *
                    row["New Weight"]
                )

                valid = True

        output.append({

            "Date":
                current_date,

            "Return":
                weekly_return
                if valid
                else np.nan,

            "Holdings":
                len(selected)
        })

    return pd.DataFrame(
        output
    )


# ============================================================
# BASIC PORTFOLIO ENGINE
# ============================================================

def equal_portfolio(
    companies,
    asof=None
):

    number = len(companies)

    if number == 0:

        return pd.DataFrame()

    rows = []

    for ticker in companies:

        entry = entry_price(
            ticker
        )

        price = selected_price(
            ticker,
            asof
        )

        allocation = (
            TOTAL_CAPITAL /
            number
        )

        if (
            pd.notna(entry)
            and
            entry > 0
        ):

            quantity = int(
                allocation /
                entry
            )

        else:

            quantity = 0

        cash = (
            allocation -
            quantity * entry
            if pd.notna(entry)
            else allocation
        )

        if pd.notna(price):

            value = (
                quantity * price +
                cash
            )

        else:

            value = cash

        rows.append({

            "Ticker":
                ticker,

            "Company":
                name(ticker),

            "Sector":
                sector(ticker),

            "Allocated":
                allocation,

            "Entry Price":
                entry,

            "Price":
                price,

            "Qty":
                quantity,

            "Cash":
                cash,

            "Value":
                value,

            "P/L":
                value - allocation
        })

    result = pd.DataFrame(
        rows
    )

    if len(result):

        result["Weight"] = (
            result["Value"] /
            result["Value"].sum()
        )

    return result


# ============================================================
# PORTFOLIO HISTORY
# ============================================================

def portfolio_series(
    companies,
    asof=None
):

    benchmark = clean_series(
        st.session_state.benchmark
    )

    if asof is not None:

        benchmark = benchmark[
            benchmark.index <=
            pd.Timestamp(asof)
        ]

    benchmark = benchmark[
        benchmark.index >=
        INVESTMENT_DATE
    ]

    if benchmark.empty:

        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float)
        )

    index = benchmark.index

    portfolio = pd.Series(
        0.0,
        index=index
    )

    weight = (
        1 /
        max(1, len(companies))
    )

    for ticker in companies:

        stock = clean_series(
            ticker
        )

        stock = stock.reindex(
            index
        ).ffill().bfill()

        if stock.dropna().empty:

            continue

        portfolio += (
            weight *
            (
                stock /
                stock.iloc[0]
            ) *
            100
        )

    benchmark_normalized = (
        benchmark /
        benchmark.iloc[0]
        * 100
    )

    return (
        portfolio,
        benchmark_normalized
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("α ALPHA")

st.sidebar.caption(
    "Dynamic Portfolio Intelligence & Strategy"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🧠 ALPHA Strategy Engine",
        "📈 Market Pulse",
        "🔎 Stock Analysis",
        "⭐ Watchlist",
        "💼 My Portfolio",
        "🧩 Portfolio Builder",
        "⚠️ Risk & Performance",
        "🏦 Debt Analysis",
        "🎯 Goals",
        "🔔 Alerts",
        "📰 News & Events",
        "📊 Comparison",
        "📐 Advanced Analytics",
        "⚙️ Settings / Methodology"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("α ALPHA")

st.caption(
    "Dynamic multi-sector portfolio dashboard • "
    "Investment date: 01 Sep 2026 • "
    "Target date: 01 Sep 2029"
)


# ============================================================
# WEEKLY REPORT SELECTOR
# ============================================================

left, right = st.columns([3, 2])


with left:

    dates = report_dates()

    labels = (
        ["LIVE — Latest Available Data"] +
        [
            d.strftime("%d %b %Y")
            for d in dates
        ]
    )

    current_index = 0

    if (
        st.session_state.mode ==
        "historical"
        and
        st.session_state.report_date
        is not None
    ):

        target_label = (
            pd.Timestamp(
                st.session_state.report_date
            ).strftime("%d %b %Y")
        )

        if target_label in labels:

            current_index = (
                labels.index(
                    target_label
                )
            )

    choice = st.selectbox(
        "Weekly report date",
        labels,
        index=current_index,
        help=(
            "Select the last available "
            "trading day of a week."
        )
    )

    if choice.startswith("LIVE"):

        st.session_state.mode = "live"

        st.session_state.report_date = None

    else:

        st.session_state.mode = (
            "historical"
        )

        st.session_state.report_date = (
            pd.Timestamp(choice)
        )

    if (
        st.session_state.mode ==
        "historical"
    ):

        d = (
            st.session_state.report_date
        )

        st.markdown(
            f"""
            **Historical report as on
            {d:%d %b %Y}**

            Investment date:
            **01 Sep 2026**

            Target date:
            **01 Sep 2029**
            """
        )


with right:

    if (
        st.session_state.mode ==
        "live"
    ):

        st.success(
            "🟢 LIVE / LATEST AVAILABLE DATA"
        )

    else:

        st.info(
            f"""
            📅 HISTORICAL SNAPSHOT

            {st.session_state.report_date:%d %b %Y}
            """
        )

        if st.button(
            "↩️ Back to Live Data",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.mode = "live"

            st.session_state.report_date = None

            st.rerun()


selected_date = (
    None
    if st.session_state.mode == "live"
    else st.session_state.report_date
)


# ============================================================
# COMMON PORTFOLIO DATA
# ============================================================

companies = st.session_state.companies

portfolio = equal_portfolio(
    companies,
    selected_date
)

current_value = (
    float(portfolio.Value.sum())
    if len(portfolio)
    else 0
)

invested = TOTAL_CAPITAL

profit = (
    current_value -
    invested
)

portfolio_return = (
    profit /
    invested *
    100
    if invested
    else 0
)

portfolio_history, benchmark_history_series = (
    portfolio_series(
        companies,
        selected_date
    )
)

benchmark_return = (
    benchmark_history_series.iloc[-1] -
    100
    if len(benchmark_history_series)
    else np.nan
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header(
        "🏠 Investor Overview"
    )

    k = st.columns(5)

    k[0].metric(
        "Portfolio Value",
        inr(current_value)
    )

    k[1].metric(
        "Invested Amount",
        inr(invested)
    )

    k[2].metric(
        "Profit / Loss",
        inr(profit),
        pct(portfolio_return)
    )

    k[3].metric(
        "NIFTY 50 Return",
        pct(benchmark_return)
    )

    k[4].metric(
        "Investor Persona",
        st.session_state.persona
    )

    st.subheader(
        "Portfolio Snapshot"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        if len(portfolio):

            fig = px.pie(
                portfolio,
                names="Sector",
                values="Value",
                hole=0.5,
                title="Sector Allocation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        if len(portfolio):

            fig = px.pie(
                portfolio,
                names="Company",
                values="Value",
                hole=0.5,
                title="Company Allocation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c3:

        if len(portfolio):

            top = portfolio.nlargest(
                5,
                "Value"
            )

            display = top[
                ["Company", "Value"]
            ].copy()

            display["Value"] = (
                display["Value"].map(inr)
            )

            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True
            )

            concentration = (
                top["Value"].sum() /
                current_value *
                100
                if current_value
                else 0
            )

            st.metric(
                "Top 5 Concentration",
                f"{concentration:.1f}%"
            )

    if len(portfolio_history):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=portfolio_history.index,
                y=portfolio_history,
                name="ALPHA Base Portfolio"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=benchmark_history_series.index,
                y=benchmark_history_series,
                name="NIFTY 50"
            )
        )

        fig.update_layout(
            title=(
                "Portfolio Growth vs NIFTY 50 "
                "(Base = 100)"
            ),
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 2 — ALPHA STRATEGY ENGINE
# ============================================================

elif page == "🧠 ALPHA Strategy Engine":

    st.header(
        "🧠 ALPHA Strategy Engine"
    )

    st.markdown(
        """
        ### Dynamic Portfolio Dashboard & Strategy

        ALPHA uses:

        **Investor Persona → Peer-relative P/E →
        Beta-weighted risk screening → Momentum →
        Quality → Stock ranking → Weekly rebalancing**
        """
    )

    p1, p2, p3 = st.columns(
        [2, 2, 3]
    )

    with p1:

        persona = st.selectbox(
            "Investor Persona",
            [
                "Conservative",
                "Balanced",
                "Aggressive"
            ],
            index=[
                "Conservative",
                "Balanced",
                "Aggressive"
            ].index(
                st.session_state.persona
            )
        )

    with p2:

        st.metric(
            "Rebalancing Frequency",
            "Weekly"
        )

    with p3:

        st.info(
            """
            Lower relative P/E and lower Beta
            improve valuation/risk scores.
            Momentum and ROE act as supporting factors.
            """
        )

    st.session_state.persona = persona

    snapshot = strategy_snapshot(
        companies,
        persona,
        selected_date
    )

    if snapshot.empty:

        st.warning(
            "Strategy data is currently unavailable."
        )

        st.stop()

    config = PERSONAS[
        persona
    ]

    st.subheader(
        f"{persona} Strategy — Weekly Selection"
    )

    q = st.columns(4)

    q[0].metric(
        "Universe",
        len(companies)
    )

    q[1].metric(
        "Selected Stocks",
        int(snapshot.Selected.sum())
    )

    q[2].metric(
        "Maximum Position",
        f"{config['max_weight']*100:.0f}%"
    )

    q[3].metric(
        "Valuation Weight",
        f"{config['valuation']*100:.0f}%"
    )

    # --------------------------------------------------------
    # STRATEGY TABLE
    # --------------------------------------------------------

    display = snapshot[
        [
            "Company",
            "Ticker",
            "Sector",
            "Peer Group",
            "P/E",
            "Peer P/E",
            "Relative P/E",
            "Beta",
            "6M Return",
            "ROE",
            "Strategy Score",
            "New Weight",
            "Action",
            "Reason"
        ]
    ].copy()

    display["New Weight"] = (
        display["New Weight"] *
        100
    )

    display["6M Return"] = (
        display["6M Return"] *
        100
    )

    display["ROE"] = (
        display["ROE"] *
        100
    )

    display = display.rename(
        columns={
            "New Weight": "Weight %",
            "6M Return": "6M Return %",
            "ROE": "ROE %"
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # SELECTED STOCKS
    # --------------------------------------------------------

    st.subheader(
        "Why ALPHA Selected These Stocks"
    )

    selected = snapshot[
        snapshot.Selected
    ]

    if len(selected):

        for _, row in selected.iterrows():

            relative_pe = (
                f"{row['Relative P/E']:.2f}x"
                if pd.notna(
                    row["Relative P/E"]
                )
                else "N/A"
            )

            beta_value = (
                f"{row['Beta']:.2f}"
                if pd.notna(
                    row["Beta"]
                )
                else "N/A"
            )

            st.write(
                f"""
                **{row['Company']}**

                Weight:
                **{row['New Weight']*100:.1f}%**

                • Relative P/E:
                **{relative_pe}**

                • Beta:
                **{beta_value}**

                • Strategy Score:
                **{row['Strategy Score']:.1f}**
                """
            )

    # --------------------------------------------------------
    # WEEKLY REBALANCING HISTORY
    # --------------------------------------------------------

    st.subheader(
        "🔄 Weekly Rebalancing History"
    )

    weekly = weekly_strategy_history(
        companies,
        persona
    )

    if len(weekly):

        weekly[
            "Cumulative Return %"
        ] = (
            1 +
            weekly["Return"].fillna(0)
        ).cumprod().sub(1) * 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=weekly["Date"],
                y=weekly[
                    "Cumulative Return %"
                ],
                mode="lines+markers",
                name="ALPHA Strategy"
            )
        )

        fig.update_layout(
            title=(
                "Weekly Rebalanced "
                "Strategy — Cumulative Return"
            ),
            yaxis_title="Return %",
            height=380
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        history_display = weekly.copy()

        history_display["Return"] = (
            history_display["Return"] *
            100
        )

        st.dataframe(
            history_display,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "Weekly backtest needs sufficient historical observations."
        )

    # --------------------------------------------------------
    # METHODOLOGY NOTE
    # --------------------------------------------------------

    st.caption(
        """
        Data note: Yahoo Finance provides current fundamental
        fields more reliably than historical point-in-time P/E.
        Therefore ALPHA uses the latest available P/E fields for
        the peer-relative valuation screen, while price, Beta and
        momentum are calculated as-of the selected weekly date.
        """
    )


# ============================================================
# PAGE 3 — MARKET PULSE
# ============================================================

elif page == "📈 Market Pulse":

    st.header(
        "📈 Market Pulse"
    )

    indices = {

        "NIFTY 50":
            "^NSEI",

        "SENSEX":
            "^BSESN",

        "NIFTY NEXT 50":
            "^NSMIDCP",

        "NIFTY 100":
            "^CNX100",

        "BANK NIFTY":
            "^NSEBANK",

        "INDIA VIX":
            "^INDIAVIX"
    }

    rows = []

    for label, ticker in indices.items():

        series = clean_series(
            ticker
        )

        if selected_date is not None:

            price = price_on_or_before(
                ticker,
                selected_date
            )

        else:

            price = (
                series.iloc[-1]
                if len(series)
                else np.nan
            )

        if selected_date is not None:

            previous = series[
                series.index <
                pd.Timestamp(
                    selected_date
                )
            ]

        else:

            previous = series

        previous_price = (
            previous.iloc[-1]
            if len(previous)
            else np.nan
        )

        day_change = (

            price /
            previous_price -
            1
        ) * 100 if (
            pd.notna(price)
            and
            pd.notna(previous_price)
            and
            previous_price
        ) else np.nan

        rows.append(
            [
                label,
                price,
                day_change
            ]
        )

    market = pd.DataFrame(
        rows,
        columns=[
            "Index",
            "Level",
            "Day Change %"
        ]
    )

    market["Level"] = market[
        "Level"
    ].map(
        lambda x:
        f"{x:,.2f}"
        if pd.notna(x)
        else "-"
    )

    market["Day Change %"] = (
        market["Day Change %"]
        .map(pct)
    )

    st.dataframe(
        market,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        """
        Index levels are shown as points.
        Market-data availability can vary by index and date.
        """
    )


# ============================================================
# PAGE 4 — STOCK ANALYSIS
# ============================================================

elif page == "🔎 Stock Analysis":

    st.header(
        "🔎 Stock / Company Analysis"
    )

    ticker = st.selectbox(
        "Company / Ticker",
        sorted(companies),
        format_func=lambda x:
        f"{name(x)} ({x})"
    )

    series = clean_series(
        ticker
    )

    fundamentals = info(
        ticker
    )

    price = selected_price(
        ticker,
        selected_date
    )

    if series.empty:

        st.warning(
            "History unavailable."
        )

        st.stop()

    if selected_date is not None:

        series = series[
            series.index <=
            selected_date
        ]

    day_change = (

        series.iloc[-1] /
        series.iloc[-2] -
        1

    ) * 100 if len(series) > 1 else np.nan

    pe = fundamentals.get(
        "trailingPE"
    )

    pb = fundamentals.get(
        "priceToBook"
    )

    eps = fundamentals.get(
        "trailingEps"
    )

    roe = fundamentals.get(
        "returnOnEquity"
    )

    debt_equity = fundamentals.get(
        "debtToEquity"
    )

    beta = beta_at(
        ticker,
        selected_date
    )

    metrics = st.columns(6)

    metrics[0].metric(
        "Price",
        inr(price)
    )

    metrics[1].metric(
        "Day Change",
        pct(day_change)
    )

    metrics[2].metric(
        "P/E",
        f"{pe:.2f}x"
        if pd.notna(pe)
        else "N/A"
    )

    metrics[3].metric(
        "Beta",
        f"{beta:.2f}"
        if pd.notna(beta)
        else "N/A"
    )

    metrics[4].metric(
        "P/B",
        f"{pb:.2f}"
        if pd.notna(pb)
        else "N/A"
    )

    metrics[5].metric(
        "ROE",
        f"{roe*100:.1f}%"
        if pd.notna(roe)
        else "N/A"
    )

    st.write(
        f"""
        **{name(ticker)}**

        Sector:
        **{sector(ticker)}**

        Peer Group:
        **{peer(ticker)}**
        """
    )

    st.plotly_chart(
        px.line(
            series,
            title=
            f"{name(ticker)} — Price History"
        ),
        use_container_width=True
    )

    fundamental_table = pd.DataFrame({

        "Metric": [
            "P/E",
            "P/B",
            "EPS",
            "ROE",
            "Debt / Equity",
            "Beta"
        ],

        "Value": [
            pe,
            pb,
            eps,
            roe,
            debt_equity,
            beta
        ]
    })

    st.dataframe(
        fundamental_table,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 5 — WATCHLIST
# ============================================================

elif page == "⭐ Watchlist":

    st.header(
        "⭐ Watchlist"
    )

    available = sorted(
        set(
            companies +
            DEFAULT_COMPANIES +
            st.session_state.watchlist
        )
    )

    ticker = st.selectbox(
        "Company",
        available,
        format_func=lambda x:
        name(x)
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "⭐ Add to Watchlist",
            use_container_width=True
        ):

            if ticker not in (
                st.session_state.watchlist
            ):

                st.session_state.watchlist.append(
                    ticker
                )

                st.rerun()

    with c2:

        if st.button(
            "🗑 Remove Selected",
            use_container_width=True
        ):

            if ticker in (
                st.session_state.watchlist
            ):

                st.session_state.watchlist.remove(
                    ticker
                )

                st.rerun()

    if st.session_state.watchlist:

        rows = []

        for ticker in (
            st.session_state.watchlist
        ):

            price = latest_price(
                ticker
            )

            fundamentals = info(
                ticker
            )

            target = (
                st.session_state
                .target_prices
                .get(ticker)
            )

            status = (
                "🟢 Target reached"
                if target
                and price <= target
                else "🟡 Watch"
            )

            rows.append({

                "Company":
                    name(ticker),

                "Ticker":
                    ticker,

                "Price":
                    price,

                "P/E":
                    fundamentals.get(
                        "trailingPE"
                    ),

                "Target Price":
                    target,

                "Status":
                    status
            })

        watchlist_table = pd.DataFrame(
            rows
        )

        watchlist_table["Price"] = (
            watchlist_table[
                "Price"
            ].map(inr)
        )

        watchlist_table[
            "Target Price"
        ] = watchlist_table[
            "Target Price"
        ].map(inr)

        st.dataframe(
            watchlist_table,
            hide_index=True,
            use_container_width=True
        )

    st.subheader(
        "Set Target Price"
    )

    target = st.number_input(
        "Target Price",
        min_value=0.0,
        value=float(
            st.session_state
            .target_prices
            .get(ticker, 0.0)
        )
    )

    if st.button(
        "Save Target Price"
    ):

        st.session_state.target_prices[
            ticker
        ] = (
            target
            if target > 0
            else None
        )

        st.success(
            "Target price saved."
        )


# ============================================================
# PAGE 6 — MY PORTFOLIO
# ============================================================

elif page == "💼 My Portfolio":

    st.header(
        "💼 My Portfolio"
    )

    display = portfolio.copy()

    if len(display):

        display["Weight"] = (
            display["Weight"] *
            100
        )

        for column in [
            "Allocated",
            "Entry Price",
            "Price",
            "Value",
            "P/L"
        ]:

            display[column] = (
                display[column]
                .map(inr)
            )

        display = display.rename(
            columns={
                "Weight":
                    "Weight %"
            }
        )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True
    )

    if len(portfolio):

        st.subheader(
            "Contributors / Detractors"
        )

        contributors = (
            portfolio
            .sort_values(
                "P/L",
                ascending=False
            )[
                [
                    "Company",
                    "P/L"
                ]
            ]
            .copy()
        )

        contributors["P/L"] = (
            contributors["P/L"]
            .map(inr)
        )

        st.dataframe(
            contributors,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# PAGE 7 — PORTFOLIO BUILDER
# ============================================================

elif page == "🧩 Portfolio Builder":

    st.header(
        "🧩 Portfolio Builder"
    )

    st.caption(
        """
        Add or delete companies.
        Whole-share allocation retains unused cash.
        """
    )

    ticker_input = st.text_input(
        "Add NSE ticker",
        placeholder=
        "e.g. TCS or RELIANCE.NS"
    ).strip().upper()

    if st.button(
        "➕ Add Company"
    ) and ticker_input:

        ticker = (
            ticker_input
            if ticker_input.endswith(".NS")
            else ticker_input + ".NS"
        )

        test_series = clean_series(
            ticker
        )

        if test_series.empty:

            st.error(
                "Ticker could not be validated through Yahoo Finance."
            )

        elif ticker not in (
            st.session_state.companies
        ):

            META.setdefault(
                ticker,
                (ticker, "Other")
            )

            PEER_GROUP.setdefault(
                ticker,
                "Other"
            )

            st.session_state.companies.append(
                ticker
            )

            st.rerun()

    if companies:

        delete_ticker = st.selectbox(
            "Delete company",
            companies,
            format_func=lambda x:
            name(x)
        )

        if st.button(
            "🗑 Delete Company"
        ):

            if len(companies) > 1:

                st.session_state.companies.remove(
                    delete_ticker
                )

                st.rerun()

    builder_table = pd.DataFrame({

        "Ticker":
            companies,

        "Company":
            [
                name(x)
                for x in companies
            ],

        "Sector":
            [
                sector(x)
                for x in companies
            ],

        "Peer Group":
            [
                peer(x)
                for x in companies
            ]
    })

    st.dataframe(
        builder_table,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 8 — RISK & PERFORMANCE
# ============================================================

elif page == "⚠️ Risk & Performance":

    st.header(
        "⚠️ Risk & Performance"
    )

    returns = (
        portfolio_history
        .pct_change()
        .dropna()
    )

    volatility = (
        returns.std() *
        sqrt(252)
        if len(returns) > 1
        else np.nan
    )

    drawdown = (

        portfolio_history /
        portfolio_history.cummax() -
        1

    ).min() if len(
        portfolio_history
    ) else np.nan

    metrics = st.columns(4)

    metrics[0].metric(
        "Volatility",
        f"{volatility*100:.2f}%"
        if pd.notna(volatility)
        else "N/A"
    )

    metrics[1].metric(
        "Maximum Drawdown",
        f"{drawdown*100:.2f}%"
        if pd.notna(drawdown)
        else "N/A"
    )

    metrics[2].metric(
        "Portfolio Return",
        pct(portfolio_return)
    )

    metrics[3].metric(
        "Benchmark Return",
        pct(benchmark_return)
    )

    if len(portfolio_history):

        drawdown_series = (
            portfolio_history /
            portfolio_history.cummax() -
            1
        )

        st.plotly_chart(
            px.line(
                x=drawdown_series.index,
                y=drawdown_series.values,
                title="Portfolio Drawdown"
            ),
            use_container_width=True
        )


# ============================================================
# PAGE 9 — DEBT ANALYSIS
# ============================================================

elif page == "🏦 Debt Analysis":

    st.header(
        "🏦 Debt / Fixed Income Analysis"
    )

    c = st.columns(4)

    instrument = c[0].text_input(
        "Instrument",
        "Custom Bond"
    )

    face_value = c[1].number_input(
        "Investment / Face Value",
        min_value=0.0,
        value=1_00_000.0
    )

    coupon = c[2].number_input(
        "Coupon %",
        min_value=0.0,
        value=7.0
    )

    maturity = c[3].date_input(
        "Maturity",
        date(2029, 9, 1)
    )

    years = max(
        0,
        (
            pd.Timestamp(maturity) -
            pd.Timestamp.today().normalize()
        ).days / 365
    )

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Annual Coupon",
        inr(
            face_value *
            coupon /
            100
        )
    )

    d2.metric(
        "Maturity Value",
        inr(face_value)
    )

    d3.metric(
        "Years to Maturity",
        f"{years:.2f}"
    )


# ============================================================
# PAGE 10 — GOALS
# ============================================================

elif page == "🎯 Goals":

    st.header(
        "🎯 Investment Goals"
    )

    c = st.columns(4)

    target_amount = c[0].number_input(
        "Target Amount",
        min_value=0.0,
        value=2_00_00_000.0
    )

    horizon = c[1].number_input(
        "Horizon (years)",
        min_value=0.1,
        value=3.0
    )

    monthly = c[2].number_input(
        "Monthly Contribution",
        min_value=0.0,
        value=0.0
    )

    expected_return = c[3].number_input(
        "Expected Return %",
        min_value=0.0,
        value=12.0
    ) / 100

    projected = (
        current_value *
        (
            1 +
            expected_return
        ) ** horizon
    )

    shortfall = (
        target_amount -
        projected
    )

    required_cagr = (

        (
            target_amount /
            current_value
        ) ** (
            1 /
            horizon
        ) - 1

    ) if current_value > 0 else np.nan

    metrics = st.columns(4)

    metrics[0].metric(
        "Current Corpus",
        inr(current_value)
    )

    metrics[1].metric(
        "Projected Corpus",
        inr(projected)
    )

    metrics[2].metric(
        "Shortfall / Surplus",
        inr(shortfall)
    )

    metrics[3].metric(
        "Required CAGR",
        f"{required_cagr*100:.2f}%"
        if pd.notna(required_cagr)
        else "N/A"
    )


# ============================================================
# PAGE 11 — ALERTS
# ============================================================

elif page == "🔔 Alerts":

    st.header(
        "🔔 Alerts"
    )

    rows = []

    for ticker in companies:

        price = latest_price(
            ticker
        )

        target = (
            st.session_state
            .target_prices
            .get(ticker)
        )

        if target:

            rows.append({

                "Company":
                    name(ticker),

                "Price":
                    price,

                "Target":
                    target,

                "Status":
                    (
                        "🟢 At / below target"
                        if price <= target
                        else
                        "🔴 Above target"
                    )
            })

    if rows:

        alerts = pd.DataFrame(
            rows
        )

        alerts["Price"] = (
            alerts["Price"]
            .map(inr)
        )

        alerts["Target"] = (
            alerts["Target"]
            .map(inr)
        )

        st.dataframe(
            alerts,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "No target-price alerts configured."
        )


# ============================================================
# PAGE 12 — NEWS & EVENTS
# ============================================================

elif page == "📰 News & Events":

    st.header(
        "📰 News & Events"
    )

    ticker = st.selectbox(
        "Company",
        companies,
        format_func=lambda x:
        name(x)
    )

    st.info(
        f"""
        Selected company:
        **{name(ticker)}**

        Live news/corporate-event aggregation
        requires a dedicated news provider/API.
        ALPHA does not fabricate headlines.
        """
    )

    st.write(
        """
        Suggested event categories:

        • Earnings

        • Dividends

        • Stock splits

        • Bonus issues

        • Regulatory developments

        • Promoter activity
        """
    )


# ============================================================
# PAGE 13 — COMPARISON
# ============================================================

elif page == "📊 Comparison":

    st.header(
        "📊 Company Comparison"
    )

    chosen = st.multiselect(
        "Select companies",
        companies,
        default=companies[:5],
        format_func=lambda x:
        name(x)
    )

    rows = []

    for ticker in chosen:

        fundamentals = info(
            ticker
        )

        rows.append({

            "Company":
                name(ticker),

            "Sector":
                sector(ticker),

            "Price":
                selected_price(
                    ticker,
                    selected_date
                ),

            "P/E":
                fundamentals.get(
                    "trailingPE"
                ),

            "P/B":
                fundamentals.get(
                    "priceToBook"
                ),

            "EPS":
                fundamentals.get(
                    "trailingEps"
                ),

            "ROE":
                fundamentals.get(
                    "returnOnEquity"
                ),

            "Beta":
                beta_at(
                    ticker,
                    selected_date
                ),

            "6M Return":
                return_at(
                    ticker,
                    selected_date,
                    126
                )
        })

    if rows:

        comparison = pd.DataFrame(
            rows
        )

        comparison["Price"] = (
            comparison["Price"]
            .map(inr)
        )

        comparison["ROE"] = (
            comparison["ROE"]
            .map(
                lambda x:
                f"{x*100:.2f}%"
                if pd.notna(x)
                else "-"
            )
        )

        comparison["6M Return"] = (
            comparison["6M Return"]
            .map(
                lambda x:
                f"{x*100:.2f}%"
                if pd.notna(x)
                else "-"
            )
        )

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# PAGE 14 — ADVANCED ANALYTICS
# ============================================================

elif page == "📐 Advanced Analytics":

    st.header(
        "📐 Advanced Analytics"
    )

    risk_free_rate = (
        st.number_input(
            "Risk-free rate %",
            min_value=0.0,
            value=6.5
        ) / 100
    )

    market_return = (
        st.number_input(
            "Expected market return %",
            value=12.0
        ) / 100
    )

    selected_beta = st.number_input(
        "Beta",
        value=1.0,
        step=0.05
    )

    capm_return = (
        risk_free_rate +
        selected_beta *
        (
            market_return -
            risk_free_rate
        )
    )

    st.metric(
        "CAPM Expected Return",
        f"{capm_return*100:.2f}%"
    )

    st.info(
        """
        CAPM is used as an analytical framework.
        ALPHA's Strategy Engine separately applies
        Beta-weighted risk screening and peer-relative
        valuation.
        """
    )


# ============================================================
# PAGE 15 — SETTINGS / METHODOLOGY
# ============================================================

elif page == "⚙️ Settings / Methodology":

    st.header(
        "⚙️ Settings / Methodology"
    )

    st.session_state.portfolio_name = (
        st.text_input(
            "Portfolio Name",
            st.session_state.portfolio_name
        )
    )

    st.session_state.benchmark = (
        st.text_input(
            "Benchmark Ticker",
            st.session_state.benchmark
        )
    )

    st.subheader(
        "Default Investment Universe"
    )

    default_table = pd.DataFrame({

        "Ticker":
            DEFAULT_COMPANIES,

        "Company":
            [
                name(x)
                for x in DEFAULT_COMPANIES
            ],

        "Peer Group":
            [
                peer(x)
                for x in DEFAULT_COMPANIES
            ]
    })

    st.dataframe(
        default_table,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "ALPHA Weekly Strategy Methodology"
    )

    st.markdown(
        """
        ### 1. Investor Persona

        ALPHA provides three investor profiles:

        **Conservative**
        - 40% valuation
        - 40% risk
        - 10% momentum
        - 10% quality

        **Balanced**
        - 30% valuation
        - 25% risk
        - 25% momentum
        - 20% quality

        **Aggressive**
        - 25% valuation
        - 15% risk
        - 40% momentum
        - 20% quality


        ### 2. Peer-Relative P/E

        ALPHA calculates:

        **Relative P/E =
        Company P/E ÷ Peer Group Median P/E**

        A value below 1.00x indicates that the
        company's P/E is below its selected peer
        median.

        Lower relative P/E receives a higher
        valuation score.


        ### 3. Beta-Weighted Risk Screening

        Beta is calculated against the selected
        benchmark using historical market returns.

        Lower Beta receives a higher risk score.

        The importance of this factor changes
        according to the investor persona.


        ### 4. Momentum

        ALPHA calculates 6-month price momentum.

        Higher momentum receives a higher score.


        ### 5. Quality

        Return on Equity is used as a supporting
        quality factor when available.


        ### 6. Strategy Score

        The final score is:

        **Strategy Score =**

        **Valuation Score × Persona Valuation Weight**

        **+ Risk Score × Persona Risk Weight**

        **+ Momentum Score × Persona Momentum Weight**

        **+ Quality Score × Persona Quality Weight**


        ### 7. Weekly Rebalancing

        Every available weekly report date is treated
        as a potential rebalancing point.

        ALPHA:

        1. Screens the investment universe.
        2. Calculates factor scores.
        3. Ranks companies.
        4. Selects the highest-ranked stocks.
        5. Calculates new portfolio weights.
        6. Applies persona-specific position limits.
        7. Rebalances the portfolio for the next week.


        ### 8. Explainability

        The dashboard displays:

        - P/E
        - Peer P/E
        - Relative P/E
        - Beta
        - 6M return
        - ROE
        - Strategy score
        - Portfolio weight
        - Selection status
        - Reason for selection
        """
    )

    st.subheader(
        "Important Data Limitation"
    )

    st.warning(
        """
        Yahoo Finance does not reliably expose
        point-in-time historical P/E for every
        Indian stock.

        Therefore ALPHA uses the latest available
        P/E fields for the peer-relative valuation
        screen, while price, Beta and momentum are
        calculated as-of the selected weekly date.

        This is deliberately disclosed instead of
        fabricating historical P/E observations.
        A paid point-in-time fundamentals provider
        can replace this component later.
        """
    )

    st.subheader(
        "Disclaimer"
    )

    st.caption(
        """
        ALPHA is an educational and analytical
        portfolio dashboard. Strategy scores,
        rankings and projections are not investment
        advice and do not guarantee future returns.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    α ALPHA • Dynamic Portfolio Intelligence & Strategy
    • Yahoo Finance data where available
    • Educational / analytical use only
    """
)
