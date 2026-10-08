import warnings
warnings.filterwarnings("ignore")

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
# ALPHA
# DYNAMIC PORTFOLIO INTELLIGENCE & STRATEGY DASHBOARD
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


# ============================================================
# DEFAULT STOCK UNIVERSE
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
# COMPANY INFORMATION
# ============================================================

META = {
    "BAJAJ-AUTO.NS": (
        "Bajaj Auto Ltd",
        "Automobiles"
    ),
    "MARUTI.NS": (
        "Maruti Suzuki India Ltd",
        "Automobiles"
    ),
    "M&M.NS": (
        "Mahindra & Mahindra Ltd",
        "Automobiles"
    ),
    "HAL.NS": (
        "Hindustan Aeronautics Ltd",
        "Defense / Capital Goods"
    ),
    "POLYCAB.NS": (
        "Polycab India Ltd",
        "Capital Goods / Manufacturing"
    ),
    "SIEMENS.NS": (
        "Siemens Ltd",
        "Capital Goods"
    ),
    "PIDILITIND.NS": (
        "Pidilite Industries Ltd",
        "Chemicals"
    ),
    "SOLARINDS.NS": (
        "Solar Industries India Ltd",
        "Chemicals / Defense"
    ),
    "PIIND.NS": (
        "PI Industries Ltd",
        "Chemicals"
    ),
    "BRITANNIA.NS": (
        "Britannia Industries Ltd",
        "FMCG"
    ),
    "HINDUNILVR.NS": (
        "Hindustan Unilever Ltd",
        "FMCG"
    ),
    "ITC.NS": (
        "ITC Ltd",
        "FMCG"
    ),
    "AXISBANK.NS": (
        "Axis Bank Ltd",
        "Financials"
    ),
    "BSE.NS": (
        "BSE Ltd",
        "Capital Markets"
    ),
    "ANGELONE.NS": (
        "Angel One Ltd",
        "Financial Services"
    ),
}


# ============================================================
# PEER GROUPS
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

def company_name(ticker):
    return META.get(
        ticker,
        (ticker, "Other")
    )[0]


def company_sector(ticker):
    return META.get(
        ticker,
        (ticker, "Other")
    )[1]


def company_peer(ticker):
    return PEER_GROUP.get(
        ticker,
        company_sector(ticker)
    )


def format_inr(value):

    if value is None or pd.isna(value):
        return "-"

    value = float(value)

    negative = value < 0
    value = abs(value)

    text = f"{value:,.2f}"

    if negative:
        return "-₹" + text

    return "₹" + text


def format_pct(value):

    if value is None or pd.isna(value):
        return "-"

    return f"{float(value):+.2f}%"


# ============================================================
# NETWORK TIMEOUT
# ============================================================

def bounded(function, timeout=NETWORK_TIMEOUT):

    try:

        with cf.ThreadPoolExecutor(
            max_workers=1
        ) as executor:

            return executor.submit(
                function
            ).result(
                timeout=timeout
            )

    except Exception:

        return None


# ============================================================
# YAHOO FINANCE DATA
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_history(ticker):

    def download():

        try:

            data = yf.Ticker(
                ticker
            ).history(
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

            if isinstance(
                data,
                pd.DataFrame
            ):

                return data

        except Exception:
            pass

        return pd.DataFrame()

    result = bounded(download)

    if isinstance(
        result,
        pd.DataFrame
    ):

        return result

    return pd.DataFrame()


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_info(ticker):

    result = bounded(
        lambda: yf.Ticker(ticker).info,
        6
    )

    if isinstance(
        result,
        dict
    ):

        return result

    return {}


# ============================================================
# PRICE SERIES
# ============================================================

def price_series(
    ticker,
    column="Close"
):

    data = get_history(ticker)

    if data.empty:
        return pd.Series(
            dtype=float
        )

    if column not in data.columns:
        return pd.Series(
            dtype=float
        )

    series = data[column].dropna()

    if series.empty:
        return pd.Series(
            dtype=float
        )

    index = pd.DatetimeIndex(
        pd.to_datetime(
            series.index
        )
    )

    # Remove timezone safely.
    if index.tz is not None:
        index = index.tz_localize(None)

    series.index = index

    return series


# ============================================================
# PRICE FUNCTIONS
# ============================================================

def price_on_or_before(
    ticker,
    selected_date
):

    series = price_series(
        ticker
    )

    if series.empty:
        return np.nan

    selected_date = pd.Timestamp(
        selected_date
    )

    series = series[
        series.index <= selected_date
    ]

    if series.empty:
        return np.nan

    return float(
        series.iloc[-1]
    )


def entry_price(ticker):

    series = price_series(
        ticker
    )

    series = series[
        series.index >=
        INVESTMENT_DATE
    ]

    if series.empty:
        return np.nan

    return float(
        series.iloc[0]
    )


def latest_price(ticker):

    series = price_series(
        ticker
    )

    if series.empty:
        return np.nan

    return float(
        series.iloc[-1]
    )


def selected_price(
    ticker,
    selected_date=None
):

    if selected_date is None:

        return latest_price(
            ticker
        )

    return price_on_or_before(
        ticker,
        selected_date
    )


# ============================================================
# IMPORTANT FIX:
# WEEKLY REPORT DATES
# ============================================================

def report_dates():

    series = price_series(
        st.session_state.benchmark
    )

    if series.empty:
        return []

    series = series[
        series.index >=
        INVESTMENT_DATE
    ]

    if series.empty:
        return []

    # Group by trading week and take
    # the LAST AVAILABLE TRADING DAY.
    grouped = (
        pd.Series(
            series.index,
            index=series.index
        )
        .groupby(
            series.index.to_period(
                "W-FRI"
            )
        )
        .max()
    )

    # IMPORTANT:
    # Convert every returned date to pd.Timestamp.
    # This prevents:
    #
    # AttributeError:
    # 'numpy.datetime64' object has no attribute 'strftime'
    #
    dates = []

    for value in grouped.tolist():

        try:

            dates.append(
                pd.Timestamp(value)
            )

        except Exception:

            continue

    dates = sorted(
        list(
            set(dates)
        )
    )

    return dates


# ============================================================
# BETA
# ============================================================

def calculate_beta(
    ticker,
    asof=None,
    window=252
):

    stock = price_series(
        ticker
    )

    benchmark = price_series(
        st.session_state.benchmark
    )

    if asof is not None:

        asof = pd.Timestamp(
            asof
        )

        stock = stock[
            stock.index <= asof
        ]

        benchmark = benchmark[
            benchmark.index <= asof
        ]

    stock_returns = (
        stock.pct_change()
    )

    benchmark_returns = (
        benchmark.pct_change()
    )

    combined = pd.concat(
        [
            stock_returns,
            benchmark_returns
        ],
        axis=1,
        join="inner"
    ).dropna()

    combined = combined.tail(
        window
    )

    if len(combined) < 30:
        return np.nan

    benchmark_variance = (
        combined.iloc[:, 1].var()
    )

    if benchmark_variance == 0:
        return np.nan

    beta = (
        combined.iloc[:, 0].cov(
            combined.iloc[:, 1]
        )
        /
        benchmark_variance
    )

    return float(beta)


# ============================================================
# MOMENTUM
# ============================================================

def calculate_return(
    ticker,
    asof=None,
    days=126
):

    series = price_series(
        ticker
    )

    if asof is not None:

        series = series[
            series.index <=
            pd.Timestamp(asof)
        ]

    if len(series) <= days:
        return np.nan

    return float(
        series.iloc[-1]
        /
        series.iloc[-days - 1]
        - 1
    )


# ============================================================
# PEER RELATIVE VALUATION
# ============================================================

def valuation_table(
    companies
):

    raw = []

    for ticker in companies:

        fundamentals = get_info(
            ticker
        )

        pe = fundamentals.get(
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
                company_peer(
                    other_ticker
                )
                ==
                company_peer(
                    ticker
                )
                and
                pd.notna(value)
                and
                0 < value < 200
            )
        ]

        if peer_values:

            peer_median = float(
                np.median(
                    peer_values
                )
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
                pe /
                peer_median
            )

        else:

            relative_pe = np.nan

        rows.append({

            "Ticker":
                ticker,

            "P/E":
                pe,

            "Peer P/E":
                peer_median,

            "Relative P/E":
                relative_pe
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    series,
    higher_is_better=True
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = values.dropna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    if valid.empty:
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
    persona,
    asof=None
):

    config = PERSONAS[
        persona
    ]

    valuation = valuation_table(
        companies
    )

    rows = []

    for _, valuation_row in (
        valuation.iterrows()
    ):

        ticker = (
            valuation_row["Ticker"]
        )

        beta = calculate_beta(
            ticker,
            asof
        )

        momentum = calculate_return(
            ticker,
            asof,
            126
        )

        fundamentals = get_info(
            ticker
        )

        roe = fundamentals.get(
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
                company_name(ticker),

            "Sector":
                company_sector(ticker),

            "Peer Group":
                company_peer(ticker),

            "P/E":
                valuation_row["P/E"],

            "Peer P/E":
                valuation_row["Peer P/E"],

            "Relative P/E":
                valuation_row["Relative P/E"],

            "Beta":
                beta,

            "6M Return":
                momentum,

            "ROE":
                roe
        })

    data = pd.DataFrame(
        rows
    )

    if data.empty:
        return data

    # --------------------------------------------------------
    # FACTOR SCORES
    # --------------------------------------------------------

    data[
        "Valuation Score"
    ] = normalize(
        data["Relative P/E"],
        higher_is_better=False
    )

    data[
        "Risk Score"
    ] = normalize(
        data["Beta"],
        higher_is_better=False
    )

    data[
        "Momentum Score"
    ] = normalize(
        data["6M Return"],
        higher_is_better=True
    )

    data[
        "Quality Score"
    ] = normalize(
        data["ROE"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # PERSONA WEIGHTED SCORE
    # --------------------------------------------------------

    data[
        "Strategy Score"
    ] = (

        data[
            "Valuation Score"
        ].fillna(0.5)
        *
        config["valuation"]

        +

        data[
            "Risk Score"
        ].fillna(0.5)
        *
        config["risk"]

        +

        data[
            "Momentum Score"
        ].fillna(0.5)
        *
        config["momentum"]

        +

        data[
            "Quality Score"
        ].fillna(0.5)
        *
        config["quality"]

    ) * 100

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    data = data.sort_values(
        "Strategy Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    selected = data.head(
        min(
            config["max_holdings"],
            len(data)
        )
    ).copy()

    if selected.empty:

        data["Selected"] = False
        data["New Weight"] = 0.0

        return data

    # --------------------------------------------------------
    # SCORE PROPORTIONAL WEIGHTS
    # --------------------------------------------------------

    scores = np.maximum(
        selected[
            "Strategy Score"
        ]
        .fillna(0)
        .values,

        0.01
    )

    weights = (
        scores /
        scores.sum()
    )

    max_weight = config[
        "max_weight"
    ]

    # Position cap
    for _ in range(30):

        over_limit = (
            weights >
            max_weight
        )

        if not over_limit.any():
            break

        excess = (
            weights[
                over_limit
            ]
            -
            max_weight
        ).sum()

        weights[
            over_limit
        ] = max_weight

        under_limit = (
            ~over_limit
        )

        if (
            under_limit.any()
            and
            weights[
                under_limit
            ].sum() > 0
        ):

            weights[
                under_limit
            ] += (
                excess
                *
                weights[
                    under_limit
                ]
                /
                weights[
                    under_limit
                ].sum()
            )

    if weights.sum() > 0:

        weights = (
            weights /
            weights.sum()
        )

    selected[
        "New Weight"
    ] = weights

    # --------------------------------------------------------
    # MERGE BACK
    # --------------------------------------------------------

    data[
        "Selected"
    ] = data[
        "Ticker"
    ].isin(
        selected[
            "Ticker"
        ]
    )

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

    data[
        "New Weight"
    ] = data[
        "New Weight"
    ].fillna(0)

    data[
        "Action"
    ] = np.where(
        data["Selected"],
        "SELECT / HOLD",
        "EXCLUDE"
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    def explanation(row):

        rel_pe = row[
            "Relative P/E"
        ]

        beta = row[
            "Beta"
        ]

        if (
            pd.notna(rel_pe)
            and
            pd.notna(beta)
        ):

            return (
                f"Relative P/E "
                f"{rel_pe:.2f}x | "
                f"Beta {beta:.2f}"
            )

        return (
            "Insufficient "
            "peer/factor data"
        )

    data[
        "Reason"
    ] = data.apply(
        explanation,
        axis=1
    )

    return data


# ============================================================
# WEEKLY STRATEGY HISTORY
# ============================================================

def weekly_strategy_history(
    companies,
    persona
):

    dates = report_dates()

    if len(dates) < 2:

        return pd.DataFrame()

    rows = []

    for i in range(
        len(dates) - 1
    ):

        current_date = pd.Timestamp(
            dates[i]
        )

        next_date = pd.Timestamp(
            dates[i + 1]
        )

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

        if selected.empty:
            continue

        weekly_return = 0.0

        valid = False

        for _, row in selected.iterrows():

            ticker = row[
                "Ticker"
            ]

            start_price = (
                price_on_or_before(
                    ticker,
                    current_date
                )
            )

            end_price = (
                price_on_or_before(
                    ticker,
                    next_date
                )
            )

            if (
                pd.notna(start_price)
                and
                pd.notna(end_price)
                and
                start_price > 0
            ):

                stock_return = (
                    end_price /
                    start_price -
                    1
                )

                weekly_return += (
                    stock_return
                    *
                    row["New Weight"]
                )

                valid = True

        if valid:

            rows.append({

                "Date":
                    current_date,

                "Next Week":
                    next_date,

                "Return":
                    weekly_return,

                "Holdings":
                    len(selected)
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# BASE PORTFOLIO
# ============================================================

def build_portfolio(
    companies,
    asof=None
):

    if not companies:
        return pd.DataFrame()

    allocation = (
        TOTAL_CAPITAL /
        len(companies)
    )

    rows = []

    for ticker in companies:

        entry = entry_price(
            ticker
        )

        current = selected_price(
            ticker,
            asof
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

            leftover = (
                allocation -
                quantity *
                entry
            )

        else:

            quantity = 0
            leftover = allocation

        if pd.notna(current):

            value = (
                quantity *
                current
                +
                leftover
            )

        else:

            value = leftover

        rows.append({

            "Ticker":
                ticker,

            "Company":
                company_name(ticker),

            "Sector":
                company_sector(ticker),

            "Allocated":
                allocation,

            "Entry Price":
                entry,

            "Price":
                current,

            "Qty":
                quantity,

            "Cash":
                leftover,

            "Value":
                value,

            "P/L":
                value -
                allocation
        })

    result = pd.DataFrame(
        rows
    )

    if (
        not result.empty
        and
        result["Value"].sum() > 0
    ):

        result[
            "Weight"
        ] = (
            result["Value"]
            /
            result["Value"].sum()
        )

    else:

        result[
            "Weight"
        ] = 0.0

    return result


# ============================================================
# BASE PORTFOLIO PERFORMANCE
# ============================================================

def portfolio_series(
    companies,
    asof=None
):

    benchmark = price_series(
        st.session_state.benchmark
    )

    if benchmark.empty:

        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float)
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
        max(
            1,
            len(companies)
        )
    )

    for ticker in companies:

        stock = price_series(
            ticker
        )

        stock = stock[
            stock.index >=
            INVESTMENT_DATE
        ]

        stock = stock.reindex(
            index
        ).ffill().bfill()

        if stock.empty:
            continue

        if pd.isna(
            stock.iloc[0]
        ):
            continue

        portfolio += (
            weight
            *
            (
                stock /
                stock.iloc[0]
            )
            *
            100
        )

    benchmark_normalized = (
        benchmark /
        benchmark.iloc[0]
        *
        100
    )

    return (
        portfolio,
        benchmark_normalized
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "α ALPHA"
)

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

st.title(
    "α ALPHA"
)

st.caption(
    "Dynamic Multi-Sector Portfolio Intelligence & Strategy"
)

st.caption(
    "Investment Date: 01 Sep 2026  |  "
    "Target Date: 01 Sep 2029"
)


# ============================================================
# WEEKLY REPORT SELECTOR
# ============================================================

left, right = st.columns(
    [3, 2]
)


with left:

    available_dates = report_dates()

    labels = [
        "LIVE — Latest Available Data"
    ]

    # FIX:
    # Every date is converted to pd.Timestamp
    # before calling strftime.

    labels.extend(
        [
            pd.Timestamp(
                d
            ).strftime(
                "%d %b %Y"
            )
            for d in available_dates
        ]
    )

    current_index = 0

    if (
        st.session_state.mode
        ==
        "historical"
        and
        st.session_state.report_date
        is not None
    ):

        historical_label = (
            pd.Timestamp(
                st.session_state.report_date
            ).strftime(
                "%d %b %Y"
            )
        )

        if historical_label in labels:

            current_index = (
                labels.index(
                    historical_label
                )
            )

    selected_label = st.selectbox(
        "Weekly Report Date",
        labels,
        index=current_index
    )

    if selected_label.startswith(
        "LIVE"
    ):

        st.session_state.mode = (
            "live"
        )

        st.session_state.report_date = (
            None
        )

    else:

        st.session_state.mode = (
            "historical"
        )

        # Convert selected label back
        # into a proper pandas Timestamp.

        st.session_state.report_date = (
            pd.to_datetime(
                selected_label,
                format="%d %b %Y"
            )
        )

    if (
        st.session_state.mode
        ==
        "historical"
    ):

        historical_date = pd.Timestamp(
            st.session_state.report_date
        )

        st.markdown(
            f"""
            **Historical Report as on
            {historical_date.strftime("%d %b %Y")}**

            Investment Date:
            **01 Sep 2026**

            Target Date:
            **01 Sep 2029**
            """
        )


with right:

    if (
        st.session_state.mode
        ==
        "live"
    ):

        st.success(
            "🟢 LIVE / LATEST AVAILABLE DATA"
        )

    else:

        historical_date = pd.Timestamp(
            st.session_state.report_date
        )

        st.info(
            "📅 HISTORICAL SNAPSHOT — "
            +
            historical_date.strftime(
                "%d %b %Y"
            )
        )

        if st.button(
            "↩️ Back to Live Data",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.mode = (
                "live"
            )

            st.session_state.report_date = (
                None
            )

            st.rerun()


selected_date = (
    None
    if st.session_state.mode == "live"
    else pd.Timestamp(
        st.session_state.report_date
    )
)


# ============================================================
# COMMON PORTFOLIO DATA
# ============================================================

companies = (
    st.session_state.companies
)

portfolio = build_portfolio(
    companies,
    selected_date
)

current_value = (
    float(
        portfolio["Value"].sum()
    )
    if not portfolio.empty
    else 0
)

invested_amount = (
    TOTAL_CAPITAL
)

profit = (
    current_value -
    invested_amount
)

portfolio_return = (
    profit /
    invested_amount *
    100
    if invested_amount > 0
    else 0
)

portfolio_history, benchmark_history_series = (
    portfolio_series(
        companies,
        selected_date
    )
)

benchmark_return = (
    benchmark_history_series.iloc[-1]
    -
    100
    if not benchmark_history_series.empty
    else np.nan
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header(
        "🏠 Investor Overview"
    )

    metrics = st.columns(
        5
    )

    metrics[0].metric(
        "Portfolio Value",
        format_inr(
            current_value
        )
    )

    metrics[1].metric(
        "Invested Amount",
        format_inr(
            invested_amount
        )
    )

    metrics[2].metric(
        "Profit / Loss",
        format_inr(
            profit
        ),
        format_pct(
            portfolio_return
        )
    )

    metrics[3].metric(
        "NIFTY 50 Return",
        format_pct(
            benchmark_return
        )
    )

    metrics[4].metric(
        "Investor Persona",
        st.session_state.persona
    )

    st.subheader(
        "Portfolio Allocation"
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        if not portfolio.empty:

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

        if not portfolio.empty:

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

        if not portfolio.empty:

            top5 = portfolio.nlargest(
                5,
                "Value"
            )

            display = top5[
                [
                    "Company",
                    "Value"
                ]
            ].copy()

            display[
                "Value"
            ] = display[
                "Value"
            ].map(
                format_inr
            )

            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True
            )

            concentration = (
                top5["Value"].sum()
                /
                current_value
                *
                100
                if current_value > 0
                else 0
            )

            st.metric(
                "Top 5 Concentration",
                f"{concentration:.1f}%"
            )

    if not portfolio_history.empty:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=portfolio_history.index,
                y=portfolio_history.values,
                name="ALPHA Base Portfolio",
                mode="lines"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=benchmark_history_series.index,
                y=benchmark_history_series.values,
                name="NIFTY 50",
                mode="lines"
            )
        )

        fig.update_layout(
            title=(
                "Portfolio Growth vs NIFTY 50 "
                "(Base = 100)"
            ),
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ALPHA STRATEGY ENGINE
# ============================================================

elif page == "🧠 ALPHA Strategy Engine":

    st.header(
        "🧠 ALPHA Strategy Engine"
    )

    st.markdown(
        """
        ### Dynamic Portfolio Dashboard & Strategy

        **Investor Persona → Peer-relative P/E →
        Beta-weighted risk screening → Momentum →
        Quality → Stock ranking → Weekly rebalancing**
        """
    )

    p1, p2, p3 = st.columns(
        [2, 2, 3]
    )

    with p1:

        persona_options = [
            "Conservative",
            "Balanced",
            "Aggressive"
        ]

        persona = st.selectbox(
            "Investor Persona",
            persona_options,
            index=persona_options.index(
                st.session_state.persona
            )
        )

        st.session_state.persona = (
            persona
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

    config = PERSONAS[
        persona
    ]

    snapshot = strategy_snapshot(
        companies,
        persona,
        selected_date
    )

    if snapshot.empty:

        st.warning(
            "Strategy data is currently unavailable."
        )

    else:

        st.subheader(
            f"{persona} Strategy — Current Selection"
        )

        k = st.columns(
            4
        )

        k[0].metric(
            "Universe",
            len(companies)
        )

        k[1].metric(
            "Selected Stocks",
            int(
                snapshot[
                    "Selected"
                ].sum()
            )
        )

        k[2].metric(
            "Maximum Position",
            f"{config['max_weight']*100:.0f}%"
        )

        k[3].metric(
            "Valuation Weight",
            f"{config['valuation']*100:.0f}%"
        )

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

        display[
            "New Weight"
        ] *= 100

        display[
            "6M Return"
        ] *= 100

        display[
            "ROE"
        ] *= 100

        display = display.rename(
            columns={
                "New Weight":
                    "Weight %",
                "6M Return":
                    "6M Return %",
                "ROE":
                    "ROE %"
            }
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True
        )

        st.subheader(
            "Why ALPHA Selected These Stocks"
        )

        selected = snapshot[
            snapshot["Selected"]
        ]

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

                • Reason:
                {row['Reason']}
                """
            )

        st.subheader(
            "🔄 Weekly Rebalancing History"
        )

        weekly = weekly_strategy_history(
            companies,
            persona
        )

        if not weekly.empty:

            weekly[
                "Cumulative Return %"
            ] = (
                1 +
                weekly[
                    "Return"
                ].fillna(0)
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
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            history_display = weekly.copy()

            history_display[
                "Return"
            ] *= 100

            history_display = (
                history_display.rename(
                    columns={
                        "Return":
                            "Weekly Return %"
                    }
                )
            )

            st.dataframe(
                history_display,
                hide_index=True,
                use_container_width=True
            )

        else:

            st.info(
                "Weekly backtest requires sufficient market history."
            )

        st.caption(
            """
            Data limitation: Yahoo Finance does not reliably
            provide point-in-time historical P/E for every
            Indian stock. ALPHA therefore uses the latest
            available P/E field for peer-relative valuation,
            while price, Beta and momentum are calculated
            as-of the selected weekly date.
            """
        )


# ============================================================
# MARKET PULSE
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

        series = price_series(
            ticker
        )

        if selected_date is not None:

            current = price_on_or_before(
                ticker,
                selected_date
            )

            previous_series = series[
                series.index <
                pd.Timestamp(
                    selected_date
                )
            ]

        else:

            current = (
                series.iloc[-1]
                if not series.empty
                else np.nan
            )

            previous_series = series.iloc[:-1]

        previous = (
            previous_series.iloc[-1]
            if not previous_series.empty
            else np.nan
        )

        daily_change = (

            current /
            previous -
            1

        ) * 100 if (
            pd.notna(current)
            and
            pd.notna(previous)
            and
            previous != 0
        ) else np.nan

        rows.append({

            "Index":
                label,

            "Level":
                current,

            "Day Change %":
                daily_change
        })

    market = pd.DataFrame(
        rows
    )

    market[
        "Level"
    ] = market[
        "Level"
    ].map(
        lambda x:
        f"{x:,.2f}"
        if pd.notna(x)
        else "-"
    )

    market[
        "Day Change %"
    ] = market[
        "Day Change %"
    ].map(
        format_pct
    )

    st.dataframe(
        market,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Index levels are points. "
        "Yahoo Finance availability can vary by index."
    )


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif page == "🔎 Stock Analysis":

    st.header(
        "🔎 Stock Analysis"
    )

    ticker = st.selectbox(
        "Company",
        sorted(companies),
        format_func=lambda x:
        f"{company_name(x)} ({x})"
    )

    series = price_series(
        ticker
    )

    fundamentals = get_info(
        ticker
    )

    current = selected_price(
        ticker,
        selected_date
    )

    if series.empty:

        st.warning(
            "Price history is unavailable."
        )

    else:

        if selected_date is not None:

            series = series[
                series.index <=
                selected_date
            ]

        daily_change = (

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

        beta = calculate_beta(
            ticker,
            selected_date
        )

        k = st.columns(
            6
        )

        k[0].metric(
            "Price",
            format_inr(current)
        )

        k[1].metric(
            "Day Change",
            format_pct(daily_change)
        )

        k[2].metric(
            "P/E",
            f"{pe:.2f}x"
            if pd.notna(pe)
            else "N/A"
        )

        k[3].metric(
            "Beta",
            f"{beta:.2f}"
            if pd.notna(beta)
            else "N/A"
        )

        k[4].metric(
            "P/B",
            f"{pb:.2f}"
            if pd.notna(pb)
            else "N/A"
        )

        k[5].metric(
            "ROE",
            f"{roe*100:.1f}%"
            if pd.notna(roe)
            else "N/A"
        )

        st.write(
            f"""
            **{company_name(ticker)}**

            Sector:
            **{company_sector(ticker)}**

            Peer Group:
            **{company_peer(ticker)}**
            """
        )

        fig = px.line(
            series,
            title=(
                f"{company_name(ticker)} "
                "— Price History"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fundamental_data = pd.DataFrame({

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
            fundamental_data,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# WATCHLIST
# ============================================================

elif page == "⭐ Watchlist":

    st.header(
        "⭐ Watchlist"
    )

    available = sorted(
        set(
            companies
            +
            DEFAULT_COMPANIES
            +
            st.session_state.watchlist
        )
    )

    ticker = st.selectbox(
        "Company",
        available,
        format_func=lambda x:
        company_name(x)
    )

    c1, c2 = st.columns(
        2
    )

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

            current = latest_price(
                ticker
            )

            fundamentals = get_info(
                ticker
            )

            target = (
                st.session_state
                .target_prices
                .get(ticker)
            )

            status = (
                "🟢 Target reached"
                if (
                    target
                    and
                    pd.notna(current)
                    and
                    current <= target
                )
                else "🟡 Watch"
            )

            rows.append({

                "Company":
                    company_name(ticker),

                "Ticker":
                    ticker,

                "Price":
                    current,

                "P/E":
                    fundamentals.get(
                        "trailingPE"
                    ),

                "Target Price":
                    target,

                "Status":
                    status
            })

        watchlist = pd.DataFrame(
            rows
        )

        watchlist[
            "Price"
        ] = watchlist[
            "Price"
        ].map(
            format_inr
        )

        watchlist[
            "Target Price"
        ] = watchlist[
            "Target Price"
        ].map(
            format_inr
        )

        st.dataframe(
            watchlist,
            hide_index=True,
            use_container_width=True
        )

    st.subheader(
        "Set Target Price"
    )

    current_target = (
        st.session_state
        .target_prices
        .get(
            ticker,
            0.0
        )
    )

    target = st.number_input(
        "Target Price",
        min_value=0.0,
        value=float(
            current_target
            or 0.0
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
# MY PORTFOLIO
# ============================================================

elif page == "💼 My Portfolio":

    st.header(
        "💼 My Portfolio"
    )

    if portfolio.empty:

        st.info(
            "Portfolio data unavailable."
        )

    else:

        display = portfolio.copy()

        display[
            "Weight"
        ] *= 100

        for column in [
            "Allocated",
            "Entry Price",
            "Price",
            "Value",
            "P/L"
        ]:

            display[
                column
            ] = display[
                column
            ].map(
                format_inr
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

        contributors[
            "P/L"
        ] = contributors[
            "P/L"
        ].map(
            format_inr
        )

        st.dataframe(
            contributors,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# PORTFOLIO BUILDER
# ============================================================

elif page == "🧩 Portfolio Builder":

    st.header(
        "🧩 Portfolio Builder"
    )

    st.caption(
        "Add or delete companies. "
        "Whole-share allocation retains unused cash."
    )

    ticker_input = st.text_input(
        "Add NSE Ticker",
        placeholder="e.g. TCS or RELIANCE.NS"
    ).strip().upper()

    if st.button(
        "➕ Add Company"
    ) and ticker_input:

        ticker = (
            ticker_input
            if ticker_input.endswith(
                ".NS"
            )
            else ticker_input + ".NS"
        )

        validation = price_series(
            ticker
        )

        if validation.empty:

            st.error(
                "Ticker could not be validated through Yahoo Finance."
            )

        elif ticker not in (
            st.session_state.companies
        ):

            META.setdefault(
                ticker,
                (
                    ticker,
                    "Other"
                )
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
            "Delete Company",
            companies,
            format_func=lambda x:
            company_name(x)
        )

        if st.button(
            "🗑 Delete Company"
        ):

            if len(companies) > 1:

                st.session_state.companies.remove(
                    delete_ticker
                )

                st.rerun()

            else:

                st.warning(
                    "At least one company must remain."
                )

    builder = pd.DataFrame({

        "Ticker":
            companies,

        "Company":
            [
                company_name(x)
                for x in companies
            ],

        "Sector":
            [
                company_sector(x)
                for x in companies
            ],

        "Peer Group":
            [
                company_peer(x)
                for x in companies
            ]
    })

    st.dataframe(
        builder,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# RISK & PERFORMANCE
# ============================================================

elif page == "⚠️ Risk & Performance":

    st.header(
        "⚠️ Risk & Performance"
    )

    if portfolio_history.empty:

        st.warning(
            "Performance data unavailable."
        )

    else:

        returns = (
            portfolio_history
            .pct_change()
            .dropna()
        )

        volatility = (
            returns.std()
            *
            sqrt(252)
            if len(returns) > 1
            else np.nan
        )

        drawdown_series = (
            portfolio_history
            /
            portfolio_history.cummax()
            -
            1
        )

        maximum_drawdown = (
            drawdown_series.min()
        )

        k = st.columns(
            4
        )

        k[0].metric(
            "Annualized Volatility",
            (
                f"{volatility*100:.2f}%"
                if pd.notna(volatility)
                else "N/A"
            )
        )

        k[1].metric(
            "Maximum Drawdown",
            (
                f"{maximum_drawdown*100:.2f}%"
                if pd.notna(
                    maximum_drawdown
                )
                else "N/A"
            )
        )

        k[2].metric(
            "Portfolio Return",
            format_pct(
                portfolio_return
            )
        )

        k[3].metric(
            "Benchmark Return",
            format_pct(
                benchmark_return
            )
        )

        fig = px.line(
            x=drawdown_series.index,
            y=drawdown_series.values,
            title="Portfolio Drawdown"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DEBT ANALYSIS
# ============================================================

elif page == "🏦 Debt Analysis":

    st.header(
        "🏦 Debt / Fixed Income Analysis"
    )

    c = st.columns(
        4
    )

    c[0].text_input(
        "Instrument",
        "Custom Bond"
    )

    face_value = c[1].number_input(
        "Investment / Face Value",
        min_value=0.0,
        value=100000.0
    )

    coupon_rate = c[2].number_input(
        "Coupon %",
        min_value=0.0,
        value=7.0
    )

    maturity = c[3].date_input(
        "Maturity",
        date(
            2029,
            9,
            1
        )
    )

    years_to_maturity = max(
        0,
        (
            pd.Timestamp(
                maturity
            )
            -
            pd.Timestamp.today().normalize()
        ).days / 365
    )

    k = st.columns(
        3
    )

    k[0].metric(
        "Annual Coupon",
        format_inr(
            face_value
            *
            coupon_rate
            /
            100
        )
    )

    k[1].metric(
        "Maturity Value",
        format_inr(
            face_value
        )
    )

    k[2].metric(
        "Years to Maturity",
        f"{years_to_maturity:.2f}"
    )


# ============================================================
# GOALS
# ============================================================

elif page == "🎯 Goals":

    st.header(
        "🎯 Investment Goals"
    )

    c = st.columns(
        4
    )

    target_amount = c[0].number_input(
        "Target Amount",
        min_value=0.0,
        value=20000000.0
    )

    horizon = c[1].number_input(
        "Horizon (Years)",
        min_value=0.1,
        value=3.0
    )

    monthly_contribution = c[2].number_input(
        "Monthly Contribution",
        min_value=0.0,
        value=0.0
    )

    expected_return = (
        c[3].number_input(
            "Expected Return %",
            min_value=0.0,
            value=12.0
        )
        /
        100
    )

    projected = (
        current_value
        *
        (
            1 +
            expected_return
        )
        **
        horizon
    )

    shortfall = (
        target_amount -
        projected
    )

    required_cagr = (

        (
            target_amount /
            current_value
        )
        **
        (
            1 /
            horizon
        )
        -
        1

    ) if current_value > 0 else np.nan

    k = st.columns(
        4
    )

    k[0].metric(
        "Current Corpus",
        format_inr(
            current_value
        )
    )

    k[1].metric(
        "Projected Corpus",
        format_inr(
            projected
        )
    )

    k[2].metric(
        "Shortfall / Surplus",
        format_inr(
            shortfall
        )
    )

    k[3].metric(
        "Required CAGR",
        (
            f"{required_cagr*100:.2f}%"
            if pd.notna(required_cagr)
            else "N/A"
        )
    )


# ============================================================
# ALERTS
# ============================================================

elif page == "🔔 Alerts":

    st.header(
        "🔔 Alerts"
    )

    rows = []

    for ticker in companies:

        current = latest_price(
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
                    company_name(ticker),

                "Price":
                    current,

                "Target":
                    target,

                "Status":
                    (
                        "🟢 At / below target"
                        if (
                            pd.notna(current)
                            and
                            current <= target
                        )
                        else
                        "🔴 Above target"
                    )
            })

    if rows:

        alerts = pd.DataFrame(
            rows
        )

        alerts[
            "Price"
        ] = alerts[
            "Price"
        ].map(
            format_inr
        )

        alerts[
            "Target"
        ] = alerts[
            "Target"
        ].map(
            format_inr
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
# NEWS & EVENTS
# ============================================================

elif page == "📰 News & Events":

    st.header(
        "📰 News & Events"
    )

    ticker = st.selectbox(
        "Company",
        companies,
        format_func=lambda x:
        company_name(x)
    )

    st.info(
        f"""
        Selected company:

        **{company_name(ticker)}**

        Live news/corporate-event aggregation
        requires a dedicated news provider/API.

        ALPHA does not fabricate headlines.
        """
    )

    st.write(
        """
        Event categories supported by the dashboard framework:

        • Earnings

        • Dividends

        • Stock splits

        • Bonus issues

        • Regulatory developments

        • Promoter activity
        """
    )


# ============================================================
# COMPARISON
# ============================================================

elif page == "📊 Comparison":

    st.header(
        "📊 Company Comparison"
    )

    chosen = st.multiselect(
        "Select Companies",
        companies,
        default=companies[
            :min(5, len(companies))
        ],
        format_func=lambda x:
        company_name(x)
    )

    rows = []

    for ticker in chosen:

        fundamentals = get_info(
            ticker
        )

        rows.append({

            "Company":
                company_name(ticker),

            "Sector":
                company_sector(ticker),

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
                calculate_beta(
                    ticker,
                    selected_date
                ),

            "6M Return":
                calculate_return(
                    ticker,
                    selected_date,
                    126
                )
        })

    if rows:

        comparison = pd.DataFrame(
            rows
        )

        comparison[
            "Price"
        ] = comparison[
            "Price"
        ].map(
            format_inr
        )

        comparison[
            "ROE"
        ] = comparison[
            "ROE"
        ].map(
            lambda x:
            f"{x*100:.2f}%"
            if pd.notna(x)
            else "-"
        )

        comparison[
            "6M Return"
        ] = comparison[
            "6M Return"
        ].map(
            lambda x:
            f"{x*100:.2f}%"
            if pd.notna(x)
            else "-"
        )

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# ADVANCED ANALYTICS
# ============================================================

elif page == "📐 Advanced Analytics":

    st.header(
        "📐 Advanced Analytics"
    )

    risk_free_rate = (
        st.number_input(
            "Risk-Free Rate %",
            min_value=0.0,
            value=6.5
        )
        /
        100
    )

    market_return = (
        st.number_input(
            "Expected Market Return %",
            value=12.0
        )
        /
        100
    )

    beta = st.number_input(
        "Beta",
        value=1.0,
        step=0.05
    )

    capm_return = (
        risk_free_rate
        +
        beta
        *
        (
            market_return
            -
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

        ALPHA's Strategy Engine separately uses
        Beta-weighted risk screening and peer-relative
        valuation.
        """
    )


# ============================================================
# SETTINGS / METHODOLOGY
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
                company_name(x)
                for x in DEFAULT_COMPANIES
            ],

        "Sector":
            [
                company_sector(x)
                for x in DEFAULT_COMPANIES
            ],

        "Peer Group":
            [
                company_peer(x)
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

        ALPHA provides:

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

        Relative P/E is calculated as:

        **Company P/E ÷ Peer Group Median P/E**

        Lower Relative P/E receives a higher
        valuation score.


        ### 3. Beta-Weighted Risk Screening

        Beta is calculated using historical stock
        returns relative to the selected benchmark.

        Lower Beta receives a higher risk score.


        ### 4. Momentum

        ALPHA uses approximately six months of
        historical price momentum.


        ### 5. Quality

        Return on Equity is used as a supporting
        quality factor.


        ### 6. Strategy Score

        The final score combines:

        **Valuation + Risk + Momentum + Quality**

        according to the selected investor persona.


        ### 7. Weekly Rebalancing

        Each available weekly trading date is treated
        as a rebalancing point.

        ALPHA:

        1. Screens the investment universe.
        2. Calculates peer-relative P/E.
        3. Calculates Beta.
        4. Calculates momentum.
        5. Calculates quality.
        6. Ranks stocks.
        7. Selects stocks.
        8. Calculates new portfolio weights.
        9. Applies position limits.
        10. Rebalances for the next week.


        ### 8. Explainability

        ALPHA displays:

        - P/E
        - Peer P/E
        - Relative P/E
        - Beta
        - 6M return
        - ROE
        - Strategy Score
        - New portfolio weight
        - Selection status
        - Selection reason
        """
    )

    st.subheader(
        "Data Limitation"
    )

    st.warning(
        """
        Yahoo Finance does not reliably provide point-in-time
        historical P/E data for every Indian stock.

        Therefore ALPHA uses the latest available P/E field
        for the peer-relative valuation screen.

        Price, Beta and momentum are calculated using the
        selected historical date.

        This limitation is explicitly disclosed instead
        of fabricating historical fundamental data.
        """
    )

    st.subheader(
        "Disclaimer"
    )

    st.caption(
        """
        ALPHA is an educational and analytical portfolio
        dashboard. Strategy scores, rankings, portfolio
        weights and projected returns are not investment
        advice and do not guarantee future returns.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "α ALPHA • Dynamic Portfolio Intelligence & Strategy "
    "• Yahoo Finance data where available "
    "• Educational / analytical use only"
)
