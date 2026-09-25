"""
MarketFlow-AI: Interactive Streamlit Lakehouse & AI Workstation
Provides real-time visualization of streaming market ticks,
Lakehouse Medallion tier telemetry, DuckDB SQL workbench, and AI Copilot.
"""
import streamlit as st
import polars as pl
import pandas as pd
from pathlib import Path
import sys

# Ensure root is in path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from src.ingestion.collectors import StreamCollector
from src.transformation.cleaner import SilverCleaner
from src.lakehouse.engine import DuckDBEngine
from src.lakehouse.marts import GoldMartsBuilder
from src.lakehouse.feature_store import FeatureStore
from src.ai.model import VolatilityRegimeModel
from src.ai.sentiment_agent import MarketAnalystAgent

st.set_page_config(
    page_title="MarketFlow-AI | Real-Time Lakehouse & AI Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark modern terminal theme
st.markdown("""
<style>
    .metric-card {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-bronze { background: #92400e; color: #fef3c7; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-silver { background: #374151; color: #f3f4f6; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-gold { background: #854d0e; color: #fef08a; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_lakehouse_components():
    collector = StreamCollector(settings.BRONZE_DIR)
    cleaner = SilverCleaner(settings.BRONZE_DIR, settings.SILVER_DIR)
    engine = DuckDBEngine(settings.DUCKDB_PATH, settings.SILVER_DIR, settings.GOLD_DIR)
    marts = GoldMartsBuilder(engine, settings.GOLD_DIR)
    features = FeatureStore(engine)
    ai_model = VolatilityRegimeModel()
    agent = MarketAnalystAgent(engine, features, ai_model)
    return collector, cleaner, engine, marts, features, ai_model, agent


collector, cleaner, engine, marts, features, ai_model, agent = get_lakehouse_components()

# Sidebar
st.sidebar.title("⚡ MarketFlow-AI")
st.sidebar.caption("Real-Time Lakehouse & AI Intelligence")

symbol = st.sidebar.selectbox("Select Asset Symbol", settings.DEFAULT_SYMBOLS, index=0)

if st.sidebar.button("🔄 Ingest Live Micro-Batch (500 ticks)", type="primary"):
    with st.spinner("Ingesting to Bronze -> Cleaning to Silver -> Building Gold Marts..."):
        collector.ingest_batch(count=500, symbols=[symbol])
        cleaner.process_bronze_files()
        engine.refresh_views()
        marts.build_all_marts()
    st.sidebar.success("Ingestion & Lakehouse refresh complete!")

st.sidebar.markdown("---")
st.sidebar.subheader("Lakehouse Telemetry")
bronze_files = len(list(settings.BRONZE_DIR.glob("*.parquet")))
st.sidebar.write(f"📁 **Bronze Partitions:** {bronze_files}")

try:
    silver_cnt = engine.query("SELECT COUNT(*) AS c FROM silver_ticks;")[0]["c"]
except Exception:
    silver_cnt = 0
st.sidebar.write(f"✨ **Silver Conformed Ticks:** {silver_cnt:,}")

try:
    gold_cnt = engine.query("SELECT COUNT(*) AS c FROM mart_ohlcv_1m;")[0]["c"]
except Exception:
    gold_cnt = 0
st.sidebar.write(f"🏆 **Gold Mart Bars:** {gold_cnt:,}")

# Top Header Banner
st.title(f"⚡ MarketFlow-AI: {symbol}")
st.caption("Zero-JVM Real-Time Streaming Data Lakehouse (DuckDB + Polars) with Embedded Predictive AI")

# Tabs
tab_live, tab_lakehouse, tab_ai, tab_sql = st.tabs([
    "📈 Live Market Monitor",
    "🏛️ Medallion Architecture",
    "🤖 AI Regime & Copilot",
    "💻 DuckDB SQL Workbench"
])

# ----------------- TAB 1: Live Market Monitor -----------------
with tab_live:
    ticks_data = engine.query(f"""
        SELECT parsed_time, price, bid, ask, spread_bps, volume, trade_side
        FROM silver_ticks
        WHERE symbol = '{symbol}'
        ORDER BY parsed_time DESC
        LIMIT 60;
    """)

    if ticks_data:
        df_ticks = pd.DataFrame(ticks_data)
        latest_price = df_ticks.iloc[0]["price"]
        prev_price = df_ticks.iloc[-1]["price"]
        pct_change = round(((latest_price - prev_price) / prev_price) * 100, 2)
        avg_spread = round(df_ticks["spread_bps"].mean(), 1)
        total_vol = int(df_ticks["volume"].sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Traded Price", f"₹{latest_price:,.2f}", f"{pct_change:+.2f}%")
        col2.metric("Average Spread", f"{avg_spread} bps")
        col3.metric("Recent Volume", f"{total_vol:,} shares")
        col4.metric("Tick Count (Sample)", len(df_ticks))

        st.subheader("Price Action (Recent Ticks)")
        st.line_chart(df_ticks.set_index("parsed_time")["price"])

        st.subheader("Recent Microstructure Stream")
        st.dataframe(df_ticks.head(15), use_container_width=True)
    else:
        st.info("No tick data available yet. Click 'Ingest Live Micro-Batch' in the sidebar to populate!")

# ----------------- TAB 2: Medallion Architecture -----------------
with tab_lakehouse:
    st.subheader("🏛️ Medallion Lakehouse Tiers")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<span class='badge-bronze'>BRONZE LAYER</span>", unsafe_allow_html=True)
        st.write("**Raw Append-Only Parquet Store**")
        st.write("Stores raw micro-batch payloads directly from collectors with zero destructive parsing.")
        st.metric("Partition Files", bronze_files)

    with c2:
        st.markdown("<span class='badge-silver'>SILVER LAYER</span>", unsafe_allow_html=True)
        st.write("**Conformed & Contract-Validated**")
        st.write("Vectorized Polars deduplication, null quarantining, and UTC normalization.")
        st.metric("Conformed Records", f"{silver_cnt:,}")

    with c3:
        st.markdown("<span class='badge-gold'>GOLD LAYER</span>", unsafe_allow_html=True)
        st.write("**High-Performance Analytical Marts**")
        st.write("1-minute OHLCV candles, VWAP, Order Flow Imbalance, and Feature Store.")
        st.metric("1m OHLCV Bars", f"{gold_cnt:,}")

    st.markdown("---")
    st.subheader("1-Minute OHLCV Gold Mart")
    ohlcv = engine.query(f"SELECT * FROM mart_ohlcv_1m WHERE symbol = '{symbol}' ORDER BY bar_time DESC LIMIT 15;")
    if ohlcv:
        st.dataframe(pd.DataFrame(ohlcv), use_container_width=True)
    else:
        st.write("No Gold mart records found. Ingest ticks to build the marts!")

# ----------------- TAB 3: AI Copilot & Regime Predictor -----------------
with tab_ai:
    st.subheader("🤖 Machine Learning Volatility Regime & Anomaly Detection")

    vec = features.get_realtime_feature_vector(symbol)
    pred = ai_model.predict(vec)
    brief = agent.analyze_symbol(symbol)

    a1, a2, a3 = st.columns(3)
    a1.metric("Predicted Volatility Regime", pred["predicted_regime"])
    a2.metric("Model Confidence", f"{pred['confidence'] * 100:.1f}%")
    a3.metric("Anomaly Score", f"{pred['anomaly_score']:.3f}", "ANOMALY" if pred["is_anomaly"] else "NORMAL")

    st.subheader("AI Market Copilot Synthesis")
    st.info(f"**Market Outlook:** {brief['market_outlook']} | **Risk Level:** {brief['risk_level']}")
    st.write(f"💡 **Recommendation:** {brief['recommendation']}")

    st.subheader("Real-Time Feature Vector (Feature Store)")
    st.json(vec)

# ----------------- TAB 4: DuckDB SQL Workbench -----------------
with tab_sql:
    st.subheader("💻 In-Process DuckDB SQL Query Workbench")
    st.caption("Run sub-millisecond SQL queries directly over your Silver & Gold Parquet tiers.")

    default_sql = """
    SELECT 
        symbol,
        COUNT(*) AS tick_count,
        ROUND(AVG(price), 2) AS mean_price,
        ROUND(AVG(spread_bps), 2) AS avg_spread_bps,
        SUM(volume) AS total_volume
    FROM silver_ticks
    GROUP BY symbol
    ORDER BY total_volume DESC;
    """
    user_query = st.text_area("SQL Query", default_sql, height=120)

    if st.button("▶ Run SQL Query"):
        try:
            res = engine.query(user_query)
            if res:
                st.dataframe(pd.DataFrame(res), use_container_width=True)
            else:
                st.write("Query returned 0 rows.")
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")
