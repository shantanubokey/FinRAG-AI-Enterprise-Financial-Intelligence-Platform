"""
Streamlit frontend for the Financial RAG System.
Professional dashboard with chat, upload, citations, and evaluation.
"""

import streamlit as st

st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Sidebar
    with st.sidebar:
        st.title("📊 Financial AI")
        st.divider()

        page = st.radio(
            "Navigation",
            ["Chat", "Upload Documents", "Analytics", "Evaluation Dashboard"],
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Filters")
        company_filter = st.multiselect(
            "Company",
            ["Apple", "Microsoft", "Amazon", "Google", "Tesla", "NVIDIA"],
        )
        year_filter = st.multiselect(
            "Year",
            list(range(2024, 2018, -1)),
        )
        filing_filter = st.multiselect(
            "Filing Type",
            ["10-K", "10-Q", "earnings_call", "investor_presentation"],
        )

        st.divider()
        retrieval_mode = st.select_slider(
            "Retrieval Mode",
            options=["sparse", "hybrid", "dense"],
            value="hybrid",
        )
        use_reranker = st.toggle("Cross-Encoder Re-ranking", value=True)

    # Route pages
    if page == "Chat":
        _render_chat(company_filter, year_filter, filing_filter, retrieval_mode, use_reranker)
    elif page == "Upload Documents":
        _render_upload()
    elif page == "Analytics":
        _render_analytics()
    elif page == "Evaluation Dashboard":
        _render_evaluation()


def _render_chat(company_filter, year_filter, filing_filter, retrieval_mode, use_reranker):
    st.title("Financial AI Assistant")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 {len(msg['citations'])} Sources"):
                    for i, cit in enumerate(msg["citations"], 1):
                        confidence = cit.get("confidence_score", 0)
                        color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                        st.markdown(
                            f"{color} **[{i}]** {cit.get('company', 'Unknown')} | "
                            f"{cit.get('filing_type', 'N/A')} | "
                            f"{cit.get('year', 'N/A')} | "
                            f"Page {cit.get('page_number', 'N/A')} | "
                            f"Confidence: {confidence:.0%}"
                        )
                        st.caption(cit.get("chunk_content", "")[:300])

            if msg.get("metrics"):
                m = msg["metrics"]
                cols = st.columns(5)
                cols[0].metric("Latency", f"{m.get('total_latency_ms', 0):.0f}ms")
                cols[1].metric("Tokens", m.get("total_tokens", 0))
                cols[2].metric("Cost", f"${m.get('estimated_cost_usd', 0):.4f}")
                cols[3].metric("Cache", "✅ Hit" if m.get("cache_hit") else "❌ Miss")
                if msg.get("evaluation"):
                    ev = msg["evaluation"]
                    cols[4].metric(
                        "Faithfulness",
                        f"{ev.get('faithfulness', 0):.0%}" if ev.get("faithfulness") else "N/A",
                    )

    # Input
    if question := st.chat_input("Ask about financial documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing financial documents..."):
                response = _call_api(
                    question=question,
                    company_filter=company_filter,
                    year_filter=year_filter,
                    filing_filter=filing_filter,
                    retrieval_mode=retrieval_mode,
                    use_reranker=use_reranker,
                )

            if response and not response.get("error"):
                st.markdown(response.get("answer", "No answer generated."))
                msg_data = {
                    "role": "assistant",
                    "content": response.get("answer", ""),
                    "citations": response.get("citations", []),
                    "metrics": response.get("metrics", {}),
                    "evaluation": response.get("evaluation"),
                }
                st.session_state.messages.append(msg_data)
                st.rerun()
            else:
                error_msg = response.get("error", "Unknown error") if response else "API unavailable"
                st.error(f"Error: {error_msg}")


def _render_upload():
    st.title("Upload Financial Documents")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "docx", "csv", "xlsx"],
            help="Supported: PDF, DOCX, CSV, Excel",
        )

    with col2:
        company = st.text_input("Company Name", placeholder="e.g. Apple")
        ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL")
        year = st.number_input("Year", min_value=1990, max_value=2030, value=2024)
        filing_type = st.selectbox(
            "Filing Type",
            ["10-K", "10-Q", "earnings_call", "investor_presentation", "other"],
        )

    if uploaded_file and st.button("Upload & Process", type="primary"):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            result = _upload_document(uploaded_file, company, ticker, year, filing_type)
            if result:
                st.success(f"Document queued for processing. ID: {result.get('id')}")
                st.json(result)
            else:
                st.error("Upload failed. Check API connection.")


def _render_analytics():
    st.title("Analytics Dashboard")
    st.info("Connect to the monitoring stack to see live metrics.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", "—")
    col2.metric("Total Queries", "—")
    col3.metric("Avg Latency", "—")
    col4.metric("Total Cost", "—")

    st.subheader("Grafana Dashboard")
    st.markdown(
        "View live metrics at [Grafana](http://localhost:3000) → Financial RAG dashboard"
    )


def _render_evaluation():
    st.title("Evaluation Dashboard")
    st.info("Auto-evaluation scores from Ragas, tracked per query.")

    import pandas as pd
    # Placeholder data — replace with real API call
    sample_data = pd.DataFrame({
        "Query": ["What was Apple revenue in 2023?", "Compare NVIDIA and AMD margins"],
        "Faithfulness": [0.92, 0.85],
        "Context Precision": [0.88, 0.79],
        "Answer Relevancy": [0.95, 0.91],
        "Latency (ms)": [1250, 1890],
    })
    st.dataframe(sample_data, use_container_width=True)


def _call_api(question, company_filter, year_filter, filing_filter, retrieval_mode, use_reranker):
    import requests

    api_url = st.secrets.get("api_url", "http://localhost:8000/api/v1")
    api_key = st.secrets.get("api_key", "dev-key")

    try:
        resp = requests.post(
            f"{api_url}/query/",
            json={
                "question": question,
                "company_filter": company_filter or None,
                "year_filter": year_filter or None,
                "filing_type_filter": filing_filter or None,
                "retrieval_mode": retrieval_mode,
                "use_reranker": use_reranker,
                "top_k": 5,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _upload_document(file, company, ticker, year, filing_type):
    import requests

    api_url = st.secrets.get("api_url", "http://localhost:8000/api/v1")
    api_key = st.secrets.get("api_key", "dev-key")

    try:
        resp = requests.post(
            f"{api_url}/ingest/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            data={
                "company": company, "ticker": ticker,
                "year": year, "filing_type": filing_type,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    main()
