import streamlit as st

from agents.pdf_agent import process_pdf
from agents.web_agent import process_website

from rag.vector_store import create_vector_store
from rag.retriever import get_retriever

from utils.llm import load_llm


from utils.evaluation import evaluate_rag


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Agentic Multimodal RAG Assistant",
    layout="wide"
)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🤖 Agentic Multimodal RAG Assistant")

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------
# PDF + WEBSITE ROW
# ------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"]
    )

with col2:

    website_url = st.text_input(
        "🌐 Website URL"
    )

# ------------------------------------------------
# CLEAR CHAT
# ------------------------------------------------

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# ------------------------------------------------
# PDF PROCESSING
# ------------------------------------------------

if uploaded_file:

    file_path = f"uploads/{uploaded_file.name}"

    with open(
        file_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    chunks = process_pdf(
        file_path
    )

    create_vector_store(
        chunks
    )

    st.success(
        "✅ PDF Processed Successfully"
    )

# ------------------------------------------------
# WEBSITE PROCESSING
# ------------------------------------------------

if website_url:

    chunks = process_website(
        website_url
    )

    create_vector_store(
        chunks
    )

    st.success(
        "✅ Website Processed Successfully"
    )

# ------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )

# ------------------------------------------------
# CHAT INPUT
# ------------------------------------------------

question = st.chat_input(
    "Ask anything about your PDF or Website..."
)

# ------------------------------------------------
# QUESTION PROCESSING
# ------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    retriever = get_retriever()

    if retriever is None:

        answer = (
            "Please upload a PDF or Website first."
        )

    else:

        try:

            with st.spinner(
                "🤔 Searching and generating answer..."
            ):

                docs = retriever.invoke(
                    question
                )

                context = "\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )

                llm = load_llm()

                prompt = f"""
You are an intelligent RAG assistant.

Rules:

1. Answer ONLY from the provided context.

2. If the answer is not found in the context,
say:

"I could not find this information in the uploaded data."

3. Give structured answers.

4. Use bullet points whenever possible.

Context:
{context}

Question:
{question}
"""

                response = llm.invoke(
                    prompt
                )

                try:
                    answer = response.text
                except:
                    answer = str(
                        response.content
                    )

                # --------------------------------
                # RAG EVALUATION
                # --------------------------------

                try:

                    ground_truth = context

                    evaluation = evaluate_rag(
                        question,
                        answer,
                        context,
                        ground_truth
                    )

                except Exception as eval_error:

                    evaluation = None

                    st.warning(
                        f"Evaluation Error: {eval_error}"
                    )

        except Exception as e:

            answer = (
                f"Error: {str(e)}"
            )

            evaluation = None

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            "### 🤖 Answer"
        )

        st.write(
            answer
        )

    # --------------------------------
    # DISPLAY RAG METRICS
    # --------------------------------

    if evaluation is not None:

        try:

            scores = evaluation.to_pandas()

            st.markdown(
                "## 📊 RAG Evaluation Metrics"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Faithfulness",
                    round(
                        scores["faithfulness"][0],
                        2
                    )
                )

            with col2:

                st.metric(
                    "Answer Relevancy",
                    round(
                        scores["answer_relevancy"][0],
                        2
                    )
                )

            with col3:

                st.metric(
                    "Context Precision",
                    round(
                        scores["context_precision"][0],
                        2
                    )
                )

            with col4:

                st.metric(
                    "Context Recall",
                    round(
                        scores["context_recall"][0],
                        2
                    )
                )

        except Exception as metric_error:

            st.warning(
                f"Metric Display Error: {metric_error}"
            )