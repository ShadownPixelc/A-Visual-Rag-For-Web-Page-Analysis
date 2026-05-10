import streamlit as st
import time
from modules.database import init_db
from modules.auth import render_auth
from modules.ui import inject_custom_css
from modules.scraper import scrape_url
from modules.rag import chunk_text, create_vector_store, retrieve_context, generate_answer, generate_quick_summary

# Set page config
st.set_page_config(page_title="Multimodal RAG", page_icon="🤖", layout="wide")

# Initialize Database and CSS globally
init_db()
inject_custom_css()

# Initialize Session State Variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'scraped_text' not in st.session_state:
    st.session_state['scraped_text'] = None
if 'quick_summary' not in st.session_state:
    st.session_state['quick_summary'] = None
if 'vector_index' not in st.session_state:
    st.session_state['vector_index'] = None
if 'text_chunks' not in st.session_state:
    st.session_state['text_chunks'] = None

# ==========================================
# AUTHENTICATION SCREEN
# ==========================================
if not st.session_state['logged_in']:
    render_auth()

# ==========================================
# MAIN DASHBOARD
# ==========================================
else:
    # 30% Left Column, 70% Right Column
    col1, col2 = st.columns([1.2, 2.8], gap="large")

    # ----------------------------------
    # LEFT COLUMN: Inputs & Preview
    # ----------------------------------
    with col1:
        # Subtle Logout Button
        if st.button("Logout", key="logout_btn"):
            st.session_state['logged_in'] = False
            st.session_state['chat_history'] = []
            st.rerun()

        st.markdown("### Add your content!")
        url_input = st.text_input("Enter webpage URL", placeholder="https://...", label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Process URL Button
        if st.button("Start RAG", use_container_width=True):
            if url_input:
                with st.status("Processing webpage...", expanded=True) as status:
                    try:
                        st.write("🌍 Scraping webpage with Firecrawl...")
                        scraped_markdown = scrape_url(url_input)
                        st.session_state['scraped_text'] = scraped_markdown
                        time.sleep(0.5) 
                        
                        st.write("📝 Generating Quick Summary...")
                        st.session_state['quick_summary'] = generate_quick_summary(scraped_markdown)
                        
                        st.write("📄 Processing document structure...")
                        chunks = chunk_text(scraped_markdown)
                        time.sleep(0.5)
                        
                        st.write("🔍 Indexing content with Vector Store...")
                        index_embeddings, raw_embeddings = create_vector_store(chunks)
                        
                        st.session_state['text_chunks'] = chunks
                        st.session_state['vector_index'] = index_embeddings
                        
                        status.update(label="Processing complete!", state="complete", expanded=False)
                        st.success("Ready to Chat!")
                        st.session_state['chat_history'] = [] 
                    except Exception as e:
                        status.update(label="Error processing URL", state="error", expanded=False)
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a URL first.")

        # Document Preview Area
        if st.session_state['scraped_text']:
            st.markdown("### Document Preview")
            if st.session_state['quick_summary']:
                st.info(st.session_state['quick_summary'])
                
            st.markdown(f'<div class="preview-box">{st.session_state["scraped_text"]}</div>', unsafe_allow_html=True)

    # ----------------------------------
    # RIGHT COLUMN: Chat Interface
    # ----------------------------------
    with col2:
        # 1. The Header Area
        header_col1, header_col2 = st.columns([5, 1])
        with header_col1:
            st.markdown("## 🌐 Visual RAG Web Analyzer")
        with header_col2:
            if st.button("Clear ⟳"):
                st.session_state['chat_history'] = []
                st.rerun()
                
        st.markdown("---")
        
        # 2. THE FIX: Chat Input moved to the TOP
        # Wrapping it in a container prevents it from falling to the bottom of the screen
        input_container = st.container()
        with input_container:
            query = st.chat_input("What's up?")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. Chat History Container (Now sits BELOW the input)
        chat_container = st.container(height=500, border=False)
        with chat_container:
            for msg in st.session_state['chat_history']:
                avatar = "🤖" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
                    
        # 4. Processing the Query
        if query:
            if st.session_state['vector_index'] is None:
                st.warning("Please process a URL on the left before asking questions.")
            else:
                # Add user query to history
                st.session_state['chat_history'].append({"role": "user", "content": query})
                
                # Show conversation in the container below
                with chat_container:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(query)
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Thinking..."):
                            context = retrieve_context(
                                query, 
                                st.session_state['vector_index'], 
                                st.session_state['text_chunks']
                            )
                            answer = generate_answer(query, context)
                            st.markdown(answer)
                
                # Save assistant response
                st.session_state['chat_history'].append({"role": "assistant", "content": answer})
                st.rerun()