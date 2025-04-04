
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import PyPDF2
import io
import csv
import re
from io import StringIO
import os
import openai

# Utility: Format check (basic)
def is_valid_key_format(key: str) -> bool:
    return key.startswith("sk-") and len(key) >= 48 and len(key) <= 80

# Utility: Live API check (calls OpenAI to verify the key works)
def is_valid_openai_key_live(key: str) -> bool:
    try:
        openai.api_key = key
        openai.Model.list()  # Lightweight test call
        return True
    except openai.error.AuthenticationError:
        return False
    except Exception as e:
        st.sidebar.error(f"Unexpected error during validation: {e}")
        return False

# API logic
if "api_key_confirmed" not in st.session_state:
    st.session_state.api_key_confirmed = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

st.sidebar.header("🔑 API Key Setup")

if not st.session_state.api_key_confirmed:
    st.session_state.openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    if st.sidebar.button("✅ Confirm API Key"):
        key = st.session_state.openai_api_key.strip()

        if not is_valid_key_format(key):
            st.sidebar.error("❌ Invalid key format. Must start with 'sk-' and be of proper length.")
        elif not is_valid_openai_key_live(key):
            st.sidebar.error("❌ Invalid or unauthorized API key. Please check your key or subscription.")
        else:
            os.environ["OPENAI_API_KEY"] = key
            openai.api_key = key
            st.session_state.api_key_confirmed = True
            st.rerun()
else:
    st.sidebar.success("✅ API Key set and confirmed.")


	
if st.session_state.api_key_confirmed:
	# Lock in mode choice
	if "mode_locked" not in st.session_state:
	    st.session_state.mode_locked = False
	if "chat_mode" not in st.session_state:
	    st.session_state.chat_mode = None
	if "model_confirmed" not in st.session_state:
	    st.session_state.model_confirmed = False  # Initialize model_confirmed if not present
	
	if "mode_locked" not in st.session_state or not st.session_state.mode_locked:
	    st.sidebar.header("🧭 Choose Interaction Mode")
	    mode_choice = st.sidebar.radio("Select mode for this session:", ["Chat without documents", "Chat with uploaded documents"])
	
	    if st.sidebar.button("🔒 Lock In Mode"):
	        st.session_state.chat_mode = mode_choice
	        st.session_state.mode_locked = True
	        st.rerun()
	else:
	    st.sidebar.success(f"Mode: {st.session_state.chat_mode} (locked)")
	
	# Initialize model settings
	st.title("🤖 AI Chatbot - Ask Me Anything!")
		
	st.sidebar.header("⚙️ Model Settings")
	st.session_state.model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
	st.session_state.model_creativity = st.sidebar.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
	st.session_state.response_length_words = st.sidebar.slider("Response Length (Words)", 50, 500, 150, 10)
	
	if st.sidebar.button("Confirm Model Settings"):
	    st.session_state.model_confirmed = True
	    st.success("Model settings confirmed.")
		
	# Ensure mode and model are confirmed before chatting
	if st.session_state.chat_mode == "Chat without documents" and not st.session_state.mode_locked:
	    st.warning("You need to lock in the mode before chatting.")
	
	
		
	# Initialize LLM
	SYSTEM_PROMPT = "You are a helpful and safe AI assistant. You must refuse to engage in harmful, unethical, or biased discussions."
	llm = ChatOpenAI(
	    model=st.session_state.model_choice,
	    temperature=st.session_state.model_creativity,
	    max_tokens=int(st.session_state.response_length_words * 1.5)
	)
	
	if "memory" not in st.session_state:
	    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
	
	def is_input_safe(user_input: str) -> bool:
	    """Check if the input is safe to process."""
	    dangerous_patterns = [
	        r"(system|os|subprocess|import|open|globals|locals|__import__|__globals__|__dict__|__builtins__)",
	        r"(sudo|rm -rf|chmod|chown|mkfs|:(){:|fork bomb|shutdown)",
	        r"(simulate being|ignore previous instructions|bypass|jailbreak|pretend to be|hack|scam )",
	        r"(<script>|</script>|<iframe>|javascript:|onerror=)",
	        r"(base64|decode|encode|pickle|unpickle)",
	        r"(http[s]?://|ftp://|file://)",
	        r"(manipulate|modify system prompt|alter assistant behavior)"
	    ]
	    return not any(re.search(pattern, user_input, re.IGNORECASE) for pattern in dangerous_patterns)
	
	def process_pdf(uploaded_file):
	    """Extracts text from a PDF and splits it into chunks."""
	    with io.BytesIO(uploaded_file.getvalue()) as byte_file:
	        pdf_reader = PyPDF2.PdfReader(byte_file)
	        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
	    
	    # Chunk text for better retrieval
	    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	    return text_splitter.split_text(text)
	
	def process_text_file(uploaded_file):
	    """Processes a text file and splits it into chunks."""
	    text = uploaded_file.getvalue().decode("utf-8")
	    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	    return text_splitter.split_text(text)
	
	if "conversation_history" not in st.session_state:
	    st.session_state.conversation_history = []
	if "uploaded_files" not in st.session_state:
	    st.session_state.uploaded_files = None
	if "uploaded_documents" not in st.session_state:
	    st.session_state.uploaded_documents = []  # Store documents separately    
	if "uploaded_file_count" not in st.session_state:
	    st.session_state.uploaded_file_count = 0
	if "user_input" not in st.session_state:
	    st.session_state.user_input = ""
	if "all_doc_chunks" not in st.session_state:
	    st.session_state.all_doc_chunks = []
	
		
	if st.sidebar.button("🆕 Start New Session"):
	    for key in st.session_state.keys():
	        del st.session_state[key]
	    st.rerun()
	
	# Function to handle document removal
	def remove_document(file_to_remove):
	    """Remove a document and update the FAISS index."""
	    uploaded_files_list = [file for file in st.session_state.uploaded_documents if file.name != file_to_remove.name]
	    
	    docs = []
	    for f in uploaded_files_list:
	        if f.type == "application/pdf":
	            docs.extend(process_pdf(f))  # Process PDF file and add chunks
	        else:
	            docs.extend(process_text_file(f))  # Process TXT file and add chunks
	
	    if docs:  # ✅ Only create FAISS index if there is something to index
	        embeddings = OpenAIEmbeddings()
	        faiss_index = FAISS.from_texts(docs, embeddings)
	        st.session_state.uploaded_files = faiss_index
	    else:
	        st.session_state.uploaded_files = None  # No files left, clear the index
	
	    st.session_state.uploaded_documents = uploaded_files_list
	    st.session_state.uploaded_file_count = len(uploaded_files_list)
	
	
	# Display file uploader only if mode is 'Chat with uploaded documents'
	if st.session_state.chat_mode == "Chat with uploaded documents":
	    st.sidebar.header("📄 Upload Documents")
	    uploaded_files = st.sidebar.file_uploader(
	        "Upload PDFs or TXT files", type=["pdf", "txt"],
	        accept_multiple_files=True, key="file_uploader"
	    )
	else:
	    uploaded_files = None  # Prevent uploads in chat-only mode
	
	
	# --- Detect file changes and update memory + FAISS ---
	def file_key(file):
	    return (file.name, file.size)
	
	uploaded_now = [file_key(f) for f in uploaded_files] if uploaded_files else []
	stored_before = [file_key(f) for f in st.session_state.get("uploaded_documents", [])]
	
	# If the file list has changed (added or removed)
	if uploaded_now != stored_before:
	    st.session_state.uploaded_documents = uploaded_files or []
	    st.session_state.uploaded_file_count = len(uploaded_files or [])
	
	    if uploaded_files:
	        with st.spinner("Rebuilding document index..."):
	            all_chunks = []
	            for f in uploaded_files:
	                if f.type == "application/pdf":
	                    all_chunks.extend(process_pdf(f))
	                else:
	                    all_chunks.extend(process_text_file(f))
	
	            all_chunks = [str(c) for c in all_chunks]
	            embeddings = OpenAIEmbeddings()
	            faiss_index = FAISS.from_texts(all_chunks, embeddings)
	
	            st.session_state.uploaded_files = faiss_index
	            st.success(f"Updated index with {len(all_chunks)} chunks.")
	    else:
	        # No files uploaded anymore — clear FAISS index
	        st.session_state.uploaded_files = None
	        st.info("All documents removed. Index cleared.")
	
			
	# Display conversation history
	for message in st.session_state.memory.chat_memory.messages:
	    if isinstance(message, HumanMessage):
	        st.chat_message("user").markdown(message.content)
	    elif isinstance(message, AIMessage):
	        st.chat_message("assistant").markdown(message.content)
	
	if st.session_state.mode_locked and st.session_state.model_confirmed:
	    query = st.chat_input("Ask a question:")
	
	    if query:
	        if is_input_safe(query):
	            if st.session_state.uploaded_files:
	                retriever = st.session_state.uploaded_files.as_retriever(search_kwargs={"k": 2})
	                qa_chain = ConversationalRetrievalChain.from_llm(
	                    llm=llm, retriever=retriever, memory=st.session_state.memory
	                )
	                response = qa_chain.run(query)
	                if not any(msg.content == response for msg in st.session_state.memory.chat_memory.messages):
	                    st.session_state.memory.chat_memory.add_ai_message(response)
	                st.chat_message("assistant").write(response)
	            else:
	                system_message = SystemMessage(content=SYSTEM_PROMPT)
	                user_message = HumanMessage(content=query)
	                
	                messages = st.session_state.memory.chat_memory.messages
	                response = llm.invoke(
	                    messages + [system_message, user_message]
	                ) 
	                st.session_state.memory.chat_memory.add_ai_message(response.content)
	                st.chat_message("assistant").write(response.content)
	
	        else:
	            warning = "⚠️ Your query violates content policies."
	            st.session_state.memory.chat_memory.add_ai_message(warning)
	            st.chat_message("assistant").write(warning)
	
	else:
	    st.warning("Lock the mode and confirm model settings before asking questions.")
	
	
	# Function to save conversation as CSV
	def save_conversation_csv():
	    output = StringIO()
	    writer = csv.writer(output)
	    writer.writerow(["Role", "Message"])
	
	    for msg in st.session_state.memory.chat_memory.messages:
	        if isinstance(msg, HumanMessage):
	            writer.writerow(["User", msg.content])
	        elif isinstance(msg, AIMessage):
	            writer.writerow(["Assistant", msg.content])
	
	    return output.getvalue()
	
	st.sidebar.header("💾 Download Conversation")
	st.sidebar.download_button("Download CSV", save_conversation_csv(), "conversation.csv", "text/csv")

else:
    st.warning("Please enter and confirm your OpenAI API key to start.")


