import queue
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph_backend_tools_RAG_ticket_routing import chatbot, retrieve_all_threads, submit_async_task, add_uploaded_document, chat_title_func


# =========================== Utilities ===========================
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# ======================= Session Initialization ===================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state or 'thread_titles' not in st.session_state:
    threads_with_titles = retrieve_all_threads()  # {thread_id: chat_title or None}
    st.session_state['chat_threads'] = list(threads_with_titles.keys())
    st.session_state['thread_titles'] = {
        tid: title for tid, title in threads_with_titles.items() if title
    }

add_thread(st.session_state['thread_id'])


# ============================ Sidebar ============================
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

if 'rag_processed_files' not in st.session_state:
    st.session_state['rag_processed_files'] = set()

uploaded_file = st.sidebar.file_uploader('Upload a file', type=['pdf', 'docx'], key='file_uploader')
if uploaded_file is not None:
    # Streamlit reruns this whole script on every interaction (including every
    # chat message), and the uploader keeps returning the same file across
    # reruns until it's cleared/replaced. Without this guard we'd re-embed
    # and re-add the same document's chunks on every single message.
    file_key = getattr(uploaded_file, 'file_id', None) or f"{uploaded_file.name}-{uploaded_file.size}"
    if file_key not in st.session_state['rag_processed_files']:
        with st.sidebar.status(f"Indexing `{uploaded_file.name}` …", expanded=False) as status:
            try:
                num_chunks = add_uploaded_document(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state['rag_processed_files'].add(file_key)
                status.update(label=f"Indexed `{uploaded_file.name}` ({num_chunks} chunks)", state='complete')
            except Exception as exc:
                status.update(label=f"Failed to index `{uploaded_file.name}`: {exc}", state='error')

st.sidebar.header('My Conversations')


for thread_id in st.session_state['chat_threads'][::-1]:
    chat_title = st.session_state['thread_titles'].get(thread_id, str(thread_id))

    if st.sidebar.button(chat_title, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# ============================ Main UI ============================

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    graph_input = {'messages': [HumanMessage(content=user_input)]}
    if st.session_state['thread_id'] not in st.session_state['thread_titles']:
        title = chat_title_func(user_input)
        st.session_state['thread_titles'][st.session_state['thread_id']] = title
        # Include chat_title in the graph input so LangGraph writes it into
        # this thread's checkpoint (chatbot.db) alongside the messages,
        # instead of it only living in Streamlit's in-memory session_state.
        graph_input['chat_title'] = title

    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'metadata': {'thread_id': st.session_state['thread_id']},
        'run_name': 'chatbot_turn',
        'recursion_limit': 5,
    }

    # Assistant streaming block
    with st.chat_message('assistant'):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {'box': None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        graph_input,
                        config=CONFIG,
                        stream_mode='messages',
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(('error', exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == 'error':
                    raise metadata

                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, 'name', 'tool')
                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder['box'].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state='running',
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder['box'] is not None:
            status_holder['box'].update(
                label=' Tool finished', state='complete', expanded=False
            )

    # Save assistant message
    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message}
    )