import streamlit as st
from data_loader import (
    get_available_dates,
    get_files_for_date,
    get_timezones,
    load_json,
    format_date,
    build_file_path
)

st.set_page_config(
    page_title='Mastodon daily topics',
    layout='centered'
)

st.markdown('''
    <style>
    .block-container {
        max-width: 950px;
        padding-top: 1rem;
    }
    h1 {
        text-align: center;
        font-size: 2.5rem !important;
    }
    h3 {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    h5 {
        text-align: center;
        margin-top: -1rem !important;
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stCaptionContainer"] p {
        text-align: center;
        font-size: 1rem !important;
    }
    </style>
''', unsafe_allow_html=True)

# Escape dollars to be rendered as text instead of LaTeX
def escape_dollars(text):
    return text.replace('$', '\\$')

st.title('What are people talking about on Mastodon today?')
st.markdown('##### A daily summary of the most popular themes in english-language posts on mastodon.social')

# Get available options
dates = get_available_dates()
timezones = get_timezones(dates[0])

# Dropdown columns
col1, col2 = st.columns(2)
with col1:
    selected_date = st.selectbox(
        '**Date**',
        options=dates,
        format_func=format_date,
        index=0
    )
with col2:
    default_tz = 'Europe/Madrid' if 'Europe/Madrid' in timezones else timezones[0]
    selected_timezone = st.selectbox(
        '**Timezone**',
        options=timezones,
        index=timezones.index(default_tz)
    )
    
st.markdown('') # Spacer

selected_file_path = build_file_path(selected_date, selected_timezone)

if not selected_file_path.exists():
    st.warning(f'No data available for this combination')
    st.stop()

topics = load_json(selected_file_path)

# Display topics
for rank, topic_data in topics.items():
    with st.container(border=True):

        # Title
        st.subheader(topic_data['title'])
        
        # Stats
        st.markdown(f"**{topic_data['total_documents']} posts / {topic_data['total_reblogs']} reblogs**")
        
        # Summary
        st.markdown(escape_dollars(topic_data['summary']))
        
        # Expandable details
        with st.expander('View details'):
            for section in topic_data['description']:
                st.markdown(f"**{section['section_title']}**")
                st.markdown(escape_dollars(section['section_content']))

st.markdown('') # Spacer

# Footer
data_info = "Data from [mastodon.social](https://mastodon.social)"
topic_info = "Topic modeling using [BERTopic](https://maartengr.github.io/BERTopic)"
llm_info = "Summaries by [Qwen3](https://qwen.ai)"
github = "View the project's code on [Github](https://github.com/terencicp/mastodon-topics)"
st.caption(f"{data_info} / {topic_info} / {llm_info} / {github}")
