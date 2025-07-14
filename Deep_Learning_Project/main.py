import streamlit as st

st.set_page_config(page_title="Youtube Classifier", page_icon=":movie_camera:", layout="wide")

st.markdown("""
<h1 style="text-align: center; color: #000000;">
    Youtube Classifier
</h1>
""", unsafe_allow_html=True)

st.write("This is a simple app that classifies the youtube video into different categories.")

col1, col2 = st.columns([1, 2])
with col1:
    st.write("Enter the YouTube channel name below:")
    channel_name = st.text_input(
        "YouTube Channel Name",
        key="channel_name",
        disabled=bool(st.session_state.get("video_url", ""))
    )

with col2:
    st.write("Enter the YouTube video URL below:")
    video_url = st.text_input(
        "YouTube Video URL",
        key="video_url",
        disabled=bool(st.session_state.get("channel_name", ""))
    )

if st.button("Classify", key="classify_button"):
    if channel_name:
        st.write(f"Classifying {channel_name}...")
    elif video_url:
        st.write(f"Classifying {video_url}...")
    else:
        st.write("Please enter a YouTube channel name or video URL.")




