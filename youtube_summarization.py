import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader,WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


def extract_transcript_details(youtube_video_url):
    try:
        video_id = get_video_id(youtube_video_url)

        if not video_id:
            return None

        transcript = YouTubeTranscriptApi().fetch(video_id)

        transcript_text = " ".join(
            snippet.text for snippet in transcript
        )

        return transcript_text

    except Exception as e:
        print("Transcript error:", e)
        return None
    
import urllib.parse as urlparse

def get_video_id(url):
    parsed_url = urlparse.urlparse(url)
    
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        query = urlparse.parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    
    return None

groq_api_key = st.secrets["GROQ_API_KEY"]

## sstreamlit APP
st.set_page_config(page_title="Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 URL Content Summarization")
st.subheader('Summarize URL')


generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)
prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""

prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                    
                    if generic_url:
                        video_id = get_video_id(generic_url)
                        print(video_id)
                        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width='stretch')

                    transcript_text = extract_transcript_details(generic_url)

                    template="""You are Yotube video summarizer. You will be taking the transcript text
                    and summarizing the entire video and providing the important summary in points
                    within 250 words. Please provide the summary of the text given here: {transcript} """

                    prompt = PromptTemplate(template=template,input_variables=['transcript'])

                    chain = prompt | llm | StrOutputParser()

                    if transcript_text:
                        summary=chain.invoke({"transcript": transcript_text})
                        st.markdown("## Detailed Notes:")
                        st.success(summary)



                else:
                    loader = WebBaseLoader(generic_url)
                      
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=200
                    )

                    split_docs = text_splitter.split_documents(docs)

                    chain = prompt | llm | StrOutputParser()

                    summaries = []

                    for doc in split_docs:
                        summary = chain.invoke({"text": doc.page_content})
                        summaries.append(summary)

                    final_summary = " ".join(summaries)

                    st.success(final_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")
                    



