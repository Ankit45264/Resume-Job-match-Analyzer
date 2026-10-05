import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download nltk resources 


nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")


# Page Setup 
st.set_page_config(page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTR4RiABlsiGgRrXNC7PC0_acl9tGA6X3bKwY_ONK0vCe4w3BT-VN4Z8Qb&s=10",page_title="Resume Match Scorer",layout='wide')

st.title("Resume job Match Analysis-")
st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match!

This tool users **TF_IDF + Cosine Similarity** to analyze your resume against job requirements.
""")

with st.sidebar:
    st.header("About")
    st.info("""

    This tool helps you:
    - Measures how your resume matches a job description.
    - Identify important job keywords.
    - Improve your resume baesd on missing terms.

    """)
    st.header("How it works")
    st.write("""

    This tool helps you:
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze Match**
    4. Review score & suggestion

    """)

'''Create Important Function''' 


# Extract text from the PDF 
 
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF:{e}")
        return ""

# Clean the text 


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Remove Stopwords 

def remove_stopwords(text):
    stopword =stopwords.words('english')
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stopword])


def calculate_similarity(resume_text,job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text,job_description])
    score = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]*100
    return round(score,2)



# main App

def main():
    uploaded_file = st.file_uploader("Upload your resume (PDF)",type=['pdf','docx'])
    job_description =  st.text_area("Paste the job description",height=200)


    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume")
            return
        if not job_description:
            st.warning("Please paste job description")
            return

        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from pdf. Please try another pdf")
                return

            # Calculate similarity 

            similarity_score = calculate_similarity(resume_text,job_description)


            # Result 

            st.subheader("Results")
            st.metric("Match Score", f"{similarity_score:.2f}%")


            # Gauge Chart 

            fig,ax=plt.subplots(figsize=(5,0.3))
            colors=['red','orange','green']
            color_index=min(int(similarity_score//33),2)
            ax.barh([0],[similarity_score],color=colors[color_index])
            ax.set_xlim(0,100)
            ax.set_xlabel("Match percentage")
            ax.set_yticks([])
            ax.set_title("Resume Job Match")
            st.pyplot(fig)



            if similarity_score<40:
                st.warning("Low Match")

            elif similarity_score<70:
                st.info("Good Match. Your resume align fairly well")
            else:
                st.success("Excellent Match! Your resume strongly aligns.")

        

if __name__=="__main__":
    main()

