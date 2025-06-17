# Ask Questions from PDF (Groq + Mixtral)

🔗 **Live Demo**: [Click here to try the app](https://aipoweredapppdfchatapp-5r7wreedp2cyxfhqpenoey.streamlit.app/)

This project is a Streamlit-based application that allows users to upload a PDF file and ask questions based on its content. The app uses **PyPDF2** to extract text from the PDF, **SentenceTransformers** for embedding, **FAISS** for similarity search, and **Groq + Mixtral** for generating answers based on the extracted context.
 
## Features

- Upload a PDF file.
- Extract and process text into manageable chunks.
- Use **FAISS** to find the most relevant chunks for a given question.
- Query **Groq + Mixtral** to generate answers based on the extracted context.
- Simple and interactive UI built with **Streamlit**.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PradeepPradeep1234/AI_powered_QA_PDF_Chatapp.git
   cd AI_powered_QA_PDF_Chatapp

2. Install the required dependencies:

   pip install -r requirements.txt

3. Set up your secrets:

   1. Create a .streamlit/secrets.toml file in the project directory.
   2. Add your Hugging Face token and Groq API key:
   3. HF_TOKEN = "your_huggingface_token"
   4. GROQ_API_KEY = "your_groq_api_key"
