

# TalktoVideo

TalktoVideo is a Streamlit-based application that allows users to extract, summarize, and chat with YouTube videos. It enables users to transcribe audio from YouTube videos, generate summaries, and ask context-based questions using a language model.

## Features
- Download and extract audio from YouTube videos.
- Transcribe the extracted audio using AssemblyAI.
- Store and process transcription for effective retrieval.
- Generate a concise summary of the transcription.
- Enable users to ask questions and receive context-aware answers using an LLM-powered chatbot.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd TalktoVideo
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On macOS/Linux
   myenv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables by creating a `.env` file:
   ```plaintext
   ASSEMBLY_AI_KEY=your_assemblyai_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage
Run the application with:
```bash
streamlit run app.py
```

## Steps Involved
1. **Transcribing Audio from YouTube Videos:**
   - Uses `yt_dlp` to download and extract audio.
   - Transcribes the extracted audio using AssemblyAI.
2. **Processing and Organizing Content for Effective Retrieval:**
   - Stores transcriptions in a structured format.
   - Splits text into chunks for efficient vector search.
3. **Creating a Summary for the Entire Transcript:**
   - Uses an LLM model to generate a concise summary.
4. **Enabling Users to Query and Receive Context-Based Answers:**
   - Implements FAISS-based vector retrieval.
   - Uses an OpenAI-powered chatbot to respond to queries.

## Technologies Used
- Python
- Streamlit
- yt_dlp
- AssemblyAI
- OpenAI API
- FAISS (Facebook AI Similarity Search)
- LangChain

## License
This project is licensed under the MIT License.


