 # LLM Summarizer

**Harness the power of LLMs to extract concise summaries from extensive text.**

## Installation

1. **Create a virtual environment:**
   - Navigate to the project directory in your terminal.
   - Create a virtual environment: `python -m venv env`
   - Activate the environment:
     - Windows: `env\Scripts\activate.bat`
     - Linux/macOS: `source env/bin/activate`

2. **Install dependencies:**
   - Install required packages: `pip install -r requirements.txt`

3. **Set environment variable:**
   - Set your Hugging Face API token:
     `export HUGGINFACEHUB_API_TOKEN=YOUR_API_TOKEN`

**## Usage**

1. **Run the main script:**
   - Execute the main script: `python main.py`

2. **Customize content (optional):**
   - To summarize different text, modify the following in the code:
     - `pdf_file`: Path to the PDF file you want to summarize.
     - `query`: The specific question or prompt to guide the summary.

**## Additional Information**

- **Key frameworks:** LangChain, Hugging Face LLMs
- **Supported document types:** PDF (currently)
- **Customization:** Adjust prompts and code for different content and queries.

**## Contributing**

We welcome contributions! Please see the CONTRIBUTING.md file for guidelines.

**## License**

This project is licensed under the MIT License. See the LICENSE file for details.

**## Contact**

For any questions or feedback, reach out to [Your Name or Project Contact Information]
