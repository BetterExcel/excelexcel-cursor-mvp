# Excel‑Cursor MVP

An Excel‑like web application with AI-powered spreadsheet assistant built with Streamlit and OpenAI GPT-4-Turbo.

## Features

- 📊 **Interactive Spreadsheet**: Editable grid with multi-sheet support
- 🧮 **Formula Support**: Excel-like formulas (`=SUM(A1:A10)`, `=AVERAGE`, `=MIN`, `=MAX`, etc.)
- 📈 **Data Operations**: Sort, filter, and visualize data
- 📁 **Import/Export**: Support for CSV and XLSX files
- 📊 **Charts**: Create line and bar charts with matplotlib
- 🤖 **AI Agent**: GPT-4-Turbo powered assistant that can:
  - Set and get cell values
  - Apply formulas to cells or columns
  - Sort and filter data
  - Create new sheets
  - Generate charts
  - Export data in various formats

## Prerequisites

- Python 3.10+
- OpenAI API key with GPT-4-Turbo access

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd excelexcel-cursor-mvp
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenAI API Key**
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY="your-openai-api-key-here"
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8501`

## Usage

### Manual Operations
- **Import Data**: Use the file uploader to import CSV or XLSX files
- **Sort Data**: Enter column name and choose ascending/descending
- **Filter Data**: Filter rows by column value equality
- **Create Charts**: Specify X and Y columns for visualization
- **Formulas**: Use the formula bar to apply Excel-like formulas

### AI Assistant
Use the sidebar chat to interact with the AI agent:

- *"Sort column A in ascending order"*
- *"Create a chart showing B vs C"*
- *"Calculate the sum of column D and put it in E1"*
- *"Add a new column F with the average of A and B"*
- *"Export the current sheet as CSV"*

## Project Structure

```
excelexcel-cursor-mvp/
├── app/
│   ├── agent/          # AI agent implementation
│   ├── services/       # Core spreadsheet logic
│   ├── ui/            # UI components and formulas
│   └── charts.py      # Chart generation
├── data/              # Sample data files
├── streamlit_app.py   # Main application entry point
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, Pandas
- **AI**: OpenAI GPT-4-Turbo with function calling
- **Charts**: Matplotlib
- **File Handling**: Pandas (CSV), OpenPyXL (Excel)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Note

This is an MVP (Minimum Viable Product) demonstration and not a full Excel replacement. The formula engine supports basic operations and is intentionally simplified.