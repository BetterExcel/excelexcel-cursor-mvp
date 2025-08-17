# Excel‑Cursor MVP - Enhanced AI Spreadsheet

A powerful Excel‑like web application with advanced AI-powered spreadsheet assistant built with Streamlit and OpenAI GPT-4-Turbo. Features a modern ribbon interface, robust file management, and comprehensive formula support.

## ✨ Features

### 🎨 **Modern Interface**
- **Ribbon-style Navigation**: Familiar Excel-like tabs (Home, Insert, Formulas, Data, Review)
- **Theme Support**: Auto, Light, and Dark themes with comprehensive styling
- **Multi-sheet Support**: Create, switch, and manage multiple spreadsheet tabs
- **Excel-style Grid**: Professional data editor with proper row/column numbering

### 📊 **Advanced Spreadsheet Operations**
- **Interactive Data Editor**: Click-to-edit cells with auto-save functionality
- **Smart Column Types**: Automatic detection of numeric, date, and text columns
- **Formula Bar**: Enhanced formula input with help documentation and examples
- **Bulk Operations**: Apply formulas to entire columns or individual cells

### 🧮 **Comprehensive Formula Engine**
- **Mathematical**: `=A1+B1`, `=SUM(A1:A10)`, `=AVERAGE(A1:A10)`, `=MIN()`, `=MAX()`
- **Statistical**: `=COUNT()`, `=STDEV()`, `=VAR()`, `=MEDIAN()`, `=MODE()`
- **Text Functions**: `=CONCATENATE()`, `=LEFT()`, `=RIGHT()`, `=MID()`, `=LEN()`, `=UPPER()`, `=LOWER()`
- **Date/Time**: `=TODAY()`, `=NOW()`, `=WEEKDAY()`, `=YEAR()`, `=MONTH()`, `=DAY()`
- **Lookup**: `=VLOOKUP()`, `=HLOOKUP()`, `=INDEX()`, `=MATCH()`

### � **Robust File Management**
- **Smart Upload System**: Duplicate prevention with session state tracking
- **Auto-save**: Automatic backups with timestamp management
- **File Selection**: Browse and load previously saved files
- **Multi-format Support**: CSV and XLSX import/export
- **Data Directory Management**: Organized file storage with cleanup utilities

### � **Data Analysis & Visualization**
- **Interactive Charts**: Line charts, bar charts, and scatter plots with Plotly
- **Quick Statistics**: Instant statistical analysis with visual summaries
- **Sort & Filter**: Advanced data manipulation with multiple criteria
- **Pivot Tables**: Data aggregation and analysis (coming soon)

### 🤖 **AI-Powered Assistant**
- **GPT-4-Turbo Integration**: Advanced AI agent with function calling
- **Natural Language Processing**: Conversational interface for spreadsheet operations
- **Contextual Understanding**: AI knows current sheet structure and data
- **Operation Logging**: Track all AI operations with timestamps
- **Model Selection**: Support for multiple OpenAI models

### 🛠 **Development Features**
- **Comprehensive Testing**: 19 unit tests covering all core functionality
- **Professional Environment**: Modern `.venv` setup with proper dependency management
- **Error Handling**: Robust error management with user-friendly messages
- **Performance Optimization**: Efficient data processing and memory management

## Prerequisites

- Python 3.10+
- OpenAI API key with GPT-4-Turbo access

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/paramshah07/excelexcel-cursor-mvp.git
   cd excelexcel-cursor-mvp
   ```

2. **Set up Python environment**

   ```bash
   python3 -m venv .venv
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
   OPENAI_CHAT_MODEL="gpt-4-turbo-2024-04-09"  # Optional: specify model
   ```

5. **Run the application**

   ```bash
   streamlit run streamlit_app_enhanced.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8501`

## 💡 Usage Examples

### Manual Operations

- **📤 Import Data**: Use the ribbon's file uploader to import CSV or XLSX files
- **🔄 Sort Data**: Select column and order in the Data tab
- **🔍 Filter Data**: Apply filters by column value in the Data tab
- **📊 Create Charts**: Use the Insert tab to build interactive visualizations
- **🧮 Apply Formulas**: Use the enhanced formula bar with auto-complete help
- **💾 Auto-save**: Changes are automatically saved with timestamp management

### AI Assistant Commands

Use the sidebar chat to interact with the AI agent using natural language:

**Data Manipulation:**
- *"Sort column A in ascending order"*
- *"Filter rows where column B equals 'Active'"*
- *"Calculate the sum of column D and put it in E1"*
- *"Add a new column F with the average of A and B"*

**Visualization:**
- *"Create a line chart showing sales over time"*
- *"Make a bar chart comparing revenue by region"*
- *"Show me a scatter plot of price vs quantity"*

**Formula Operations:**
- *"Apply =SUM(A1:A10) to cell B11"*
- *"Calculate running totals in column C"*
- *"Find the maximum value in column D"*

**File Operations:**
- *"Export the current sheet as CSV"*
- *"Create a new sheet called 'Analysis'"*
- *"Load the sample data from the data directory"*

## 📁 Project Structure

```
excelexcel-cursor-mvp/
├── app/
│   ├── agent/              # AI agent implementation
│   │   ├── agent.py       # Main AI agent with function calling
│   │   └── tools.py       # AI tool definitions and handlers
│   ├── services/          # Core spreadsheet logic
│   │   └── workbook.py    # Workbook and sheet management
│   ├── ui/                # UI components and formulas
│   │   ├── app.py         # Original UI components
│   │   └── formula.py     # Enhanced formula evaluation engine
│   └── charts.py          # Interactive chart generation with Plotly
├── tests/                 # Comprehensive test suite
│   ├── test_agent.py      # AI agent functionality tests
│   ├── test_formula.py    # Formula engine tests
│   ├── test_integration.py# End-to-end integration tests
│   └── test_workbook.py   # Core workbook operation tests
├── data/                  # Auto-managed data directory
│   ├── sample.csv         # Sample dataset
│   └── *.csv             # User uploaded and auto-saved files
├── .venv/                 # Python virtual environment
├── streamlit_app_enhanced.py  # Main application entry point
├── data_manager.py        # Data directory management utility
├── run_app.sh            # Application startup script
├── SETUP_STATUS.md       # Development environment documentation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore patterns
└── README.md            # This documentation
```

## 🛠 Technology Stack

### **Frontend & UI**

- **Streamlit 1.48+**: Modern web framework with advanced components
- **Custom CSS**: Comprehensive theming system (Light/Dark/Auto)
- **Responsive Design**: Professional Excel-like interface with ribbon navigation

### **Backend & Data Processing**

- **Python 3.12**: Modern Python with type hints and enhanced performance
- **Pandas 2.3+**: Advanced data manipulation and analysis
- **NumPy 2.3+**: Numerical computing foundation
- **OpenPyXL 3.1+**: Excel file format support (.xlsx)

### **AI & Machine Learning**

- **OpenAI API 1.99+**: GPT-4-Turbo integration with function calling
- **Function Calling**: Structured AI interactions with spreadsheet operations
- **Context Management**: Intelligent session state and conversation tracking

### **Visualization & Charts**

- **Plotly 6.3+**: Interactive charts and data visualization
- **Matplotlib 3.10+**: Statistical plotting and analysis
- **Altair 5.5+**: Grammar of graphics visualizations

### **Development & Testing**

- **Pytest**: Comprehensive testing framework (19 tests)
- **Virtual Environment**: Modern `.venv` setup with dependency isolation
- **Git Workflow**: Feature branch development with detailed commit history
- **Linting**: Code quality and formatting standards

### **File Management**

- **Glob Pattern Matching**: Advanced file discovery and management
- **Session State**: Robust upload duplicate prevention
- **Auto-save**: Intelligent backup system with timestamp management
- **Data Organization**: Structured directory management with cleanup utilities

## 🧪 Testing & Development

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_formula.py -v  # Formula engine tests
python -m pytest tests/test_agent.py -v   # AI agent tests
python -m pytest tests/test_workbook.py -v # Core functionality tests
```

### Development Setup

```bash
# Clone and setup
git clone https://github.com/paramshah07/excelexcel-cursor-mvp.git
cd excelexcel-cursor-mvp

# Setup development environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create environment file
cp .env.example .env  # Edit with your API keys

# Run tests
python -m pytest tests/ -v

# Start development server
streamlit run streamlit_app_enhanced.py
```

### Data Management

```bash
# Use the data management utility
python data_manager.py list     # List all files in data directory
python data_manager.py clean    # Remove duplicate files
python data_manager.py show sample.csv  # Preview file contents
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper testing
4. **Run the test suite**: `python -m pytest tests/ -v`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines

- **Code Quality**: Follow PEP 8 style guidelines
- **Testing**: Add tests for new functionality
- **Documentation**: Update README and code comments
- **Commit Messages**: Use clear, descriptive commit messages

## 📋 Current Limitations & Roadmap

### Known Limitations

- **Formula Engine**: Simplified implementation, not full Excel compatibility
- **File Size**: Large datasets may impact performance
- **Concurrent Users**: Single-user application (not multi-tenant)
- **Advanced Excel Features**: No VBA, macros, or complex Excel functions

### Upcoming Features

- **🔄 Real-time Collaboration**: Multi-user editing with conflict resolution
- **📊 Advanced Charts**: Pivot charts, combo charts, and custom visualizations
- **🔍 Advanced Filtering**: Multi-criteria filters and custom filter expressions
- **📋 Pivot Tables**: Full pivot table functionality with drag-and-drop interface
- **🔗 Data Connections**: Database connectivity and external data sources
- **🎨 Enhanced Formatting**: Cell styling, conditional formatting, and themes
- **📱 Mobile Responsiveness**: Optimized mobile and tablet experience

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **OpenAI**: For providing the GPT-4-Turbo API that powers the AI assistant
- **Streamlit**: For the excellent web framework that makes this application possible
- **Pandas & NumPy**: For robust data processing capabilities
- **Plotly**: For interactive data visualization components

## ⚠️ Disclaimer

This is an MVP (Minimum Viable Product) demonstration showcasing AI-powered spreadsheet capabilities. While functional and feature-rich, it is not intended as a complete replacement for Microsoft Excel or Google Sheets. The formula engine implements core functionality and is intentionally simplified for demonstration purposes.

## 🔗 Links

- **Repository**: [https://github.com/paramshah07/excelexcel-cursor-mvp](https://github.com/paramshah07/excelexcel-cursor-mvp)
- **Issues**: [Report bugs or request features](https://github.com/paramshah07/excelexcel-cursor-mvp/issues)
- **Pull Requests**: [Contribute to the project](https://github.com/paramshah07/excelexcel-cursor-mvp/pulls)

---

Built with ❤️ by the Excel-Cursor team