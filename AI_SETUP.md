# AI-Powered Data Generation Setup

## Quick Start (Using Intelligent Templates)
The system works immediately with intelligent context-aware templates (as demonstrated). No setup required!

## Enhanced AI Mode (Optional)
For even more sophisticated data generation, you can enable OpenAI integration:

### 1. Get OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create account and get your API key
3. Ensure you have credits/billing set up

### 2. Set Environment Variable
```bash
# Add to your ~/.zshrc or ~/.bash_profile
export OPENAI_API_KEY="your-api-key-here"

# Or set for current session
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Install OpenAI Package (if not already installed)
```bash
pip install openai
```

## How It Works

### Intelligent Template Mode (Current)
- ✅ **Always Available** - No API key required
- ✅ **Context-Aware** - Understands data types from column names
- ✅ **Professional Quality** - Generates realistic business data
- ✅ **Fast** - Instant generation
- ✅ **Reliable** - Works offline

### Enhanced AI Mode (with API key)
- 🚀 **Ultra-Realistic** - GPT-4 powered contextual understanding
- 🚀 **Natural Language** - Understands complex descriptions
- 🚀 **Creative** - Generates unique, varied content
- 🚀 **Domain Expert** - Deep knowledge across industries

## Example Differences

### Template Mode Output:
```
Restaurant: "Green Bistro" | Cuisine: "Chinese" | Price: "$36.64"
```

### AI Mode Output (with API):
```
Restaurant: "Sakura Ramen House" | Cuisine: "Authentic Japanese" | Price: "$18.50"
```

Both modes produce professional, Excel-ready data. The AI mode just adds extra sophistication and uniqueness.

## Testing Your Setup

Run the demo to see your current mode:
```bash
python demo_ai_generation.py
```

- If you see "using intelligent template generation" → Template mode (works great!)
- If you see "Generated realistic data using AI" → Enhanced AI mode

## Recommendation

Start with the intelligent template mode (it's already excellent for most use cases). Add OpenAI integration later if you need the extra creativity and domain expertise.
