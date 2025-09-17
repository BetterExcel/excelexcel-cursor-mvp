import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt


def quick_plot(df: pd.DataFrame, x: str, ys: List[str], kind: str = "line"):
    """Legacy matplotlib function for compatibility."""
    if x not in df.columns:
        raise ValueError(f"x column '{x}' not found")
    for y in ys:
        if y not in df.columns:
            raise ValueError(f"y column '{y}' not found: {y}")

    fig = plt.figure()
    if kind == "line":
        for y in ys:
            plt.plot(df[x], df[y], label=y)
    else:
        # naive multi-series bar: stacked by index
        for y in ys:
            plt.bar(df[x], df[y], label=y)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_cols: List[str], 
                title: str = "", width: int = 800, height: int = 500):
    """
    Create various types of charts using Plotly.
    
    Args:
        df: DataFrame containing the data
        chart_type: Type of chart ('line', 'bar', 'scatter', 'area', 'pie', 'histogram')
        x_col: Column to use for x-axis
        y_cols: List of columns to use for y-axis
        title: Chart title
        width: Chart width
        height: Chart height
    """
    
    if df.empty or not x_col or not y_cols:
        st.warning("⚠️ Please select valid columns for the chart")
        return None
    
    try:
        # Clean data - remove rows where any selected column is null
        chart_df = df[[x_col] + y_cols].dropna()
        
        if chart_df.empty:
            st.warning("⚠️ No valid data found for selected columns")
            return None
        
        # Create the chart based on type
        fig = None
        
        if chart_type == "line":
            fig = px.line(chart_df, x=x_col, y=y_cols, title=title,
                         width=width, height=height)
            
        elif chart_type == "bar":
            if len(y_cols) == 1:
                fig = px.bar(chart_df, x=x_col, y=y_cols[0], title=title,
                           width=width, height=height)
            else:
                # Multiple series bar chart
                melted = chart_df.melt(id_vars=[x_col], value_vars=y_cols,
                                     var_name='Series', value_name='Value')
                fig = px.bar(melted, x=x_col, y='Value', color='Series', 
                           title=title, width=width, height=height,
                           barmode='group')
                           
        elif chart_type == "scatter":
            if len(y_cols) >= 1:
                fig = px.scatter(chart_df, x=x_col, y=y_cols[0], title=title,
                               width=width, height=height)
                # Add additional y-columns as separate traces
                for i, y_col in enumerate(y_cols[1:], 1):
                    fig.add_scatter(x=chart_df[x_col], y=chart_df[y_col], 
                                  mode='markers', name=y_col)
                                  
        elif chart_type == "area":
            fig = px.area(chart_df, x=x_col, y=y_cols, title=title,
                         width=width, height=height)
                         
        elif chart_type == "pie":
            if len(y_cols) >= 1:
                # For pie charts, use x_col as labels and first y_col as values
                fig = px.pie(chart_df, names=x_col, values=y_cols[0], title=title,
                           width=width, height=height)
                           
        elif chart_type == "histogram":
            if len(y_cols) >= 1:
                fig = px.histogram(chart_df, x=y_cols[0], title=title,
                                 width=width, height=height, nbins=20)
                                 
        elif chart_type == "box":
            fig = go.Figure()
            for y_col in y_cols:
                fig.add_trace(go.Box(y=chart_df[y_col], name=y_col))
            fig.update_layout(title=title, width=width, height=height)
            
        else:
            st.error(f"❌ Unsupported chart type: {chart_type}")
            return None
        
        if fig:
            # Enhance the chart appearance
            fig.update_layout(
                template="plotly_white",
                showlegend=True,
                hovermode='x unified',
                font=dict(size=12),
                title_font_size=16
            )
            
            return fig
            
    except Exception as e:
        st.error(f"❌ Error creating chart: {str(e)}")
        return None


def display_chart_builder(df: pd.DataFrame):
    """
    Display an interactive chart builder interface.
    """
    st.markdown("### 📊 Chart Builder")
    
    if df.empty:
        st.warning("⚠️ No data available for charting")
        return
    
    # Chart configuration columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        chart_type = st.selectbox(
            "📈 Chart Type",
            ["line", "bar", "scatter", "area", "pie", "histogram", "box"],
            help="Select the type of chart to create"
        )
        
        title = st.text_input("📝 Chart Title", value="Data Visualization")
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        x_col = st.selectbox(
            "📍 X-Axis Column",
            all_cols,
            help="Select column for X-axis"
        )
        
    with col3:
        if chart_type == "pie":
            y_cols = st.multiselect(
                "📊 Value Column",
                numeric_cols,
                max_selections=1,
                help="Select one numeric column for pie chart values"
            )
        elif chart_type == "histogram":
            y_cols = st.multiselect(
                "📊 Data Column",
                numeric_cols,
                max_selections=1,
                help="Select one numeric column for histogram"
            )
        else:
            y_cols = st.multiselect(
                "📊 Y-Axis Columns",
                numeric_cols,
                help="Select one or more numeric columns for Y-axis"
            )
    
    # Chart size controls
    size_col1, size_col2 = st.columns(2)
    with size_col1:
        width = st.slider("Width", 400, 1200, 800, 50)
    with size_col2:
        height = st.slider("Height", 300, 800, 500, 50)
    
    # Generate chart button
    if st.button("🚀 Generate Chart", type="primary"):
        if x_col and y_cols:
            with st.spinner("🔄 Creating chart..."):
                fig = create_chart(df, chart_type, x_col, y_cols, title, width, height)
                if fig:
                    st.plotly_chart(fig, width='stretch')
                    
                    # Chart export options
                    st.markdown("### 💾 Export Options")
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        if st.button("📥 Download as PNG"):
                            # This would require additional setup for image export
                            st.info("💡 Right-click the chart and select 'Download plot as PNG'")
                    
                    with export_col2:
                        if st.button("📋 Get Chart Code"):
                            st.code(f"""
# Plotly chart code:
import plotly.express as px

fig = px.{chart_type}(df, x='{x_col}', y={y_cols}, title='{title}')
fig.show()
                            """, language="python")
        else:
            st.warning("⚠️ Please select both X-axis and Y-axis columns")


def quick_stats_chart(df: pd.DataFrame, column: str):
    """
    Create a quick statistics visualization for a numeric column.
    """
    if column not in df.columns:
        return None
        
    col_data = pd.to_numeric(df[column], errors='coerce').dropna()
    
    if col_data.empty:
        return None
    
    # Create a subplot with histogram and box plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'{column} - Distribution', f'{column} - Box Plot'],
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=col_data, name="Distribution", nbinsx=20),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(x=col_data, name="Box Plot", orientation='h'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text=f"Quick Stats: {column}",
        template="plotly_white"
    )
    
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """
    Create a correlation heatmap for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        width=600,
        height=600,
        template="plotly_white"
    )
    
    return fig