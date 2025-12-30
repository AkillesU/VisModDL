import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="Neuro Data Browser")

st.title("Neural Unit Distribution Browser")

# 1. Folder Selection
folder_path = st.text_input("Enter the path to your CSV folder:", value="./")

if os.path.exists(folder_path):
    # Filter for csv files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not all_files:
        st.warning(f"No CSV files found in {folder_path}")
    
    with st.sidebar:
        st.header("1. Data Selection")
        selected_files = st.multiselect("Select Files", all_files)
        
        st.header("2. Variable Selection")
        metric = st.radio("Metric Prefix", ["mw", "hg"])
        
        # New Feature: Absolute Transformation Toggle
        use_abs = st.checkbox("Apply Absolute Transformation", value=False, 
                              help="hg: |val| , mw: |val - 600|")
        
        categories = st.multiselect(
            "Categories", 
            ["animals", "faces", "objects", "places"],
            default=["animals"]
        )
        
        st.header("3. Plot Settings")
        plot_type = st.selectbox("Y-Axis Type", ["count", "probability density"])
        bin_size = st.slider("Bin Width (Approx)", 0.01, 50.0, 1.0)
        opacity = st.slider("Layer Opacity", 0.0, 1.0, 0.5)

    if selected_files and categories:
        combined_data = []

        for file in selected_files:
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                
                # Check if layer column exists
                if 'layer' not in df.columns:
                    st.error(f"File {file} is missing the 'layer' column.")
                    continue

                # Dynamically find layers in this specific file
                available_layers = df['layer'].unique().tolist()
                selected_layers = st.sidebar.multiselect(f"Layers in {file}", available_layers, key=file)
                
                for layer in selected_layers:
                    layer_df = df[df['layer'] == layer]
                    
                    for cat in categories:
                        col_name = f"{metric}_{cat}"
                        if col_name in layer_df.columns:
                            # Apply transformations
                            values = layer_df[col_name].copy()
                            
                            if use_abs:
                                if metric == "hg":
                                    values = values.abs()
                                elif metric == "mw":
                                    values = (values - 600).abs()

                            temp_df = pd.DataFrame({
                                'Value': values,
                                'File': file,
                                'Layer': layer,
                                'Category': cat,
                                'Label': f"{file} | {layer} | {cat}"
                            })
                            combined_data.append(temp_df)
            except Exception as e:
                st.error(f"Error reading {file}: {e}")

        if combined_data:
            plot_df = pd.concat(combined_data)

            # Map histogram type
            histnorm = 'probability density' if plot_type == "probability density" else None

            # Calculate number of bins based on data range and bin_size
            data_range = plot_df['Value'].max() - plot_df['Value'].min()
            nbins = int(data_range / bin_size) if bin_size > 0 and data_range > 0 else 30

            fig = px.histogram(
                plot_df, 
                x="Value", 
                color="Label", 
                barmode="overlay",
                marginal="rug", 
                histnorm=histnorm,
                opacity=opacity,
                nbins=nbins,
                title=f"Distribution Analysis: {metric.upper()} {'(Absolute)' if use_abs else ''}"
            )

            fig.update_layout(
                bargap=0.01,
                xaxis_title=f"{metric} Value {' (Transformed)' if use_abs else ''}",
                yaxis_title=plot_type.capitalize(),
                legend_title="Permutations",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Optional: Add summary statistics
            if st.checkbox("Show Statistics Table"):
                stats = plot_df.groupby('Label')['Value'].agg(['mean', 'median', 'std', 'count'])
                st.table(stats)
                
        else:
            st.info("Select layers for the files in the sidebar to visualize data.")
    else:
        st.info("Please select at least one file and category to begin.")
else:
    st.error("Invalid folder path. Please enter a valid directory containing your CSVs.")