from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import pickle
import os
import glob
from plotly.subplots import make_subplots

def extract_filename(filepath):
    """Extract the base filename without extension from a path"""
    return os.path.splitext(os.path.basename(filepath))[0]

def overlay_figures(file_paths, color_map=None, title="Combined Figure with Shared Axes", width=1000, height=800):
    """
    Overlay multiple plotly figures from pickle files with different colors
    
    Args:
        file_paths: List of paths to pickle files containing plotly figures
        color_map: Dictionary mapping file paths or filenames to colors (rgba strings)
        title: Title for the combined figure
        width: Width of the figure in pixels
        height: Height of the figure in pixels
        
    Returns:
        combined_fig: A plotly figure with all traces overlaid
    """
    if not file_paths:
        raise ValueError("No file paths provided")
    
    # Initialize color map if not provided
    if color_map is None:
        color_map = {}
    
    # Create a new figure with shared axes
    combined_fig = make_subplots()
    
    # Track the number of traces added from each file for legend grouping
    file_trace_counts = {}
    total_traces_added = 0
    
    # Process each file
    for idx, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'rb') as f:
                fig = pickle.load(f)
            
            if not isinstance(fig, go.Figure):
                print(f"Warning: {file_path} does not contain a plotly Figure object")
                continue
                
            # Get traces from the figure
            traces = [trace for trace in fig.data]
            
            if not traces:
                print(f"Warning: No traces found in {file_path}")
                continue
                
            # Get a display name for the file
            file_display_name = extract_filename(file_path)
            
            # Determine color for this file
            file_key = file_path
            file_name_key = file_display_name
            
            # Check if color is specified for this file
            if file_key in color_map:
                color = color_map[file_key]
            elif file_name_key in color_map:
                color = color_map[file_name_key]
            else:
                # Default color if not specified (light gray)
                color = 'rgba(150, 150, 150, 0.25)'
            
            # Keep track of how many traces we're adding from this file
            file_trace_counts[file_display_name] = len(traces)
            
            # Add all traces with the determined color
            for trace in traces:
                # For Scatter traces that use line_color
                if isinstance(trace, go.Scatter) and hasattr(trace, 'line_color'):
                    trace.line = dict(color=color)
                    # Remove the original line_color to avoid conflicts
                    if hasattr(trace, '_prop_descriptions') and 'line_color' in trace._prop_descriptions:
                        delattr(trace, 'line_color')
                # For standard Scatter traces
                elif hasattr(trace, 'line'):
                    trace.line.color = color
                # For scatter plots
                if hasattr(trace, 'marker'):
                    trace.marker.color = color
                # For filled area plots
                if hasattr(trace, 'fillcolor'):
                    trace.fillcolor = color
                
                combined_fig.add_trace(trace)
                total_traces_added += 1
                
            # Store layout information from the first valid figure
            if idx == 0:
                first_fig = fig
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if total_traces_added == 0:
        raise ValueError("No valid traces were found in any of the provided files")
    
    # Update layout using properties from the first figure
    combined_fig.update_layout(
        title=title,
        xaxis_title=first_fig.layout.xaxis.title.text if hasattr(first_fig.layout.xaxis, 'title') and hasattr(first_fig.layout.xaxis.title, 'text') else None,
        yaxis_title=first_fig.layout.yaxis.title.text if hasattr(first_fig.layout.yaxis, 'title') and hasattr(first_fig.layout.yaxis.title, 'text') else None,
        width=width,   # Set figure width
        height=height, # Set figure height
        margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins for better space utilization
        autosize=False,  # Disable autosize to enforce our dimensions
        legend=dict(
            orientation="h",     # Horizontal legend
            yanchor="bottom",    # Anchor point
            y=-0.2,              # Position below the plot
            xanchor="center",    # Center horizontally
            x=0.5                # Center position
        )
    )
    
    # Update legend names to include file information
    trace_index = 0
    for file_name, trace_count in file_trace_counts.items():
        for i in range(trace_count):
            if trace_index < len(combined_fig.data):
                trace = combined_fig.data[trace_index]
                # Update trace name to include file name
                original_name = trace.name if trace.name else f"Trace {i+1}"
                trace.name = f"{file_name}: {original_name}"
                # For the first trace of each file, ensure it shows in the legend
                if i == 0:
                    trace.showlegend = True
                trace_index += 1
    
    return combined_fig

if __name__ == '__main__':
    file_paths = [
        'hidden/traces/thermostat_hidden_0.0005_long.pkl',
        'hidden/traces/thermostat_simulation_long.pkl'
    ]
    
    # Define custom colors for specific files (can use full path or just filename)
    color_map = {
        'thermostat_hidden_0.0005_long': 'rgba(255, 0, 0, 0.25)',  # Red for hidden
        'thermostat_simulation_long': 'rgba(0, 0, 255, 0.25)',     # Blue for simulation
        # Add more mappings as needed
        # 'other_file': 'rgba(0, 255, 0, 0.25)',              # Green for another file
    }
    
    print(f"Overlaying {len(file_paths)} figures from files: {file_paths}")
    
    combined_fig = overlay_figures(
        file_paths,
        color_map=color_map,
        width=1200,    # Width in pixels
        height=1000     # Height in pixels
    )
    combined_fig.write_image('hidden/figures/long_0.0005.png')
    combined_fig.show()