"""
Visualization Module for Gene Expression Clustering App
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_scatter_2d(X, labels, title="Scatter Plot", hover_data=None):
    """
    Create 2D scatter plot using first 2 principal components.
    
    Args:
        X: DataFrame with PCA components
        labels: Series with cluster labels
        title: Plot title
        hover_data: Additional hover information
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Prepare data
    df_plot = pd.DataFrame({
        'PC1': X.iloc[:, 0],
        'PC2': X.iloc[:, 1],
        'Cluster': labels.astype(str)
    })
    
    # Create figure
    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=title,
        template='plotly_white',
        hover_data=hover_data,
        height=500
    )
    
    # Update layout
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        legend=dict(
            title="Class",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def plot_scatter_2d_with_colors(X, labels, color_map, title="Scatter Plot"):
    """
    Create 2D scatter plot with custom color mapping.
    
    Args:
        X: DataFrame with PCA components
        labels: Series with cluster labels
        color_map: Dictionary mapping labels to colors
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Plot each class separately
    for label in labels.unique():
        mask = (labels == label).values  # Convert to numpy array for proper indexing
        fig.add_trace(go.Scatter(
            x=X.iloc[mask, 0].values,
            y=X.iloc[mask, 1].values,
            mode='markers',
            name=str(label),
            marker=dict(
                size=8,
                color=color_map.get(label, '#808080'),
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
            text=[label] * mask.sum()
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='PC1 (Principal Component 1)',
        yaxis_title='PC2 (Principal Component 2)',
        template='plotly_white',
        font=dict(size=12),
        title_font_size=16,
        height=500,
        legend=dict(
            title="Cancer Type",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig



def plot_metrics_comparison(metrics_dict):
    """
    Create bar chart comparing ARI scores across models.
    
    Args:
        metrics_dict: Dictionary with model names and their metrics
    
    Returns:
        plotly.graph_objects.Figure
    """
    models = list(metrics_dict.keys())
    ari_scores = [metrics_dict[m]['ARI'] for m in models]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=models,
        y=ari_scores,
        text=[f'{score:.4f}' for score in ari_scores],
        textposition='outside',
        marker=dict(
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'],
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>%{x}</b><br>ARI: %{y:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Comparison of ARI Scores Across Models',
        xaxis_title='Model',
        yaxis_title='Adjusted Rand Index (ARI)',
        template='plotly_white',
        font=dict(size=12),
        title_font_size=16,
        height=400,
        yaxis=dict(range=[0.9, 1.0])
    )
    
    return fig


def plot_weights_pie(weights):
    """
    Create pie chart showing ensemble weights.
    
    Args:
        weights: Dictionary with model names and their weights
    
    Returns:
        plotly.graph_objects.Figure
    """
    labels = list(weights.keys())
    values = list(weights.values())
    
    # Create figure
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        text=[f'{v*100:.0f}%' for v in values],
        textposition='inside',
        textfont_size=14,
        marker=dict(
            colors=['#4ECDC4', '#45B7D1', '#FF6B6B'],
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>'
    )])
    
    # Update layout
    fig.update_layout(
        title='Adaptive Weights in Ensemble Model',
        template='plotly_white',
        font=dict(size=12),
        title_font_size=16,
        height=400,
        showlegend=True
    )
    
    return fig


def plot_class_distribution(labels):
    """
    Create bar chart showing class distribution.
    
    Args:
        labels: Series with class labels
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Count classes
    counts = labels.value_counts().sort_index()
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=counts.index,
        y=counts.values,
        text=counts.values,
        textposition='outside',
        marker=dict(
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>%{x}</b><br>Samples: %{y}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Class Distribution',
        xaxis_title='Cancer Type',
        yaxis_title='Number of Samples',
        template='plotly_white',
        font=dict(size=12),
        title_font_size=16,
        height=400
    )
    
    return fig
