import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .cluster-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load the customer data"""
    # You can modify this to load from different sources
    try:
        df = pd.read_csv('Mall_Customers.csv')
        return df
    except:
        # Create sample data if file not found
        np.random.seed(42)
        sample_data = {
            'CustomerID': range(1, 201),
            'Gender': np.random.choice(['Male', 'Female'], 200),
            'Age': np.random.randint(18, 70, 200),
            'Annual Income (k$)': np.random.randint(15, 140, 200),
            'Spending Score (1-100)': np.random.randint(1, 100, 200)
        }
        return pd.DataFrame(sample_data)

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform K-means clustering"""
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Also run on original data for better visualization
    kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_original = kmeans_original.fit_predict(X)
    
    # Calculate metrics
    silhouette_scaled = silhouette_score(X_scaled, cluster_labels)
    silhouette_original = silhouette_score(X, cluster_labels_original)
    
    return (cluster_labels_original, kmeans_original.cluster_centers_, 
            silhouette_original, scaler, X_scaled)

def create_cluster_visualization(df, centroids):
    """Create interactive cluster visualization"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    fig = go.Figure()
    
    # Add cluster points
    for i in range(df['Cluster'].nunique()):
        cluster_data = df[df['Cluster'] == i]
        fig.add_trace(go.Scatter(
            x=cluster_data['Annual Income (k$)'],
            y=cluster_data['Spending Score (1-100)'],
            mode='markers',
            name=f'Cluster {i}',
            marker=dict(
                color=colors[i % len(colors)],
                size=8,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Cluster %{text}</b><br>' +
                         'Income: $%{x}k<br>' +
                         'Spending Score: %{y}<br>' +
                         '<extra></extra>',
            text=[i] * len(cluster_data)
        ))
    
    # Add centroids
    fig.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        name='Centroids',
        marker=dict(
            color='black',
            size=15,
            symbol='x',
            line=dict(width=3, color='white')
        ),
        hovertemplate='<b>Centroid</b><br>' +
                     'Income: $%{x:.1f}k<br>' +
                     'Spending Score: %{y:.1f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Customer Segmentation - Interactive Clustering',
        xaxis_title='Annual Income (k$)',
        yaxis_title='Spending Score (1-100)',
        hovermode='closest',
        height=600,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_cluster_profiles(df):
    """Generate cluster profiles and insights"""
    profiles = []
    cluster_names = {
        0: "💰 Budget Conscious",
        1: "🎯 Young Spenders", 
        2: "⚖️ Balanced Customers",
        3: "💎 Wealthy Conservatives",
        4: "👑 Premium Customers"
    }
    
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        
        profile = {
            'cluster_id': cluster_id,
            'name': cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'avg_age': cluster_data['Age'].mean(),
            'avg_income': cluster_data['Annual Income (k$)'].mean(),
            'avg_spending': cluster_data['Spending Score (1-100)'].mean(),
            'gender_dist': cluster_data['Gender'].value_counts().to_dict(),
            'age_range': f"{cluster_data['Age'].min()}-{cluster_data['Age'].max()}",
            'income_range': f"${cluster_data['Annual Income (k$)'].min()}k-${cluster_data['Annual Income (k$)'].max()}k"
        }
        profiles.append(profile)
    
    return profiles

def main():
    # Main title
    st.markdown('<h1 class="main-header">👥 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=5)
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    
    # Perform clustering
    cluster_labels, centroids, silhouette_score_val, scaler, X_scaled = perform_clustering(df, n_clusters)
    df['Cluster'] = cluster_labels
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Number of Clusters", n_clusters)
    with col3:
        st.metric("Silhouette Score", f"{silhouette_score_val:.3f}")
    with col4:
        st.metric("Avg Customer Age", f"{df['Age'].mean():.1f} years")
    
    st.markdown("---")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎯 Clustering", "📈 Analysis", "👥 Profiles", "💡 Insights"])
    
    with tab1:
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(df, x='Age', nbins=20, title='Age Distribution',
                                 color_discrete_sequence=['#4ECDC4'])
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Gender distribution
            gender_counts = df['Gender'].value_counts()
            fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                              title='Gender Distribution', color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            fig_gender.update_layout(height=400)
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # Income distribution
            fig_income = px.histogram(df, x='Annual Income (k$)', nbins=20, 
                                    title='Annual Income Distribution',
                                    color_discrete_sequence=['#45B7D1'])
            fig_income.update_layout(height=400)
            st.plotly_chart(fig_income, use_container_width=True)
            
            # Spending distribution
            fig_spending = px.histogram(df, x='Spending Score (1-100)', nbins=20,
                                      title='Spending Score Distribution',
                                      color_discrete_sequence=['#FFEAA7'])
            fig_spending.update_layout(height=400)
            st.plotly_chart(fig_spending, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Correlation Matrix", color_continuous_scale='RdBu')
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Customer Clustering Results")
        
        # Main clustering visualization
        fig_cluster = create_cluster_visualization(df, centroids)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster size distribution
            cluster_counts = df['Cluster'].value_counts().sort_index()
            fig_sizes = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                             title='Cluster Size Distribution',
                             color=cluster_counts.values,
                             color_continuous_scale='viridis')
            fig_sizes.update_layout(height=400, xaxis_title='Cluster', yaxis_title='Number of Customers')
            st.plotly_chart(fig_sizes, use_container_width=True)
        
        with col2:
            # 3D scatter plot
            fig_3d = px.scatter_3d(df, x='Annual Income (k$)', y='Spending Score (1-100)', z='Age',
                                 color='Cluster', title='3D Customer Segmentation',
                                 color_continuous_scale='viridis')
            fig_3d.update_layout(height=400)
            st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Detailed Analysis")
        
        # Box plots for each feature by cluster
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age_box = px.box(df, x='Cluster', y='Age', title='Age Distribution by Cluster',
                               color='Cluster', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_age_box, use_container_width=True)
            
            fig_income_box = px.box(df, x='Cluster', y='Annual Income (k$)', 
                                  title='Income Distribution by Cluster',
                                  color='Cluster', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_income_box, use_container_width=True)
        
        with col2:
            fig_spending_box = px.box(df, x='Cluster', y='Spending Score (1-100)', 
                                    title='Spending Score by Cluster',
                                    color='Cluster', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_spending_box, use_container_width=True)
            
            # Gender distribution by cluster
            gender_cluster = df.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
            fig_gender_cluster = px.bar(gender_cluster, title='Gender Distribution by Cluster',
                                      color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            fig_gender_cluster.update_layout(xaxis_title='Cluster', yaxis_title='Count')
            st.plotly_chart(fig_gender_cluster, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary by Cluster")
        summary_stats = df.groupby('Cluster').agg({
            'Age': ['mean', 'std'],
            'Annual Income (k$)': ['mean', 'std'],
            'Spending Score (1-100)': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        
        summary_stats.columns = ['Age Mean', 'Age Std', 'Income Mean', 'Income Std', 
                               'Spending Mean', 'Spending Std', 'Count']
        st.dataframe(summary_stats, use_container_width=True)
    
    with tab4:
        st.subheader("👥 Customer Profiles")
        
        profiles = create_cluster_profiles(df)
        
        for profile in profiles:
            with st.container():
                st.markdown(f"""
                <div class="cluster-card">
                    <h3>{profile['name']} (Cluster {profile['cluster_id']})</h3>
                    <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                        <div><strong>Size:</strong> {profile['size']} customers ({profile['percentage']:.1f}%)</div>
                        <div><strong>Avg Age:</strong> {profile['avg_age']:.1f} years</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                        <div><strong>Avg Income:</strong> ${profile['avg_income']:.1f}k</div>
                        <div><strong>Avg Spending:</strong> {profile['avg_spending']:.1f}/100</div>
                    </div>
                    <div style="margin: 1rem 0;">
                        <strong>Demographics:</strong> {dict(profile['gender_dist'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab5:
        st.subheader("💡 Business Insights & Recommendations")
        
        # Generate insights based on clusters
        insights = {
            "🎯 High-Value Segments": [
                "Focus marketing efforts on high-income, high-spending clusters",
                "Develop premium product lines for wealthy customer segments",
                "Create loyalty programs to retain top spenders"
            ],
            "📈 Growth Opportunities": [
                "Target medium-income customers with personalized offers",
                "Develop budget-friendly options for price-sensitive segments",
                "Create age-specific marketing campaigns"
            ],
            "⚠️ Risk Mitigation": [
                "Monitor low-engagement clusters for churn risk",
                "Implement win-back campaigns for declining segments",
                "Diversify customer base to reduce dependency"
            ],
            "🚀 Strategic Actions": [
                "Implement dynamic pricing based on customer segments",
                "Personalize product recommendations by cluster",
                "Optimize inventory based on segment preferences"
            ]
        }
        
        for category, recommendations in insights.items():
            st.markdown(f"### {category}")
            for rec in recommendations:
                st.markdown(f"• {rec}")
            st.markdown("---")
        
        # ROI Calculator
        st.subheader("💰 ROI Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            marketing_budget = st.number_input("Marketing Budget ($)", min_value=1000, value=10000, step=1000)
            conversion_rate = st.slider("Expected Conversion Rate (%)", min_value=1, max_value=20, value=5)
        
        with col2:
            avg_order_value = st.number_input("Average Order Value ($)", min_value=10, value=100, step=10)
            target_cluster = st.selectbox("Target Cluster", options=sorted(df['Cluster'].unique()))
        
        cluster_size = len(df[df['Cluster'] == target_cluster])
        expected_conversions = int(cluster_size * (conversion_rate / 100))
        expected_revenue = expected_conversions * avg_order_value
        roi = ((expected_revenue - marketing_budget) / marketing_budget) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Customers", cluster_size)
        with col2:
            st.metric("Expected Conversions", expected_conversions)
        with col3:
            st.metric("Expected ROI", f"{roi:.1f}%")
    
    # Sidebar additional info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("Silhouette Score", f"{silhouette_score_val:.3f}")
    st.sidebar.markdown("*Score > 0.5 indicates good clustering*")
    
    if show_raw_data:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Raw Data Preview")
        st.sidebar.dataframe(df.head())
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "Created with ❤️ using Streamlit | Customer Segmentation Dashboard"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()