# Unsupervised Machine Learning Analysis of Credit-Constrained Households: A Clustering Approach Using the 2019 Survey of Consumer Finances

                                                           by Imonikhe Ayeni

> 📖 **Read the illustrated case study:** [imonikheayeni.com/Customer-Segmentation.html](https://imonikheayeni.com/Customer-Segmentation.html)

## Abstract

This study applies unsupervised machine learning techniques to segment credit-constrained households using data from the 2019 Survey of Consumer Finances (SCF). Through the implementation of K-means clustering and Principal Component Analysis (PCA), I identify distinct subgroups within households facing credit access challenges. My methodology combines feature selection based on variance thresholds, dimensionality reduction through PCA, and optimal cluster determination using both elbow method and silhouette analysis. The results reveal two primary clusters of credit-constrained households, each with distinct demographic and financial characteristics. This segmentation provides valuable insights for targeted financial services, policy interventions, and marketing strategies aimed at underserved populations. The analysis achieved a silhouette score of 0.713, indicating strong cluster separation and validation of our approach.

**Keywords:** Consumer Finance, Clustering Analysis, Credit Constraints, Machine Learning, Azure Databricks, Principal Component Analysis, silhouette analysis, Financial Inclusion

## 1. Introduction

### 1.1 Background and Motivation

Financial inclusion remains a critical challenge in modern economies, with significant portions of households facing constraints in accessing credit markets. Understanding the heterogeneity within credit-constrained populations is essential for developing targeted interventions and financial products. Traditional approaches to analyzing consumer financial behavior often rely on predetermined segmentation criteria, potentially missing nuanced patterns in household characteristics.

Unsupervised machine learning techniques, particularly clustering algorithms, offer a data-driven approach to identify natural groupings within complex financial datasets. This study leverages the comprehensive nature of the Federal Reserve's Survey of Consumer Finances to explore the underlying structure of credit-constrained households through advanced analytical methods.

### 1.2 Research Objectives

The primary objectives of this research are:

1. **Identification of Credit-Constrained Households**: Develop a comprehensive methodology to identify households experiencing credit access difficulties using multiple SCF indicators
2. **Feature Engineering and Selection**: Apply systematic approaches to identify relevant variables for clustering analysis
3. **Dimensionality Reduction**: Implement Principal Component Analysis to reduce computational complexity while preserving data variance
4. **Cluster Analysis**: Employ K-means clustering to identify distinct subgroups within credit-constrained households
5. **Validation and Interpretation**: Validate clustering results and provide meaningful interpretations of identified segments
6. **Visualization and Communication**: Develop interactive tools for stakeholder engagement and result dissemination

### 1.3 Contribution to Literature

This study contributes to the existing literature on consumer finance and financial inclusion by:
- Providing a systematic, data-driven approach to household segmentation
- Demonstrating the application of modern machine learning techniques to traditional survey data
- Offering insights into the heterogeneity of credit-constrained populations
- Establishing a replicable methodology for ongoing analysis of consumer financial behavior

## 2. Literature Review

Consumer financial behavior analysis has evolved significantly with the advent of machine learning techniques. Hennig and Liao (2013) demonstrated the application of clustering methods to socio-economic stratification, highlighting the importance of appropriate clustering techniques for mixed-type variables. Their work established foundational principles for applying unsupervised learning to demographic and financial data.

Tatsat, Puri, and Lookabaugh (2020) provided comprehensive frameworks for machine learning applications in finance, emphasizing the importance of feature engineering and model validation in financial contexts. Their work underscores the potential for clustering techniques in customer segmentation and risk assessment.

The Federal Reserve's Survey of Consumer Finances has been extensively used for analyzing household financial behavior, providing rich datasets for understanding credit access patterns. Previous studies have primarily relied on traditional statistical methods, creating opportunities for machine learning approaches to uncover new insights.

## 3. Methodology

### 3.1 Data Source and Description

This analysis utilizes the US 2019 Survey of Consumer Finances (SCF), conducted by the Federal Reserve Board, USA. The SCF is a triennial survey that provides comprehensive information on household assets, liabilities, income, and demographic characteristics. The dataset contains detailed information on household financial behaviors, including credit applications, loan terms, and financial constraints.

### 3.2 Environment Setup and Configuration

The analysis was conducted using Azure Databricks, a cloud-based analytics platform that provides scalable computing resources and integrated machine learning capabilities. The following code establishes the analytical environment:

```python
# Environment configuration
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Core data manipulation libraries
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Machine learning libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
```

**Code Explanation**: This setup imports essential libraries for data manipulation (pandas, numpy), machine learning (scikit-learn), and visualization (matplotlib, plotly). Warning suppression is implemented to handle known compatibility issues in the Databricks environment.

### 3.3 Credit Constraint Identification

The identification of credit-constrained households employs a multi-criteria approach based on established literature and SCF survey questions:

```python
def identify_credit_constrained(df):
    """
    Identify households experiencing credit constraints using multiple indicators
    
    Parameters:
    df (DataFrame): Input dataset containing SCF variables
    
    Returns:
    DataFrame: Original dataset with credit constraint indicator
    """
    # Initialize binary indicator
    df['credit_constrained'] = 0
    
    # Criterion 1: Credit application denial
    if 'TURNDOWN' in df.columns:
        df.loc[df['TURNDOWN'] == 1, 'credit_constrained'] = 1
    
    # Criterion 2: Discouraged borrowing (self-censoring)
    if 'DISCOUR' in df.columns:
        df.loc[df['DISCOUR'] == 1, 'credit_constrained'] = 1
    
    # Criterion 3: High debt service burden
    if 'PAYMORT1' in df.columns and 'INCOME' in df.columns:
        df['payment_to_income'] = df['PAYMORT1'] / (df['INCOME'] + 1)
        df.loc[df['payment_to_income'] > 0.4, 'credit_constrained'] = 1
    
    return df
```

**Code Explanation**: This function implements a comprehensive approach to identifying credit constraints by combining explicit denials (TURNDOWN), discouraged applications (DISCOUR), and excessive debt service ratios. The multi-criteria approach captures both revealed and latent credit constraints.

### 3.4 Feature Engineering and Selection

Feature selection employs a systematic approach combining domain knowledge and statistical methods:

```python
def preprocess_features(df, features):
    """
    Comprehensive feature preprocessing for clustering analysis
    
    Parameters:
    df (DataFrame): Input dataset
    features (list): List of feature column names
    
    Returns:
    DataFrame: Processed dataset ready for clustering
    """
    processed_df = df[features].copy()
    
    # Missing value imputation
    for col in processed_df.columns:
        if processed_df[col].dtype in ['int64', 'float64']:
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
        else:
            processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    
    # Financial ratio creation
    if 'INCOME' in processed_df.columns and 'DEBT' in processed_df.columns:
        processed_df['debt_to_income'] = processed_df['DEBT'] / (processed_df['INCOME'] + 1)
    
    if 'ASSET' in processed_df.columns and 'DEBT' in processed_df.columns:
        processed_df['leverage_ratio'] = processed_df['DEBT'] / (processed_df['ASSET'] + 1)
    
    # Log transformation for skewed financial variables
    financial_cols = ['INCOME', 'NETWORTH', 'ASSET', 'DEBT']
    for col in financial_cols:
        if col in processed_df.columns:
            processed_df[f'{col}_log'] = np.log1p(processed_df[col].clip(lower=0))
    
    return processed_df
```

**Code Explanation**: This preprocessing function addresses common challenges in financial data analysis: missing values through median/mode imputation, creation of meaningful financial ratios, and log transformation to handle skewed distributions typical in financial variables.

Variance-based feature selection removes variables with insufficient variation:

```python
def select_features_by_variance(df, variance_threshold=0.01):
    """
    Feature selection based on variance threshold
    
    Parameters:
    df (DataFrame): Input dataset
    variance_threshold (float): Minimum variance threshold
    
    Returns:
    tuple: (selected_dataframe, selected_feature_names)
    """
    # Separate numerical and categorical variables
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Variance threshold selection for numerical features
    selector = VarianceThreshold(threshold=variance_threshold)
    numerical_selected = selector.fit_transform(df[numerical_cols])
    selected_numerical_cols = numerical_cols[selector.get_support()].tolist()
    
    # Categorical feature selection based on uniqueness
    selected_categorical_cols = [col for col in categorical_cols 
                               if df[col].nunique() > 1]
    
    selected_features = selected_numerical_cols + selected_categorical_cols
    return df[selected_features], selected_features
```

**Code Explanation**: This function implements automated feature selection by removing variables with low variance (which provide little discriminatory power for clustering) while preserving categorical variables with sufficient diversity.

### 3.5 Principal Component Analysis Implementation

Dimensionality reduction through PCA reduces computational complexity while preserving data structure:

```python
def apply_pca(data, variance_threshold=0.95):
    """
    Apply Principal Component Analysis for dimensionality reduction
    
    Parameters:
    data (array): Standardized input data
    variance_threshold (float): Target cumulative variance to preserve
    
    Returns:
    tuple: (transformed_data, pca_model, cumulative_variance)
    """
    # Initial PCA to determine component requirements
    pca = PCA()
    pca.fit(data)
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Final PCA with optimal components
    pca_final = PCA(n_components=n_components)
    pca_transformed = pca_final.fit_transform(data)
    
    print(f"Original dimensions: {data.shape[1]}")
    print(f"PCA dimensions: {n_components}")
    print(f"Variance explained: {cumulative_variance[n_components-1]:.3f}")
    
    return pca_transformed, pca_final, cumulative_variance
```

**Code Explanation**: This implementation automatically determines the optimal number of principal components required to preserve a specified proportion of data variance (default 95%), balancing dimensionality reduction with information preservation.

### 3.6 Optimal Cluster Determination

The selection of optimal cluster numbers employs dual validation approaches:

```python
def find_optimal_clusters(data, max_clusters=10):
    """
    Determine optimal cluster count using elbow method and silhouette analysis
    
    Parameters:
    data (array): PCA-transformed data for clustering
    max_clusters (int): Maximum number of clusters to evaluate
    
    Returns:
    tuple: (cluster_range, inertias, silhouette_scores)
    """
    inertias = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for k in cluster_range:
        # K-means clustering with consistent initialization
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate performance metrics
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, cluster_labels))
    
    return cluster_range, inertias, silhouette_scores
```
![](https://i.imgur.com/ClF0wGZ.png)

**Code Explanation**: This function systematically evaluates clustering performance across different numbers of clusters using two complementary metrics: within-cluster sum of squares (inertia) for the elbow method, and silhouette score for cluster separation quality.

### 3.7 K-Means Clustering Implementation

Final clustering applies the K-means algorithm with determined optimal parameters:

```python
def apply_kmeans_clustering(data, n_clusters):
    """
    Apply K-means clustering with specified parameters
    
    Parameters:
    data (array): PCA-transformed data
    n_clusters (int): Number of clusters determined by optimization
    
    Returns:
    tuple: (cluster_labels, fitted_model)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(data)
    
    return cluster_labels, kmeans
```

**Code Explanation**: This implementation ensures reproducible results through fixed random seed, uses multiple initializations (n_init=10) to avoid local optima, and sets appropriate convergence criteria.

## 4. Results and Analysis

### 4.1 Cluster Optimization Results

The cluster optimization analysis revealed clear patterns in both elbow method and silhouette analysis:

**Elbow Method Analysis:**
- Steepest decline in inertia occurs between k=2 and k=3
- Clear elbow point at k=2, indicating natural data structure
- Diminishing returns evident beyond k=2

**Silhouette Analysis:**
- Maximum silhouette score: 0.713 at k=2
- Substantial decline to 0.195 at k=3
- Consistently lower scores for k>2

Based on these analyses, k=2 emerged as the optimal number of clusters, representing two distinct segments within credit-constrained households.

### 4.2 Cluster Characteristics Analysis

```python
def analyze_cluster_characteristics(df, cluster_col='cluster', features=None):
    """
    Comprehensive analysis of cluster characteristics
    
    Parameters:
    df (DataFrame): Clustered dataset
    cluster_col (str): Column name containing cluster assignments
    features (list): Features to analyze across clusters
    
    Returns:
    DataFrame: Statistical summary by cluster
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if cluster_col in features:
            features.remove(cluster_col)
    
    # Calculate cluster statistics
    cluster_summary = df.groupby(cluster_col)[features].agg([
        'count', 'mean', 'median', 'std', 
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ])
    
    # Rename aggregation functions
    cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
    
    return cluster_summary
```

**Code Explanation**: This analysis function provides comprehensive statistical summaries for each cluster, including central tendencies, variability measures, and quartile distributions to understand cluster-specific characteristics.

### 4.3 Cluster Visualization and Interpretation

```python
def create_cluster_comparison_visualization(df, features, cluster_col='cluster'):
    """
    Create comprehensive cluster comparison visualizations
    
    Parameters:
    df (DataFrame): Clustered dataset
    features (list): Features to visualize
    cluster_col (str): Cluster assignment column
    
    Returns:
    plotly.graph_objects.Figure: Interactive comparison chart
    """
    cluster_means = df.groupby(cluster_col)[features].mean()
    
    # Create subplot structure
    n_features = len(features)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=features,
        vertical_spacing=0.12
    )
    
    colors = px.colors.qualitative.Set1[:len(cluster_means)]
    
    for i, feature in enumerate(features):
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {j}" for j in cluster_means.index],
                y=cluster_means[feature],
                name=feature,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=300 * rows,
        title_text="Cluster Characteristics Comparison",
        title_x=0.5
    )
    
    return fig
```
![](https://i.imgur.com/RL4YyaO.png)
**Code Explanation**: This visualization function creates side-by-side bar charts comparing cluster characteristics across multiple features, enabling stakeholders to quickly identify distinguishing characteristics of each segment.

### 4.4 Model Validation and Performance Metrics

The clustering analysis achieved strong validation metrics:

- **Silhouette Score**: 0.713 (indicating excellent cluster separation)
- **Cluster Balance**: Reasonably balanced cluster sizes preventing single-cluster dominance
- **Feature Importance**: PCA successfully reduced dimensionality while preserving 95% of data variance
- **Stability**: Consistent results across multiple random initializations

## 5. Discussion and Implications

### 5.1 Cluster Interpretation

**Cluster 0: High-Risk Constrained Households**
- Characterized by higher debt-to-income ratios
- Lower liquid asset holdings
- Higher likelihood of credit application denials
- Typically younger household heads with limited credit history

**Cluster 1: Moderate-Risk Constrained Households**
- Better financial positions relative to Cluster 0
- Credit constraints primarily due to discouraged borrowing
- Higher income levels but still facing access barriers
- More established households with complex financial needs

### 5.2 Policy and Business Implications

**For Financial Institutions:**
1. **Targeted Product Development**: Different cluster characteristics suggest needs for distinct financial products
2. **Risk Assessment Refinement**: Understanding cluster-specific risk profiles enables more nuanced underwriting
3. **Marketing Strategy**: Tailored messaging for each segment can improve engagement rates

**For Policymakers:**
1. **Financial Inclusion Programs**: Cluster-specific interventions can address unique barriers
2. **Regulatory Considerations**: Understanding household heterogeneity informs consumer protection policies
3. **Economic Monitoring**: Cluster analysis provides framework for tracking financial inclusion progress

### 5.3 Methodological Contributions

This study demonstrates the effectiveness of combining multiple machine learning techniques:
- **Feature Engineering**: Systematic approach to creating meaningful financial ratios
- **Dimensionality Reduction**: PCA implementation balances complexity and information preservation
- **Validation Methods**: Dual optimization criteria ensure robust cluster identification
- **Scalability**: Cloud-based implementation enables analysis of large datasets

## 6. Limitations and Future Research

### 6.1 Current Limitations

1. **Cross-sectional Analysis**: Single time point limits understanding of dynamic credit constraint patterns
2. **Survey Limitations**: Self-reported data may contain response biases
3. **Feature Selection**: Automated selection may miss domain-specific important variables
4. **Generalizability**: Results specific to 2019 economic conditions

### 6.2 Future Research Directions

1. **Longitudinal Analysis**: Panel data analysis to track household transitions between clusters
2. **Alternative Clustering Methods**: Exploration of hierarchical clustering and density-based methods
3. **Predictive Modeling**: Development of models to predict cluster membership for new households
4. **External Validation**: Comparison with alternative data sources and segmentation approaches

## 7. Conclusion

This study successfully demonstrates the application of unsupervised machine learning techniques to identify and analyze distinct segments within credit-constrained households. The methodology combining feature engineering, PCA-based dimensionality reduction, and K-means clustering revealed two primary household segments with distinct characteristics and financial profiles.

The high silhouette score (0.713) validates the quality of cluster separation, while the comprehensive analysis provides actionable insights for both financial institutions and policymakers. The identification of high-risk and moderate-risk constrained households enables targeted interventions and product development strategies.

The technical framework developed in this study provides a replicable methodology for ongoing consumer finance analysis, contributing to the broader understanding of financial inclusion challenges. The integration of traditional survey data with modern machine learning techniques demonstrates the potential for enhanced insights in consumer financial behavior research.

Future applications of this methodology could include real-time monitoring of credit market conditions, development of early warning systems for financial stress, and creation of more inclusive financial products tailored to specific household segments.

## References

Federal Reserve Board. (2020). *2019 Survey of Consumer Finances*. Washington, DC: Board of Governors of the Federal Reserve System.

Hennig, C., & Liao, T. F. (2013). How to find an appropriate clustering for mixed-type variables with application to socio-economic stratification. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 62(3), 309–369.

Tatsat, H., Puri, S., & Lookabaugh, B. (2020). *Machine learning and data science blueprints for finance: From building trading strategies to robo-advisors using Python*. O'Reilly Media.


---

**Author Information:** Imonikhe Ayeni

**Correspondence:** iayeni@cardiffmet.ac.uk, +44 07901364528

**Data Availability:** Code and documentation available here

**Ethics Statement:** This research uses publicly available, anonymized survey data in compliance with Federal Reserve data use policies.
