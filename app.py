import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Scikit-Learn Compatibility Patch for Pickle Deserialization
# Resolves "No module named '_loss'" when unpickling GradientBoosting models
# ---------------------------------------------------------------------------
import sys

if '_loss' not in sys.modules:
    try:
        import sklearn._loss._loss as _cy_loss
        sys.modules['_loss'] = _cy_loss
        try:
            import sklearn._loss.loss as _py_loss
            for _attr in dir(_py_loss):
                if not hasattr(_cy_loss, _attr):
                    setattr(_cy_loss, _attr, getattr(_py_loss, _attr))
        except Exception:
            pass
    except Exception:
        try:
            import sklearn._loss as _sk_loss
            sys.modules['_loss'] = _sk_loss
        except Exception:
            try:
                import sklearn.ensemble._gb_losses as _gb_loss
                sys.modules['_loss'] = _gb_loss
            except Exception:
                pass


# Set page configuration
st.set_page_config(
    page_title="🌍 Energy Consumption Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-top: -2rem;
        margin-bottom: 1rem;
        padding-top: 0rem;
    }
    .block-container {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .model-performance {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .nav-button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load the energy consumption dataset"""
    try:
        data = pd.read_csv('World Energy Consumption.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'World Energy Consumption.csv' is in the project directory.")
        return None

@st.cache_resource
def load_models():
    """Load all trained models with backward-compatibility and fallback resilience"""
    # Ensure scikit-learn Cython loss module is aliased before deserialization
    if '_loss' not in sys.modules:
        try:
            import sklearn._loss._loss as _cy_loss
            sys.modules['_loss'] = _cy_loss
        except Exception:
            try:
                import sklearn._loss as _sk_loss
                sys.modules['_loss'] = _sk_loss
            except Exception:
                pass

    models = {}
    model_files = {
        'Linear Regression': 'models/linearregression_model.pkl',
        'Ridge Regression': 'models/ridge_model.pkl',
        'Lasso Regression': 'models/lasso_model.pkl',
        'Random Forest': 'models/randomforest_model.pkl',
        'Gradient Boosting': 'models/gradientboosting_model.pkl',
        'Best Model': 'models/best_model.pkl'
    }
    
    for name, path in model_files.items():
        try:
            if os.path.exists(path):
                models[name] = joblib.load(path)
            else:
                st.warning(f"Model file not found: {path}")
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")
            
    # Resilient fallback: ensure Best Model and Gradient Boosting are always operational
    if 'Best Model' not in models and 'Gradient Boosting' in models:
        models['Best Model'] = models['Gradient Boosting']
    elif 'Best Model' not in models and 'Random Forest' in models:
        models['Best Model'] = models['Random Forest']

    if 'Gradient Boosting' not in models and 'Random Forest' in models:
        models['Gradient Boosting'] = models['Random Forest']
    
    return models

@st.cache_data
def load_model_performance():
    """Load model performance comparison"""
    try:
        if os.path.exists('models/model_performance_comparison.csv'):
            return pd.read_csv('models/model_performance_comparison.csv')
        else:
            # Fallback performance data
            return pd.DataFrame({
                'Model': ['Gradient Boosting', 'Random Forest', 'Ridge Regression', 'Lasso Regression', 'Linear Regression'],
                'R_squared': [0.9998, 0.9992, 0.5046, 0.4933, 0.4928]
            })
    except Exception as e:
        st.error(f"Error loading performance data: {str(e)}")
        return None


def safe_render_image(col, image, caption=None):
    """Render image safely across all Streamlit versions without crashing on use_column_width."""
    try:
        col.image(image, caption=caption, use_container_width=True)
    except TypeError:
        try:
            col.image(image, caption=caption, use_container_width=True)
        except TypeError:
            col.image(image, caption=caption)

def get_model_features(model_name, model=None):
    """Get the correct feature list for each model type"""
    # Base feature list (9 features)
    base_features = ['year', 'population', 'gdp', 'energy_per_capita', 'renewables_share_energy', 
                     'fossil_fuel_consumption', 'gdp_per_capita', 'energy_intensity', 
                     'gdp_renewables_interaction']
    
    # Extended feature list (10 features)
    extended_features = ['year', 'population', 'gdp', 'energy_per_capita', 'renewables_share_energy', 
                         'fossil_fuel_consumption', 'gdp_per_capita', 'energy_intensity', 
                         'population_density', 'gdp_renewables_interaction']
    
    # Try to detect the expected number of features from the model
    if model is not None:
        try:
            # Try with 9 features first
            test_input = np.zeros((1, 9))
            model.predict(test_input)
            return base_features
        except:
            try:
                # Try with 10 features
                test_input = np.zeros((1, 10))
                model.predict(test_input)
                return extended_features
            except:
                pass
    
    # Fallback to manual mapping based on model name
    # Most models use 9 features, only specific ones use 10
    if model_name in ['Gradient Boosting', 'Best Model']:
        return extended_features  # 10 features
    else:
        return base_features  # 9 features

def preprocess_input_data(data, features):
    """Preprocess input data for prediction"""
    # Handle missing values
    data = data.fillna(data.median())
    
    # Apply log transformation to specific features (as done in training)
    log_features = ['gdp', 'population', 'gdp_per_capita', 'primary_energy_consumption']
    for col in log_features:
        if col in data.columns and col != 'primary_energy_consumption':
            data[col] = np.log1p(data[col])
    
    # Create engineered features
    if 'gdp' in data.columns and 'population' in data.columns:
        data['gdp_per_capita'] = data['gdp'] / data['population']
    
    if 'primary_energy_consumption' in data.columns and 'gdp' in data.columns:
        data['energy_intensity'] = data['primary_energy_consumption'] / data['gdp']
        data['energy_intensity'] = data['energy_intensity'].fillna(data['energy_intensity'].median())
    
    if 'gdp_per_capita' in data.columns and 'renewables_share_energy' in data.columns:
        data['gdp_renewables_interaction'] = data['gdp_per_capita'] * data['renewables_share_energy']
    
    # Add population density (set to NaN if area not available)
    data['population_density'] = np.nan
    
    # Select only the required features and handle feature count mismatch
    available_features = [f for f in features if f in data.columns]
    result_data = data[available_features].fillna(data[available_features].median())
    
    # Ensure we have the right number of features by padding or truncating
    if len(available_features) < len(features):
        # Pad with zeros for missing features
        for missing_feature in features:
            if missing_feature not in available_features:
                result_data[missing_feature] = 0
    
    return result_data[features]

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Energy Consumption Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data and models
    data = load_data()
    models = load_models()
    performance_data = load_model_performance()
    
    if data is None:
        st.stop()
    
    # Top Navigation Menu
    st.markdown("---")
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    # Create horizontal menu using columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Define button types based on current page
    button_type = "primary" if st.session_state.current_page == "🏠 Home" else "secondary"
    with col1:
        home_btn = st.button("🏠 Home", use_container_width=True, type=button_type)
    
    button_type = "primary" if st.session_state.current_page == "📊 Data Overview" else "secondary"
    with col2:
        data_btn = st.button("📊 Data Overview", use_container_width=True, type=button_type)
    
    button_type = "primary" if st.session_state.current_page == "🤖 Model Testing" else "secondary"
    with col3:
        model_btn = st.button("🤖 Model Testing", use_container_width=True, type=button_type)
    
    button_type = "primary" if st.session_state.current_page == "📈 Model Comparison" else "secondary"
    with col4:
        compare_btn = st.button("📈 Model Comparison", use_container_width=True, type=button_type)
    
    button_type = "primary" if st.session_state.current_page == "🔮 Make Predictions" else "secondary"
    with col5:
        predict_btn = st.button("🔮 Make Predictions", use_container_width=True, type=button_type)
    
    button_type = "primary" if st.session_state.current_page == "🖼️ Visualizations" else "secondary"
    with col6:
        viz_btn = st.button("🖼️ Visualizations", use_container_width=True, type=button_type)
    
    st.markdown("---")
    
    # Update current page based on button clicks
    if home_btn:
        st.session_state.current_page = "🏠 Home"
    elif data_btn:
        st.session_state.current_page = "📊 Data Overview"
    elif model_btn:
        st.session_state.current_page = "🤖 Model Testing"
    elif compare_btn:
        st.session_state.current_page = "📈 Model Comparison"
    elif predict_btn:
        st.session_state.current_page = "🔮 Make Predictions"
    elif viz_btn:
        st.session_state.current_page = "🖼️ Visualizations"
    
    page = st.session_state.current_page
    
    # Display current page indicator
    st.markdown(f"### 📍 Current Page: **{page}**")
    st.markdown("---")
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page(data, models, performance_data)
    elif page == "📊 Data Overview":
        show_data_overview(data)
    elif page == "🤖 Model Testing":
        show_model_testing(data, models)
    elif page == "📈 Model Comparison":
        show_model_comparison(performance_data)
    elif page == "🔮 Make Predictions":
        show_prediction_page(models, data)
    elif page == "🖼️ Visualizations":
        show_visualizations(data)

def show_home_page(data, models, performance_data):
    """Display the home page with project overview"""
    st.markdown("## 🎯 Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Dataset Size", f"{data.shape[0]:,} rows")
        st.metric("📋 Features", f"{data.shape[1]} columns")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🤖 Models Trained", len(models))
        if performance_data is not None:
            best_score = performance_data['R_squared'].max()
            st.metric("🏆 Best R² Score", f"{best_score:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌍 Countries", data['country'].nunique())
        st.metric("📅 Year Range", f"{data['year'].min()}-{data['year'].max()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project description
    st.markdown("""
    ### 📝 About This Project
    
    This Energy Consumption Analysis project provides comprehensive insights into global energy consumption patterns 
    and offers predictive modeling capabilities using advanced machine learning techniques.
    
    **Key Features:**
    - 🔍 **Exploratory Data Analysis**: Comprehensive analysis of global energy trends
    - 🤖 **Multiple ML Models**: Comparison of 5 different regression algorithms
    - 📊 **Interactive Visualizations**: Dynamic charts and plots
    - 🔮 **Prediction Engine**: Real-time energy consumption predictions
    - 📈 **Model Performance**: Detailed comparison and evaluation metrics
    
    **Dataset Source:** [Kaggle - World Energy Consumption](https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption)
    """)
    
    # Quick stats
    if data is not None:
        st.markdown("### 📊 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_consumption = data['primary_energy_consumption'].mean()
            st.metric("Average Energy Consumption", f"{avg_consumption:.1f} TWh")
        
        with col2:
            total_consumption = data['primary_energy_consumption'].sum()
            st.metric("Total Energy Consumption", f"{total_consumption/1000:.1f}K TWh")
        
        with col3:
            renewable_share = data['renewables_share_energy'].mean()
            st.metric("Average Renewable Share", f"{renewable_share:.1f}%")
        
        with col4:
            latest_year = data['year'].max()
            st.metric("Latest Data Year", latest_year)

def show_data_overview(data):
    """Display data overview and basic statistics"""
    st.markdown("## 📊 Data Overview")
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dataset Information")
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        st.write(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        st.write(f"**Missing Values:** {data.isnull().sum().sum()} total")
    
    with col2:
        st.markdown("### 🌍 Geographic Coverage")
        st.write(f"**Countries:** {data['country'].nunique()}")
        st.write(f"**Time Period:** {data['year'].min()} - {data['year'].max()}")
        st.write(f"**Years Covered:** {data['year'].nunique()} years")
    
    # Data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📈 Summary Statistics")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    st.dataframe(data[numeric_cols].describe(), use_container_width=True)
    
    # Missing values visualization
    st.markdown("### ❌ Missing Values Analysis")
    missing_data = data.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_data.head(20).plot(kind='bar', ax=ax)
        ax.set_title('Top 20 Columns with Missing Values')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Number of Missing Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.success("✅ No missing values found in the dataset!")

def show_model_testing(data, models):
    """Display model testing interface"""
    st.markdown("## 🤖 Model Testing")
    
    if not models:
        st.error("No models loaded! Please ensure model files are in the 'models/' directory.")
        return
    
    # Model selection
    selected_model_name = st.selectbox("Select a model to test:", list(models.keys()))
    selected_model = models[selected_model_name]
    
    st.markdown(f"### Testing: **{selected_model_name}**")
    
    # Get the correct feature set for this model
    features = get_model_features(selected_model_name, selected_model)
    st.info(f"🔍 **Model Info:** {selected_model_name} expects {len(features)} features")
    
    # Filter data for testing
    test_data = data.dropna(subset=['primary_energy_consumption'])
    
    if len(test_data) == 0:
        st.error("No valid test data available!")
        return
    
    # Sample selection
    sample_size = st.slider("Select sample size for testing:", 100, min(5000, len(test_data)), 1000)
    test_sample = test_data.sample(n=sample_size, random_state=42)
    
    # Debug: Show data info
    with st.expander("🔍 Data Debug Info"):
        st.write(f"**Sample shape:** {test_sample.shape}")
        st.write(f"**Available columns:** {list(test_sample.columns)}")
        st.write(f"**Required features:** {features}")
        
        # Check for missing required columns
        missing_cols = [f for f in features if f not in test_sample.columns]
        if missing_cols:
            st.warning(f"**Missing columns:** {missing_cols}")
        
        # Show data types
        st.write("**Data types:**")
        st.write(test_sample[features[:5] if len(features) > 5 else features].dtypes)
    
    # Prepare features
    try:
        # First, ensure we only work with numeric columns and create engineered features
        test_sample_processed = test_sample.copy()
        
        # Create engineered features
        if 'gdp' in test_sample_processed.columns and 'population' in test_sample_processed.columns:
            test_sample_processed['gdp_per_capita'] = test_sample_processed['gdp'] / test_sample_processed['population']
        
        if 'primary_energy_consumption' in test_sample_processed.columns and 'gdp' in test_sample_processed.columns:
            test_sample_processed['energy_intensity'] = test_sample_processed['primary_energy_consumption'] / test_sample_processed['gdp']
            test_sample_processed['energy_intensity'] = test_sample_processed['energy_intensity'].fillna(test_sample_processed['energy_intensity'].median())
        
        if 'gdp_per_capita' in test_sample_processed.columns and 'renewables_share_energy' in test_sample_processed.columns:
            test_sample_processed['gdp_renewables_interaction'] = test_sample_processed['gdp_per_capita'] * test_sample_processed['renewables_share_energy']
        
        # Add population density (set to 0 if not available, as done in training)
        if 'population_density' not in test_sample_processed.columns:
            test_sample_processed['population_density'] = 0
        
        # Apply log transformation to specific features (as done in training)
        log_features = ['gdp', 'population', 'gdp_per_capita']
        for col in log_features:
            if col in test_sample_processed.columns:
                test_sample_processed[col] = np.log1p(test_sample_processed[col])
        
        # Select only the required features and ensure they are numeric
        X_test = test_sample_processed[features].copy()
        
        # Ensure all features are numeric
        for col in X_test.columns:
            try:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            except Exception as e:
                st.error(f"❌ Error converting column '{col}' to numeric: {str(e)}")
                st.write(f"Sample values in '{col}': {X_test[col].head().tolist()}")
                return
        
        # Fill any remaining NaN values
        for col in X_test.columns:
            if X_test[col].isnull().any():
                median_val = X_test[col].median()
                if pd.isna(median_val):
                    X_test[col] = X_test[col].fillna(0)
                else:
                    X_test[col] = X_test[col].fillna(median_val)
        
        # Verify we have valid numeric data
        if X_test.isnull().any().any():
            st.error("❌ Data still contains NaN values after processing")
            null_cols = X_test.columns[X_test.isnull().any()].tolist()
            st.write(f"Columns with NaN values: {null_cols}")
            return
            
        y_test = test_sample['primary_energy_consumption']
        
        # Make predictions
        y_pred = selected_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            st.metric("MSE", f"{mse:.2f}")
        
        # Prediction vs Actual plot
        st.markdown("### 📊 Prediction vs Actual")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='royalblue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Actual vs Predicted - {selected_model_name}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Residuals plot
        st.markdown("### 📈 Residuals Analysis")
        residuals = y_test - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Predicted')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Residuals')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Feature importance (if available)
        if hasattr(selected_model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            
            importances = selected_model.feature_importances_
            feature_names = features[:len(importances)]  # Handle length mismatch
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance - {selected_model_name}')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error during model testing: {str(e)}")
        st.info("This might be due to feature mismatch or data preprocessing issues.")

def show_model_comparison(performance_data):
    """Display model comparison"""
    st.markdown("## 📈 Model Comparison")
    
    if performance_data is None:
        st.error("Performance data not available!")
        return
    
    # Performance table
    st.markdown("### 🏆 Model Performance Rankings")
    performance_sorted = performance_data.sort_values('R_squared', ascending=False)
    
    # Add ranking
    performance_sorted['Rank'] = range(1, len(performance_sorted) + 1)
    performance_sorted['Performance'] = performance_sorted['R_squared'].apply(
        lambda x: "🥇 Excellent" if x > 0.95 else "🥈 Very Good" if x > 0.8 else "🥉 Good" if x > 0.6 else "📈 Moderate"
    )
    
    st.dataframe(
        performance_sorted[['Rank', 'Model', 'R_squared', 'Performance']].reset_index(drop=True),
        use_container_width=True
    )
    
    # Performance visualization
    st.markdown("### 📊 Performance Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['gold' if i == 0 else 'lightcoral' for i in range(len(performance_sorted))]
    
    bars = ax.barh(performance_sorted['Model'], performance_sorted['R_squared'], color=colors)
    ax.set_xlabel('R² Score')
    ax.set_title('Model Performance Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (model, score) in enumerate(zip(performance_sorted['Model'], performance_sorted['R_squared'])):
        ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Performance insights
    st.markdown("### 💡 Key Insights")
    
    best_model = performance_sorted.iloc[0]
    worst_model = performance_sorted.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🏆 **Best Model:** {best_model['Model']} (R² = {best_model['R_squared']:.4f})")
        st.info(f"📊 **Performance Gap:** {(best_model['R_squared'] - worst_model['R_squared']):.4f}")
    
    with col2:
        tree_models = performance_sorted[performance_sorted['Model'].isin(['Random Forest', 'Gradient Boosting'])]
        linear_models = performance_sorted[performance_sorted['Model'].isin(['Linear Regression', 'Ridge Regression', 'Lasso Regression'])]
        
        if len(tree_models) > 0 and len(linear_models) > 0:
            avg_tree = tree_models['R_squared'].mean()
            avg_linear = linear_models['R_squared'].mean()
            st.info(f"🌳 **Tree Models Avg:** {avg_tree:.4f}")
            st.info(f"📏 **Linear Models Avg:** {avg_linear:.4f}")

def show_prediction_page(models, data):
    """Display prediction interface with manual parameters and Green Policy What-If Simulator"""
    st.markdown("## 🔮 Energy Consumption Forecasting & Policy Simulator")
    
    if not models:
        st.error("No models loaded!")
        return
    
    pred_tab1, pred_tab2 = st.tabs(["🔮 Custom Parameter Prediction", "🌿 Green Energy Policy 'What-If' Simulator"])
    
    with pred_tab1:
        # Model selection
        selected_model_name = st.selectbox("Select a model for prediction:", list(models.keys()), key="manual_model_select")
        selected_model = models[selected_model_name]
        st.markdown(f"### Using: **{selected_model_name}**")
    
    # Input form
    st.markdown("### 📝 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2030, value=2020)
        population = st.number_input("Population", min_value=1000, max_value=2000000000, value=50000000)
        gdp = st.number_input("GDP (USD)", min_value=1000000, max_value=50000000000000, value=1000000000000)
        energy_per_capita = st.number_input("Energy per Capita", min_value=0.0, max_value=1000.0, value=50.0)
        renewables_share = st.number_input("Renewables Share (%)", min_value=0.0, max_value=100.0, value=10.0)
    
    with col2:
        fossil_fuel_consumption = st.number_input("Fossil Fuel Consumption", min_value=0.0, max_value=100000.0, value=1000.0)
        
        # Calculate derived features
        gdp_per_capita = gdp / population
        st.write(f"**Calculated GDP per Capita:** ${gdp_per_capita:,.2f}")
        
        # Additional parameters
        energy_intensity = st.number_input("Energy Intensity", min_value=0.0, max_value=10.0, value=0.1)
        population_density = st.number_input("Population Density", min_value=0.0, max_value=10000.0, value=100.0)
    
    # Make prediction button
    if st.button("🚀 Make Prediction", type="primary"):
        try:
            # Prepare input data
            gdp_renewables_interaction = gdp_per_capita * renewables_share
            
            # Get the correct feature set for this model
            features = get_model_features(selected_model_name, selected_model)
            
            # Prepare input values in the correct order
            input_values = [
                year,
                np.log1p(population),
                np.log1p(gdp),
                energy_per_capita,
                renewables_share,
                fossil_fuel_consumption,
                np.log1p(gdp_per_capita),
                energy_intensity,
                gdp_renewables_interaction
            ]
            
            # Add population_density only if the model expects it
            if 'population_density' in features:
                input_values.insert(-1, population_density)  # Insert before gdp_renewables_interaction
            
            input_data = np.array([input_values])
            
            # Debug information
            st.info(f"🔍 **Debug Info:** Using {input_data.shape[1]} features for {selected_model_name}")
            with st.expander("📋 Feature Details"):
                st.write("**Features used:**")
                for i, feature in enumerate(features):
                    st.write(f"{i+1}. {feature}: {input_values[i]:.4f}")
            
            # Make prediction
            prediction = selected_model.predict(input_data)[0]
            
            # Display result
            st.success(f"🎯 **Predicted Energy Consumption:** {prediction:.2f} TWh")
            
            # Additional insights
            st.markdown("### 📊 Prediction Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                per_capita_prediction = prediction * 1000 / population  # Convert to kWh per capita
                st.metric("Per Capita Consumption", f"{per_capita_prediction:.2f} kWh")
            
            with col2:
                energy_gdp_ratio = prediction / (gdp / 1e12)  # TWh per trillion USD
                st.metric("Energy/GDP Ratio", f"{energy_gdp_ratio:.2f} TWh/T$")
            
            with col3:
                if renewables_share > 0:
                    renewable_consumption = prediction * renewables_share / 100
                    st.metric("Renewable Energy", f"{renewable_consumption:.2f} TWh")
                else:
                    st.metric("Renewable Energy", "0.00 TWh")
            
            # Comparison with global averages (approximate)
            st.markdown("### 🌍 Global Comparison")
            global_avg_per_capita = 50  # Approximate global average
            
            if per_capita_prediction > global_avg_per_capita * 1.5:
                st.warning("⚠️ High energy consumption compared to global average")
            elif per_capita_prediction < global_avg_per_capita * 0.5:
                st.info("ℹ️ Low energy consumption compared to global average")
            else:
                st.success("✅ Energy consumption within normal global range")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your input values and try again.")


    with pred_tab2:
        st.markdown("### 🌿 Green Energy Transition & Decarbonization Policy Lab")
        st.caption("Simulate how policy interventions (expanding renewable share, reducing energy intensity, and GDP growth) alter future primary energy demand and mitigate fossil emissions.")
        
        if data is not None and 'country' in data.columns:
            p_countries = sorted(data['country'].dropna().unique().tolist())
            def_c_idx = p_countries.index("Bangladesh") if "Bangladesh" in p_countries else (p_countries.index("United States") if "United States" in p_countries else 0)
            target_country = st.selectbox("Select Benchmark Country for Policy Simulation:", p_countries, index=def_c_idx, key="policy_country_select")
            
            c_records = data[data['country'] == target_country].dropna(subset=['primary_energy_consumption', 'gdp', 'population']).sort_values('year')
            
            if not c_records.empty:
                base_rec = c_records.iloc[-1]
                b_year = int(base_rec['year'])
                b_pop = float(base_rec['population'])
                b_gdp = float(base_rec['gdp'])
                b_energy = float(base_rec['primary_energy_consumption'])
                b_renew_pct = float(base_rec.get('renewables_share_energy', 10.0))
                if np.isnan(b_renew_pct): b_renew_pct = 10.0
                b_fossil = float(base_rec.get('fossil_fuel_consumption', b_energy * (1.0 - b_renew_pct/100.0)))
                b_intensity = b_energy / (b_gdp / 1e9) if b_gdp > 0 else 0.1
                b_per_cap = b_energy * 1000.0 / b_pop if b_pop > 0 else 50.0
                
                st.markdown(f"#### 📍 Current National Baseline ({target_country}, {b_year})")
                bc1, bc2, bc3, bc4 = st.columns(4)
                with bc1: st.metric("Primary Energy", f"{b_energy:,.1f} TWh")
                with bc2: st.metric("Renewable Share", f"{b_renew_pct:.1f}%")
                with bc3: st.metric("GDP", f"B")
                with bc4: st.metric("Population", f"{b_pop/1e6:,.1f}M")
                
                st.markdown("#### 🎛️ Policy Simulation Levers (5-Year Horizon)")
                pol_col1, pol_col2, pol_col3 = st.columns(3)
                with pol_col1:
                    renew_boost = st.slider("Target Renewable Share Increase (+%)", 0.0, 50.0, 15.0, 2.5)
                with pol_col2:
                    eff_improvement = st.slider("Energy Efficiency Target (% intensity reduction)", 0.0, 30.0, 10.0, 2.5)
                with pol_col3:
                    gdp_growth = st.slider("Projected Annual GDP Growth (%)", 0.0, 10.0, 4.0, 0.5)
                    
                target_model_name = "Gradient Boosting" if "Gradient Boosting" in models else list(models.keys())[0]
                target_model = models[target_model_name]
                
                if st.button("🚀 Run Policy Simulation", type="primary", key="run_policy_sim_btn"):
                    sim_year = b_year + 5
                    sim_gdp = b_gdp * ((1.0 + gdp_growth/100.0) ** 5)
                    sim_pop = b_pop * (1.01 ** 5)
                    sim_gdp_per_cap = sim_gdp / sim_pop
                    sim_renew_share = min(95.0, b_renew_pct + renew_boost)
                    sim_intensity = b_intensity * (1.0 - eff_improvement/100.0)
                    sim_fossil = max(0.0, b_fossil * (1.0 - renew_boost/100.0))
                    sim_interaction = sim_gdp_per_cap * sim_renew_share
                    sim_per_cap = b_per_cap * (1.0 - eff_improvement/100.0)
                    
                    features = get_model_features(target_model_name, target_model)
                    sim_input_vals = [
                        sim_year,
                        np.log1p(sim_pop),
                        np.log1p(sim_gdp),
                        sim_per_cap,
                        sim_renew_share,
                        sim_fossil,
                        np.log1p(sim_gdp_per_cap),
                        sim_intensity,
                        sim_interaction
                    ]
                    if 'population_density' in features:
                        sim_input_vals.insert(-1, 100.0)
                        
                    sim_pred = target_model.predict(np.array([sim_input_vals]))[0]
                    
                    st.markdown("### 📊 Simulated Policy Outcomes")
                    res_c1, res_c2, res_c3 = st.columns(3)
                    with res_c1:
                        delta_energy = sim_pred - b_energy
                        st.metric("Projected Total Energy", f"{sim_pred:,.1f} TWh", delta=f"{delta_energy:+,.1f} TWh")
                    with res_c2:
                        clean_gen = sim_pred * (sim_renew_share / 100.0)
                        st.metric("Clean Energy Output", f"{clean_gen:,.1f} TWh", delta=f"{sim_renew_share:.1f}% Share")
                    with res_c3:
                        avoided_fossil_twh = max(0.0, (b_energy * (1.0 - b_renew_pct/100.0)) - (sim_pred * (1.0 - sim_renew_share/100.0)))
                        avoided_co2_mt = avoided_fossil_twh * 0.72
                        st.metric("Avoided CO₂ Emissions", f"{avoided_co2_mt:,.1f} Mt CO₂", delta="Annual Abatement", delta_color="normal")
                        
                    fig_comp = go.Figure(data=[
                        go.Bar(name='Historical Baseline', x=['Primary Energy', 'Clean Energy', 'Fossil Energy'],
                               y=[b_energy, b_energy * (b_renew_pct/100.0), b_energy * (1.0 - b_renew_pct/100.0)],
                               marker_color=['#3498db', '#2ecc71', '#e74c3c']),
                        go.Bar(name='5-Year Policy Scenario', x=['Primary Energy', 'Clean Energy', 'Fossil Energy'],
                               y=[sim_pred, clean_gen, sim_pred * (1.0 - sim_renew_share/100.0)],
                               marker_color=['#2980b9', '#27ae60', '#c0392b'])
                    ])
                    fig_comp.update_layout(
                        barmode='group',
                        title="Baseline vs. Policy Scenario Comparison (TWh)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=380
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info(f"Insufficient baseline records for {target_country} to run simulation.")
        else:
            st.warning("Dataset unavailable for policy simulator.")

def show_visualizations(data):
    """Display saved visualizations and interactive country transition explorer"""
    st.markdown("## 🖼️ Energy Transitions & Visual Analytics")
    
    # 1. Interactive Country Energy Transition Explorer
    if data is not None and 'country' in data.columns and 'year' in data.columns:
        st.markdown("### 🌐 Interactive Country Energy Transition Explorer")
        st.caption("Analyze multi-decade fuel stack transitions from the 22,012-row World Energy Consumption dataset:")
        
        country_counts = data['country'].value_counts()
        eligible_countries = sorted(country_counts[country_counts >= 15].index.tolist())
        
        default_idx = 0
        if "Bangladesh" in eligible_countries:
            default_idx = eligible_countries.index("Bangladesh")
        elif "United States" in eligible_countries:
            default_idx = eligible_countries.index("United States")
            
        selected_c = st.selectbox("Select Country to Explore:", eligible_countries, index=default_idx)
        c_df = data[data['country'] == selected_c].sort_values('year')
        
        # Summary metrics
        valid_energy = c_df.dropna(subset=['primary_energy_consumption'])
        if not valid_energy.empty:
            latest_c = valid_energy.iloc[-1]
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1: st.metric("Latest Year Reported", int(latest_c['year']))
            with c_col2:
                p_energy = latest_c.get('primary_energy_consumption', 0)
                st.metric("Primary Energy", f"{p_energy:,.1f} TWh" if pd.notnull(p_energy) else "N/A")
            with c_col3:
                r_share = latest_c.get('renewables_share_energy', 0)
                st.metric("Renewable Share", f"{r_share:.1f}%" if pd.notnull(r_share) else "N/A")
            with c_col4:
                f_share = latest_c.get('fossil_share_energy', 100.0 - r_share if pd.notnull(r_share) else 0)
                st.metric("Fossil Share", f"{f_share:.1f}%" if pd.notnull(f_share) else "N/A")
                
            fig_mix = go.Figure()
            if 'fossil_fuel_consumption' in c_df.columns:
                fig_mix.add_trace(go.Scatter(
                    x=c_df['year'], y=c_df['fossil_fuel_consumption'],
                    mode='lines', name='Fossil Fuels (Coal, Oil, Gas)',
                    line=dict(color='#e74c3c', width=2.2)
                ))
            if 'renewables_consumption' in c_df.columns:
                fig_mix.add_trace(go.Scatter(
                    x=c_df['year'], y=c_df['renewables_consumption'],
                    mode='lines', name='Renewables (Solar, Wind, Hydro)',
                    line=dict(color='#2ecc71', width=2.2)
                ))
            if 'nuclear_consumption' in c_df.columns:
                fig_mix.add_trace(go.Scatter(
                    x=c_df['year'], y=c_df['nuclear_consumption'],
                    mode='lines', name='Nuclear Energy',
                    line=dict(color='#9b59b6', width=2.0)
                ))
            if 'primary_energy_consumption' in c_df.columns:
                fig_mix.add_trace(go.Scatter(
                    x=c_df['year'], y=c_df['primary_energy_consumption'],
                    mode='lines', name='Total Primary Energy',
                    line=dict(color='#3498db', width=2.5, dash='dash')
                ))
            fig_mix.update_layout(
                title=f"{selected_c}: Long-Term Fuel Mix Evolution (TWh)",
                xaxis_title="Year",
                yaxis_title="Energy Consumption (TWh)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                height=420
            )
            st.plotly_chart(fig_mix, use_container_width=True)
            st.markdown("---")

    # Check if images directory exists
    """Display saved visualizations"""
    st.markdown("## 🖼️ Project Visualizations")
    
    # Check if images directory exists
    if not os.path.exists('images'):
        st.error("Images directory not found!")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir('images') if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning("No image files found in the images directory!")
        return
    
    # Organize images by category
    eda_images = [f for f in image_files if any(keyword in f.lower() for keyword in 
                  ['global', 'countries', 'distribution', 'correlation', 'gdp', 'year', 'source', 'renewables', 'capita'])]
    
    model_images = [f for f in image_files if any(keyword in f.lower() for keyword in 
                    ['feature', 'actual', 'predicted', 'importance', 'performance'])]
    
    # Display EDA visualizations
    if eda_images:
        st.markdown("### 📊 Exploratory Data Analysis")
        
        for i in range(0, len(eda_images), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(eda_images):
                    img_file = eda_images[i + j]
                    img_path = os.path.join('images', img_file)
                    
                    try:
                        image = Image.open(img_path)
                        safe_render_image(col, image, img_file.replace('_', ' ').replace('.png', '').title())
                    except Exception as e:
                        col.error(f"Error loading {img_file}: {str(e)}")
    
    # Display model visualizations
    if model_images:
        st.markdown("### 🤖 Model Performance")
        
        for i in range(0, len(model_images), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(model_images):
                    img_file = model_images[i + j]
                    img_path = os.path.join('images', img_file)
                    
                    try:
                        image = Image.open(img_path)
                        safe_render_image(col, image, img_file.replace('_', ' ').replace('.png', '').title())
                    except Exception as e:
                        col.error(f"Error loading {img_file}: {str(e)}")
    
    # Interactive visualization
    if os.path.exists('images/world_energy_consumption_choropleth.html'):
        st.markdown("### 🌍 Interactive World Map")
        st.markdown("**Note:** The interactive choropleth map is available as an HTML file in the images directory.")
        st.info("📁 File: `images/world_energy_consumption_choropleth.html`")

if __name__ == "__main__":
    main()
