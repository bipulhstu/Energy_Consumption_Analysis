# 🚀 Deployment Guide - Energy Consumption Analysis Dashboard

This guide will help you deploy and run the Energy Consumption Analysis Streamlit application.

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM recommended
- Modern web browser

### Required Files
Ensure you have the following files in your project directory:
```
Energy_Consumption_Analysis/
├── app.py                    # Main Streamlit application
├── run_app.py               # Application runner script
├── requirements.txt         # Python dependencies
├── World Energy Consumption.csv  # Dataset
├── models/                  # Trained models directory
│   ├── best_model.pkl
│   ├── gradientboosting_model.pkl
│   ├── randomforest_model.pkl
│   └── ... (other model files)
├── images/                  # Generated visualizations
│   ├── global_energy_consumption_over_time.png
│   ├── feature_importances.png
│   └── ... (other image files)
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## 🛠️ Installation

### Method 1: Using the Runner Script (Recommended)

1. **Navigate to project directory:**
```bash
cd Energy_Consumption_Analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python run_app.py
```

### Method 2: Direct Streamlit Command

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run Streamlit directly:**
```bash
streamlit run app.py
```

## 🌐 Accessing the Application

Once the application starts:
- **Local URL:** http://localhost:8501
- The app will automatically open in your default web browser
- If it doesn't open automatically, copy the URL from the terminal

## 📱 Application Features

### 🏠 Home Page
- Project overview and statistics
- Quick metrics and dataset information
- Key project highlights

### 📊 Data Overview
- Dataset information and statistics
- Data preview and summary
- Missing values analysis
- Geographic coverage details

### 🤖 Model Testing
- Interactive model testing interface
- Real-time performance metrics
- Prediction vs actual visualizations
- Residuals analysis
- Feature importance plots

### 📈 Model Comparison
- Performance rankings table
- Interactive comparison charts
- Model insights and recommendations
- Tree vs Linear model analysis

### 🔮 Make Predictions
- Interactive prediction interface
- Real-time energy consumption forecasting
- Input parameter validation
- Prediction insights and comparisons

### 🖼️ Visualizations
- All generated project visualizations
- EDA plots and charts
- Model performance visualizations
- Interactive elements

## 🔧 Configuration

### Streamlit Configuration
The app uses a custom configuration file (`.streamlit/config.toml`) with:
- Custom theme colors
- Optimized server settings
- Enhanced user experience settings

### Model Loading
- Models are automatically cached for performance
- Supports all trained model types
- Graceful error handling for missing models

## 🐛 Troubleshooting

### Common Issues

1. **"Dataset not found" error:**
   - Ensure `World Energy Consumption.csv` is in the project root
   - Check file permissions

2. **"Models not loaded" error:**
   - Verify the `models/` directory exists
   - Check that model files (.pkl) are present
   - Ensure proper file permissions

3. **"Images not found" error:**
   - Verify the `images/` directory exists
   - Check that image files (.png) are present

4. **Import errors:**
   - Run: `pip install -r requirements.txt`
   - Check Python version compatibility

5. **Port already in use:**
   - Stop other Streamlit applications
   - Or use: `streamlit run app.py --server.port 8502`

### Performance Issues

1. **Slow loading:**
   - Reduce sample size in model testing
   - Clear browser cache
   - Restart the application

2. **Memory issues:**
   - Close other applications
   - Reduce dataset size for testing
   - Use a machine with more RAM

## 🌐 Deployment Options

### Local Development
- Use the provided runner script
- Perfect for testing and development

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository

#### Heroku
1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
2. Deploy using Heroku CLI

#### Docker
1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📊 Usage Tips

### Best Practices
1. **Model Testing:**
   - Start with smaller sample sizes
   - Compare multiple models
   - Analyze residuals for insights

2. **Predictions:**
   - Use realistic input values
   - Compare with global averages
   - Consider economic context

3. **Visualizations:**
   - Use full-screen mode for better viewing
   - Download images for presentations
   - Explore interactive features

### Performance Optimization
- Use caching for large datasets
- Limit sample sizes for testing
- Close unused browser tabs

## 🔒 Security Considerations

- The app runs locally by default
- No sensitive data is transmitted
- Model files are loaded locally
- Consider authentication for production deployment

## 📞 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify all requirements are met
3. Check the console for error messages
4. Ensure all files are present and accessible

## 🎯 Next Steps

After successful deployment:
1. Explore all application features
2. Test different models and parameters
3. Generate predictions for analysis
4. Use visualizations for presentations
5. Consider extending functionality

---

**Happy Analyzing! 🌍⚡**
