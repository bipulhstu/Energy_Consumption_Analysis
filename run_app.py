#!/usr/bin/env python3
"""
Energy Consumption Analysis - Streamlit App Runner
This script helps run the Streamlit application with proper setup.
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import plotly
        import joblib
        from PIL import Image
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements using: pip install -r requirements.txt")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'World Energy Consumption.csv',
        'models/',
        'images/'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files/directories:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files are present!")
    return True

def run_app():
    """Run the Streamlit application"""
    print("🚀 Starting Energy Consumption Analysis Dashboard...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    print("Press Ctrl+C to stop the application")
    print("="*50 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main function"""
    print("🌍 Energy Consumption Analysis - App Runner")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check files
    if not check_files():
        return
    
    # Run the app
    run_app()

if __name__ == "__main__":
    main()
