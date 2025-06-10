import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def predict_house_price(beds, baths, sqft):
    """
    Advanced house price prediction based on 3 features
    Returns: estimated price in USD
    """
    try:
        # Base price calculation - square footage is primary driver
        if sqft <= 800:
            base_price_per_sqft = 220  # Small homes premium
        elif sqft <= 1200:
            base_price_per_sqft = 200  # Compact homes
        elif sqft <= 1800:
            base_price_per_sqft = 180  # Medium homes
        elif sqft <= 2500:
            base_price_per_sqft = 165  # Large homes
        elif sqft <= 3500:
            base_price_per_sqft = 155  # Very large homes
        else:
            base_price_per_sqft = 145  # Luxury/estate homes
        
        # Calculate base price
        base_price = sqft * base_price_per_sqft
        
        # Bedroom adjustment multiplier
        if beds == 1:
            bedroom_multiplier = 0.75    # Studio/1BR significant discount
        elif beds == 2:
            bedroom_multiplier = 0.90    # 2BR moderate discount
        elif beds == 3:
            bedroom_multiplier = 1.00    # 3BR baseline
        elif beds == 4:
            bedroom_multiplier = 1.20    # 4BR premium
        elif beds == 5:
            bedroom_multiplier = 1.40    # 5BR high premium
        else:  # 6+ bedrooms
            bedroom_multiplier = 1.60    # Luxury home premium
        
        # Bathroom adjustment (dollar amount)
        if baths < 1.5:
            bathroom_adjustment = -25000   # Single bath penalty
        elif baths < 2.5:
            bathroom_adjustment = 0        # Standard 2 bath
        elif baths < 3.5:
            bathroom_adjustment = 35000    # Extra bath bonus
        elif baths < 4.5:
            bathroom_adjustment = 70000    # Multiple bath premium
        else:
            bathroom_adjustment = 110000   # Luxury bath suite premium
        
        # Apply bedroom multiplier to base price
        price_after_bedrooms = base_price * bedroom_multiplier
        
        # Add bathroom adjustment
        price_after_bathrooms = price_after_bedrooms + bathroom_adjustment
        
        # Market and location factor (simulated premium based on overall features)
        market_factor = 1.0 + (beds * 0.015) + (baths * 0.025) + (sqft * 0.000008)
        
        # Calculate final price
        final_price = price_after_bathrooms * market_factor
        
        # Apply reasonable bounds
        final_price = max(40000, min(final_price, 8000000))
        
        return round(final_price, 0)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def get_property_category(price):
    """Categorize property based on price"""
    if price < 150000:
        return "🏠 Starter Home", "blue"
    elif price < 300000:
        return "🏡 Family Home", "green"
    elif price < 600000:
        return "🏘️ Premium Home", "orange"
    elif price < 1000000:
        return "🏛️ Luxury Home", "purple"
    else:
        return "🏰 Estate/Mansion", "red"

def create_comparison_chart(beds, baths, sqft, predicted_price):
    """Create comparison chart with similar properties"""
    
    comparisons = [
        {'Property': 'Your Home', 'Price': predicted_price, 'Type': 'Target'},
        {'Property': 'Similar -1 Bed', 'Price': predict_house_price(max(1, beds-1), baths, sqft), 'Type': 'Comparison'},
        {'Property': 'Similar +1 Bed', 'Price': predict_house_price(beds+1, baths, sqft), 'Type': 'Comparison'},
        {'Property': 'Similar -0.5 Bath', 'Price': predict_house_price(beds, max(1, baths-0.5), sqft), 'Type': 'Comparison'},
        {'Property': 'Similar +0.5 Bath', 'Price': predict_house_price(beds, baths+0.5, sqft), 'Type': 'Comparison'},
        {'Property': 'Similar -300 sqft', 'Price': predict_house_price(beds, baths, max(400, sqft-300)), 'Type': 'Comparison'},
        {'Property': 'Similar +300 sqft', 'Price': predict_house_price(beds, baths, sqft+300), 'Type': 'Comparison'}
    ]
    
    df = pd.DataFrame(comparisons)
    
    # Create color map
    colors = ['#FF6B6B' if x == 'Target' else '#4ECDC4' for x in df['Type']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Property'],
            y=df['Price'],
            marker_color=colors,
            text=[f'${x:,.0f}' for x in df['Price']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Price Comparison with Similar Properties",
        xaxis_title="Property Type",
        yaxis_title="Estimated Price ($)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_feature_impact_chart(beds, baths, sqft):
    """Show impact of each feature on price"""
    
    # Calculate baseline (minimum features)
    baseline_price = predict_house_price(1, 1, 600)
    
    # Calculate individual impacts
    bed_impact = predict_house_price(beds, 1, 600) - baseline_price
    bath_impact = predict_house_price(1, baths, 600) - baseline_price
    sqft_impact = predict_house_price(1, 1, sqft) - baseline_price
    
    features = ['Bedrooms', 'Bathrooms', 'Square Footage']
    impacts = [bed_impact, bath_impact, sqft_impact]
    
    fig = px.bar(
        x=features,
        y=impacts,
        title="Feature Impact on Price",
        color=impacts,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Price Impact ($)",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Get instant price estimates using Bedrooms, Bathrooms, and Square Footage")
    st.markdown("---")
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    # INPUT SECTION
    with col1:
        st.subheader("🏡 Enter Property Details")
        
        # Bedrooms input
        st.markdown('<div class="feature-input">', unsafe_allow_html=True)
        beds = st.number_input(
            "🛏️ Bedrooms",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help="Number of bedrooms in the property",
            key="beds_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Bathrooms input
        st.markdown('<div class="feature-input">', unsafe_allow_html=True)
        baths = st.number_input(
            "🚿 Bathrooms",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            help="Number of bathrooms (including half baths)",
            key="baths_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Square footage input
        st.markdown('<div class="feature-input">', unsafe_allow_html=True)
        sqft = st.number_input(
            "📐 Square Feet",
            min_value=400,
            max_value=8000,
            value=1500,
            step=100,
            help="Total living space in square feet",
            key="sqft_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        predict_btn = st.button(
            "💰 GET PRICE ESTIMATE", 
            type="primary", 
            use_container_width=True,
            help="Click to calculate estimated price"
        )
        
        # Quick presets
        st.markdown("### 🎯 Quick Presets")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Starter Home\n2BR/1.5BA/1000sqft", use_container_width=True):
                st.session_state.beds_input = 2
                st.session_state.baths_input = 1.5
                st.session_state.sqft_input = 1000
                st.rerun()
        
        with col_b:
            if st.button("Family Home\n4BR/2.5BA/2200sqft", use_container_width=True):
                st.session_state.beds_input = 4
                st.session_state.baths_input = 2.5
                st.session_state.sqft_input = 2200
                st.rerun()
    
    # RESULTS SECTION
    with col2:
        st.subheader("🎯 Price Prediction Results")
        
        if predict_btn:
            # Calculate prediction
            predicted_price = predict_house_price(beds, baths, sqft)
            
            if predicted_price:
                # Main price display
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.markdown(f"""
                ## 💰 Estimated Price
                # ${predicted_price:,.0f}
                
                **Property Configuration:**  
                🛏️ {beds} Bedrooms • 🚿 {baths} Bathrooms • 📐 {sqft:,} sqft
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Key metrics
                price_per_sqft = predicted_price / sqft
                category, color = get_property_category(predicted_price)
                
                # Metrics row
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Price per Sq Ft",
                        f"${price_per_sqft:,.0f}",
                        help="Cost per square foot of living space"
                    )
                
                with metric_col2:
                    st.metric(
                        "Property Category",
                        category,
                        help="Market category based on price range"
                    )
                
                with metric_col3:
                    bedroom_ratio = sqft / beds if beds > 0 else 0
                    st.metric(
                        "Sqft per Bedroom",
                        f"{bedroom_ratio:,.0f}",
                        help="Living space per bedroom"
                    )
                
                # Property insights
                st.markdown("### 📊 Property Analysis")
                
                insights = []
                if price_per_sqft < 120:
                    insights.append("💰 Excellent value per square foot")
                elif price_per_sqft > 220:
                    insights.append("💎 Premium price point")
                
                if beds >= 4:
                    insights.append("👨‍👩‍👧‍👦 Great for large families")
                
                if baths >= 3:
                    insights.append("🛁 Luxury bathroom amenities")
                
                if sqft >= 2500:
                    insights.append("🏠 Spacious living area")
                
                if sqft < 1000:
                    insights.append("🏡 Compact and efficient")
                
                # Display insights
                if insights:
                    for insight in insights:
                        st.markdown(f"• {insight}")
                
                # Charts section
                st.markdown("### 📈 Market Analysis")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    comparison_chart = create_comparison_chart(beds, baths, sqft, predicted_price)
                    st.plotly_chart(comparison_chart, use_container_width=True)
                
                with chart_col2:
                    impact_chart = create_feature_impact_chart(beds, baths, sqft)
                    st.plotly_chart(impact_chart, use_container_width=True)
                
                # Sensitivity analysis
                st.markdown("### 🔍 What-If Analysis")
                
                scenarios = [
                    {"name": "Current Property", "beds": beds, "baths": baths, "sqft": sqft},
                    {"name": "Add 1 Bedroom", "beds": beds + 1, "baths": baths, "sqft": sqft},
                    {"name": "Add 0.5 Bathroom", "beds": beds, "baths": baths + 0.5, "sqft": sqft},
                    {"name": "Add 200 sq ft", "beds": beds, "baths": baths, "sqft": sqft + 200},
                    {"name": "Reduce 1 Bedroom", "beds": max(1, beds - 1), "baths": baths, "sqft": sqft},
                ]
                
                scenario_data = []
                for scenario in scenarios:
                    scenario_price = predict_house_price(scenario["beds"], scenario["baths"], scenario["sqft"])
                    price_change = scenario_price - predicted_price
                    percent_change = (price_change / predicted_price) * 100 if predicted_price > 0 else 0
                    
                    scenario_data.append({
                        "Scenario": scenario["name"],
                        "Estimated Price": f"${scenario_price:,.0f}",
                        "Price Change": f"${price_change:+,.0f}" if price_change != 0 else "$0",
                        "% Change": f"{percent_change:+.1f}%" if price_change != 0 else "0.0%"
                    })
                
                scenario_df = pd.DataFrame(scenario_data)
                st.dataframe(scenario_df, use_container_width=True, hide_index=True)
                
            else:
                st.error("❌ Unable to calculate price estimate")
        
        else:
            # Default state
            st.info("👆 Enter your property details and click 'GET PRICE ESTIMATE' to see results")
            
            # Show sample prediction
            st.markdown("### 📋 Sample Estimates")
            sample_properties = [
                {"desc": "2BR/1BA Starter Home (900 sqft)", "beds": 2, "baths": 1, "sqft": 900},
                {"desc": "3BR/2BA Family Home (1600 sqft)", "beds": 3, "baths": 2, "sqft": 1600},
                {"desc": "4BR/3BA Premium Home (2400 sqft)", "beds": 4, "baths": 3, "sqft": 2400},
            ]
            
            for prop in sample_properties:
                sample_price = predict_house_price(prop["beds"], prop["baths"], prop["sqft"])
                st.markdown(f"**{prop['desc']}**: ${sample_price:,.0f}")
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ How This Works"):
        st.markdown("""
        **Prediction Algorithm:**
        
        🏠 **Base Price Calculation:**
        - Square footage is the primary driver
        - Price per sq ft varies by home size (smaller homes have higher $/sqft)
        
        🛏️ **Bedroom Impact:**
        - 1BR: 25% discount from baseline
        - 2BR: 10% discount 
        - 3BR: Standard baseline
        - 4BR: 20% premium
        - 5+BR: 40%+ luxury premium
        
        🚿 **Bathroom Impact:**
        - Under 1.5 baths: $25K penalty
        - 2-2.5 baths: Standard
        - 3+ baths: $35K-$110K premium
        
        📍 **Market Factors:**
        - Location premium based on overall features
        - Final price bounded between $40K - $8M
        
        **Note:** This is a simplified model for demonstration. Actual home values depend on many additional factors including location, condition, market trends, and local amenities.
        """)

if __name__ == "__main__":
    main()