import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# --- Global Streamlit Page Configuration (can only be called once) ---
st.set_page_config(
    page_title="Mattress Company Call Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Consolidated Custom CSS for Styling ---
# Merged CSS from both original dashboards.
# Added specific styling for st.tabs to ensure visibility.
st.markdown(f"""
<style>
/* Main Block Container (from Outbound) */
.block-container {{
    padding-top: 1rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;
}}

/* Metric Card Styling (from Outbound, similar to Inbound) */
div[data-testid="metric-container"] {{
    background-color: #f0f2f6;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
}}

div[data-testid="stMetricLabel"] p {{
    font-size: 16px; /* Label font size, as requested */
    font-weight: bold;
    color: #555555; /* Darker label color */
}}

div[data-testid="stMetricValue"] {{
    font-size: 36px; /* Value font size */
    font-weight: bold;
    color: #267EBB; /* Mariner from CUSTOM_COLORS */
}}

/* Header Styling (from Outbound, similar to Inbound) */
h1 {{
    color: #183F5E; /* Biscay from CUSTOM_COLORS */
    text-align: center;
    font-size: 36px; /* Increased for better visibility */
    margin-bottom: 6px; /* Added for global title spacing */
}}
h2 {{
    color: #267EBB; /* Mariner from CUSTOM_COLORS */
    border-bottom: 2px solid #73B4B6; /* Neptune from CUSTOM_COLORS */
    padding-bottom: 5px;
    margin-top: 40px; /* From Outbound, overrides Inbound's 20px for consistency */
    font-size: 28px; /* Increased for better visibility */
}}
h5 {{
    color: #183F5E; /* Biscay from CUSTOM_COLORS */
    margin-top: 15px;
    margin-bottom: 5px;
    font-size: 20px; /* Increased for better visibility */
}}

/* General paragraph text size (from Inbound) */
p {{
    font-size: 16px; /* All paragraph text should be at least 16px */
}}

/* Adjust font size for markdown lists and other general text that might not be a direct <p> (from Inbound) */
ul, ol, li, div.stMarkdown, div.stInfo, div.stWarning {{
    font-size: 16px !important; /* Force font size for info/warning boxes and list items */
}}

/* Specific adjustment for sidebar elements (from Inbound) */
.stMultiSelect p {{
    font-size: 16px !important;
}}
.stFileUploader p {{
    font-size: 16px !important;
}}

/* Business Question Strip Styling (from Inbound, slightly adjusted for consistency) */
/* This combines features from both original question strip styles */
.question-strip {{
    background-color: #E6EEF6; /* Light blue-gray background from Inbound */
    padding: 15px 20px; /* From Outbound */
    margin-top: 30px; /* Space above the strip, from Inbound */
    margin-bottom: 20px; /* Space below the strip, from Inbound */
    border-radius: 10px; /* From Outbound */
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* From Outbound */
}}

.question-strip h3 {{
    color: #333333; /* Darker text for the question */
    margin: 0;
    font-size: 20px; /* Slightly larger for the question */
    text-align: center; /* Centered from Outbound */
}}

/* NEW: Explicit Tab Text Styling for Visibility (Crucial for tab labels) */
/* Target the paragraph text within the tab buttons */
button[data-testid="stTabButton"] p {{
    color: #333333 !important; /* Dark grey for visibility of inactive tabs */
    font-weight: bold !important; /* Make them stand out */
    font-size: 18px !important; /* Slightly larger than general text for prominence */
}}

/* Style for the active tab's text */
button[data-testid="stTabButton"][aria-selected="true"] p {{
    color: #183F5E !important; /* Biscay for active tab text */
}}

/* Optional: Style the tab list container for better visual separation */
div[role="tablist"] {{
    background-color: #f8f8f8 !important; /* A very light background */
    border-bottom: 1px solid #e0e0e0 !important; /* A subtle separator */
    padding-bottom: 5px; /* Give some space for the underline */
    margin-bottom: 20px; /* Space between tabs and dashboard content */
    border-radius: 8px; /* Slightly rounded corners */
}}

</style>
""", unsafe_allow_html=True)


# --- Dashboard Function for Outbound Call Analytics (Exact original code) ---
def run_outbound_dashboard(uploaded_file):
    # --- Custom Color Palette --- (Original from Outbound code)
    CUSTOM_COLORS = {
        'Biscay': '#183F5E',    # Darkest Blue (Negative/Strong emphasis)
        'Edgewater': '#BADED7', # Lightest Blue-Green (Positive)
        'Neptune': '#73B4B6',   # Mid Blue-Green (Neutral/Service-related)
        'Mariner': '#267EBB'    # Medium Blue (Sales-related/General)
    }

    # Mapping for Sentiment labels (Original from Outbound code)
    SENTIMENT_COLOR_MAP = {
        'Positive': CUSTOM_COLORS['Edgewater'], # Lightest for positive
        'Neutral': CUSTOM_COLORS['Neptune'],    # Mid-tone for neutral
        'Negative': CUSTOM_COLORS['Biscay'],    # Darkest for negative
        'Unknown': '#CCCCCC' # A neutral grey for unknown/unspecified (though 'Unknown' will be filtered for plots)
    }

    # Mapping for Agent Rating (Original from Outbound code)
    AGENT_RATING_COLOR_MAP = {
        'High': CUSTOM_COLORS['Edgewater'],  # Positive
        'Medium': CUSTOM_COLORS['Neptune'],  # Neutral
        'Low': CUSTOM_COLORS['Biscay'],     # Negative
        'Unknown': '#CCCCCC' # (will be filtered for plots)
    }

    # Helper functions (Original from Outbound code)
    @st.cache_data
    def load_data_outbound(file): # Renamed to avoid conflict with Inbound's load_data
        """Loads and preprocesses data from an uploaded CSV file."""
        try:
            string_data = io.StringIO(file.getvalue().decode('utf-8'))
            df = pd.read_csv(string_data)
            df.columns = df.columns.str.strip()

            if 'CallID' in df.columns:
                df = df[df['CallID'] != 'PARSE_FAILED']

            nan_fill_defaults = {
                'UserIntentToBuy': 'Not Specified', 'CallSentiment': 'Unknown',
                'Major Purchase Barrier Theme': 'Not Specified', 'Top3Themes': 'Not Specified',
                'AgentNextAction': 'Unknown', 'AgentRating': 'Unknown', 'Call Type': 'Unknown'
            }
            for col, default_val in nan_fill_defaults.items():
                if col in df.columns:
                    # Replace 'nan', 'N/A', and empty strings with default_val
                    df[col] = df[col].astype(str).replace(['nan', 'N/A', ''], default_val, regex=False)
                    df[col] = df[col].fillna(default_val).str.strip()
                else:
                    df[col] = default_val
            return df
        except Exception as e:
            st.error(f"Error processing uploaded file for Outbound: {e}. Please ensure it's a valid CSV.")
            return pd.DataFrame()

    def find_col_outbound(df_columns, possible_names): # Renamed
        """Finds a column name, case-insensitively."""
        df_columns_lower = {c.lower(): c for c in df_columns}
        for name in possible_names:
            if name.lower() in df_columns_lower:
                return df_columns_lower[name.lower()]
        return None

    def ordered_intent_counts_outbound(df_filtered, col): # Renamed
        """Orders user intent counts by a predefined order and includes percentages."""
        order = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Not Specified']
        if df_filtered.empty or col not in df_filtered.columns:
            return pd.DataFrame()

        counts = df_filtered[col].value_counts().reindex(order, fill_value=0)
        out_df = counts.reset_index()
        out_df.columns = ['intent', 'count']
        out_df = out_df[out_df['count'] > 0] # Filter out intents with zero count
        total = out_df['count'].sum()
        if total > 0:
            out_df['pct'] = (out_df['count'] / total * 100).round(1)
            out_df['text_label'] = out_df.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
        else:
            out_df['pct'], out_df['text_label'] = 0, "0 (0.0%)"

        out_df['intent'] = pd.Categorical(out_df['intent'], categories=order, ordered=True)
        out_df = out_df.sort_values('intent')
        return out_df

    def display_placeholder_outbound(message, height_px=250): # Renamed
        """Displays an info message with a specific height."""
        st.info(message)
        st.markdown(f"<div style='height: {height_px-80}px;'></div>", unsafe_allow_html=True) # Adjust 80px for st.info height

    def display_business_question_outbound(question_text): # Renamed
        """Displays a business question in a styled strip."""
        # Note: The CSS for .question-strip is now global for consistency.
        st.markdown(f'<div class="question-strip"><h3>{question_text}</h3></div>', unsafe_allow_html=True)

    # --- Start of Main Dashboard Logic (Original from Outbound code) ---

    if not uploaded_file:
        st.info("Upload your Outbound Call Analysis CSV from the sidebar to view this dashboard.")
        return # Use return instead of st.stop()

    df = load_data_outbound(uploaded_file)
    if df.empty:
        st.warning("Outbound data could not be loaded or is empty after processing. Please check the file content and try again.")
        return # Use return instead of st.stop()

    # Find columns (Original from Outbound code)
    col_calltype = find_col_outbound(df.columns, ['Call Type'])
    col_intent = find_col_outbound(df.columns, ['UserIntentToBuy'])
    col_sentiment = find_col_outbound(df.columns, ['CallSentiment'])
    col_barrier = find_col_outbound(df.columns, ['Major Purchase Barrier Theme'])
    col_top3themes = find_col_outbound(df.columns, ['Top3Themes'])
    col_nextaction = find_col_outbound(df.columns, ['AgentNextAction'])
    col_agent_rating = find_col_outbound(df.columns, ['AgentRating'])
    col_city = find_col_outbound(df.columns, ['City'])

    if not col_calltype:
        st.error("Mandatory column 'Call Type' not found in Outbound data. Please check your CSV."); return # Use return

    # Sidebar Filters (Original from Outbound code, using unique keys)
    st.sidebar.markdown("### Filter Outbound Data") # Changed from st.sidebar.header
    filtered_df = df.copy()
    if col_city:
        all_cities = ['All'] + sorted(df[col_city].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox('Select City (Outbound)', options=all_cities, key='outbound_city_filter')
        if selected_city != 'All':
            filtered_df = df[df[col_city] == selected_city]
    else:
        st.sidebar.info("No 'City' column found for Outbound filtering.") # Added info

    if filtered_df.empty:
        st.warning("No data available for the selected Outbound filters. Please adjust your filter selections or upload a different file."); return # Use return

    # Create dedicated dataframes for Sales and Service (Original from Outbound code)
    sales_df = filtered_df[filtered_df[col_calltype].str.lower() == 'sale']
    service_df = filtered_df[filtered_df[col_calltype].str.lower() == 'service']

    # --- Main Dashboard Layout (Original from Outbound code) ---
    st.title("Outbound Call Analytics Dashboard")
    st.markdown("Insights into customer interactions from outbound calls – sales and service performance at a glance.")

    # --- Overall KPIs (Top Metric) (Original from Outbound code) ---
    total_calls = len(filtered_df)
    num_sales = len(sales_df)
    num_service = len(service_df)

    col_metric, col_spacer = st.columns([1, 3])

    with col_metric:
        st.metric(label="Total Calls Analysed (Outbound)", value=f"{total_calls:,}", delta=f"Sales {num_sales:,} • Service {num_service:,}")

    st.markdown("---")

    # --- Overall Insights (Original from Outbound code) ---
    st.header("Overall Insights")
    display_business_question_outbound("Q1. What type of Calls do we get?")
    call_type_counts = filtered_df[col_calltype].value_counts().reset_index()
    call_type_counts.columns = ['Call Type', 'Count']
    total_calls_q1 = call_type_counts['Count'].sum()
    if total_calls_q1 > 0:
        call_type_counts['Percentage'] = (call_type_counts['Count'] / total_calls_q1 * 100).round(1)
        call_type_counts['text_label'] = call_type_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

        fig = px.bar(call_type_counts, x='Call Type', y='Count', text='text_label',
                     color='Call Type', color_discrete_map={'Sale': CUSTOM_COLORS['Mariner'], 'Service': CUSTOM_COLORS['Neptune']})
        fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
        fig.update_layout(height=350, showlegend=False, yaxis_title='Number of Calls',
                          xaxis_title="", # Removed "Call Type" as it's clear from categories
                          yaxis_range=[0, call_type_counts['Count'].max() * 1.2],
                          xaxis_title_font_size=16, yaxis_title_font_size=16,
                          xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                          bargap=1.0) # Apply bargap here
        st.plotly_chart(fig, use_container_width=True)

    else:
        display_placeholder_outbound("No 'Call Type' data to display.")

    st.markdown("---")

    # --- Sales Insights (Original from Outbound code) ---
    st.header("Sales Insights")

    display_business_question_outbound("Q2. What is the User Intent to Buy in Sales calls?")
    if col_intent and not sales_df.empty:
        intent_df = ordered_intent_counts_outbound(sales_df, col_intent) # This already generates text_label
        if not intent_df.empty:
            fig = px.bar(intent_df, x='intent', y='count', text='text_label',
            labels={'intent': 'Intent Level', 'count': 'Number of Calls'},
            color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
            fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
            fig.update_layout(height=400, yaxis_title='Number of Calls', xaxis_title='Intent Level',
            yaxis_range=[0, intent_df['count'].max() * 1.25],
            xaxis_title_font_size=16, yaxis_title_font_size=16,
            xaxis_tickfont_size=14, yaxis_tickfont_size=14
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_placeholder_outbound("No 'UserIntentToBuy' data available for Sales calls.")
    else:
        display_placeholder_outbound("Column 'UserIntentToBuy' not found or no Sales calls in data.")

    display_business_question_outbound("Q3. What is the sentiment of Sales Calls, broken down by Intent?")
    if col_intent and col_sentiment and not sales_df.empty:
        # Get raw counts first
        raw_counts_cross_tab = pd.crosstab(index=sales_df[col_intent], columns=sales_df[col_sentiment])

        # Get percentages (normalize by index means row-wise percentage)
        pct_cross_tab = pd.crosstab(index=sales_df[col_intent], columns=sales_df[col_sentiment], normalize='index').mul(100).round(1)

        sentiment_display_order = ['Positive', 'Neutral', 'Negative']
        intent_order_for_plot = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

        # Reindex both tables to ensure consistent columns and row order
        raw_counts_cross_tab = raw_counts_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in raw_counts_cross_tab.index],
                                                            columns=sentiment_display_order, fill_value=0)
        pct_cross_tab = pct_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in pct_cross_tab.index],
                                              columns=sentiment_display_order, fill_value=0)

        # Melt both into long format
        plot_df_counts = raw_counts_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Count')
        plot_df_pct = pct_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Percentage')

        # Merge to combine counts and percentages
        plot_df = pd.merge(plot_df_counts, plot_df_pct, on=[col_intent, 'Sentiment'])

        # Create the combined text label
        plot_df['text_label'] = plot_df.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)" if row['Count'] > 0 else "", axis=1)

        # Filter out rows where Percentage is 0 if text is empty (to avoid clutter if there's no data for that segment)
        plot_df = plot_df[plot_df['Count'] > 0] # Only plot segments that actually have calls

        fig = px.bar(plot_df, x=col_intent, y='Percentage', color='Sentiment', barmode='group',
                     text='text_label', # Use the new combined label
                     color_discrete_map=SENTIMENT_COLOR_MAP,
                     category_orders={col_intent: intent_order_for_plot, "Sentiment": sentiment_display_order},
                     labels={'Percentage': 'Percentage of Calls', col_intent: 'User Intent to Buy', 'Count': 'Number of Calls'})
        fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=14)
        fig.update_layout(height=500, yaxis_title='Sentiment Distribution (%)', legend_title_text='Sentiment', yaxis_range=[0, 115],
                          xaxis_title_font_size=16, yaxis_title_font_size=16,
                          xaxis_tickfont_size=14, yaxis_tickfont_size=14
                          )
        st.plotly_chart(fig, use_container_width=True)

    else:
        display_placeholder_outbound("Cannot generate Intent vs. Sentiment chart. Check for 'UserIntentToBuy', 'CallSentiment', and Sales calls.")

    display_business_question_outbound("Q4. What are the reasons why Sales Customers are not Converting?")
    if col_barrier and not sales_df.empty:
        barrier_counts = sales_df[col_barrier].value_counts().reset_index()
        barrier_counts.columns = ['Barrier', 'Count']
        # Filter out non-descriptive barriers including 'Not Specified' and 'No Barrier / Info Unavailable'
        barrier_counts = barrier_counts[~barrier_counts['Barrier'].isin(['Not Specified', 'No Barrier / Info Unavailable'])]

        total_barriers_q4 = barrier_counts['Count'].sum()
        if total_barriers_q4 > 0:
            barrier_counts['Percentage'] = (barrier_counts['Count'] / total_barriers_q4 * 100).round(1)
            barrier_counts['text_label'] = barrier_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not barrier_counts.empty:
                fig = px.bar(barrier_counts.head(10).sort_values('Count', ascending=True),
                             x='Count', y='Barrier', orientation='h', text='text_label', title="Top Purchase Barriers in Sales Calls",
                             color_discrete_sequence=[CUSTOM_COLORS['Biscay']])
                fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
                fig.update_layout(height=400, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, barrier_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  yaxis_automargin=True,
                                  )
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder_outbound("No significant purchase barrier data found for Sales calls.")
        else:
            display_placeholder_outbound("No significant purchase barrier data found for Sales calls.")

    else:
        display_placeholder_outbound("Column 'Major Purchase Barrier Theme' not found or no Sales calls in data.")

    display_business_question_outbound("Q5. What are the Talking Points in a Sales Call?")
    if col_top3themes and not sales_df.empty:
        themes = sales_df[col_top3themes].str.split(',').explode().str.strip()
        # Filter out non-descriptive themes
        themes = themes.dropna()[~themes.isin(['Not Specified', 'No Barrier / Info Unavailable', 'Other'])]

        total_themes_q5 = themes.shape[0] # Total mentions
        if total_themes_q5 > 0:
            theme_counts = themes.value_counts().reset_index()
            theme_counts.columns = ['Theme', 'Count']
            theme_counts['Percentage'] = (theme_counts['Count'] / total_themes_q5 * 100).round(1)
            theme_counts['text_label'] = theme_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not theme_counts.empty:
                fig = px.bar(theme_counts.head(10).sort_values('Count', ascending=True),
                             x='Count', y='Theme', orientation='h', text='text_label', title="Top 10 Talking Points in Sales Calls",
                             color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
                fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
                fig.update_layout(height=450, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, theme_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  yaxis_automargin=True
                                  )
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder_outbound("No talking points data available for Sales calls.")
        else:
            display_placeholder_outbound("No talking points data available for Sales calls.")

    else:
        display_placeholder_outbound("Column 'Top3Themes' not found or no Sales calls in data.")

    # Sales - Q6 and Q7 in columns
    col1, col2 = st.columns(2)
    with col1: # Wrapped in with col1
        display_business_question_outbound("Q6. How many Sales Calls Require a Follow-up?")
        if col_nextaction and not sales_df.empty:
            action_counts = sales_df[col_nextaction].value_counts()
            # Filter out 'Unknown'
            action_counts = action_counts[action_counts.index != 'Unknown']

            total_actions_q6 = action_counts.sum()
            if total_actions_q6 > 0:
                action_counts = action_counts.reset_index()
                action_counts.columns = ['Follow-up', 'Count']
                action_counts['Percentage'] = (action_counts['Count'] / total_actions_q6 * 100).round(1)
                action_counts['text_label'] = action_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

                if not action_counts.empty:
                    fig = px.bar(action_counts, x='Follow-up', y='Count', text='text_label', color='Follow-up',
                                 color_discrete_map={'Yes': CUSTOM_COLORS['Mariner'], 'No': CUSTOM_COLORS['Biscay']}) # No 'Unknown' in map
                    fig.update_traces(textposition='outside', cliponaxis=False, width=0.4, textfont_size=16)
                    fig.update_layout(height=400, title="Agent Follow-up for Sales", showlegend=False, yaxis_range=[0, action_counts['Count'].max() * 1.25],
                                      xaxis_title_font_size=16, yaxis_title_font_size=16,
                                      xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                      bargap=1.0) # Apply bargap here
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    display_placeholder_outbound("No 'AgentNextAction' data (excluding Unknown) for Sales calls.")
            else:
                display_placeholder_outbound("No 'AgentNextAction' data (excluding Unknown) for Sales calls.")

        else:
            display_placeholder_outbound("Column 'AgentNextAction' not found or no Sales calls.")

    with col2: # Wrapped in with col2
        display_business_question_outbound("Q7. How have Agents performed in Sales Calls?")
        if col_agent_rating and not sales_df.empty:
            rating_counts = sales_df[col_agent_rating].value_counts()
            # Filter out 'Unknown'
            rating_counts = rating_counts[rating_counts.index != 'Unknown']

            total_ratings_q7 = rating_counts.sum()
            if total_ratings_q7 > 0:
                rating_order = ['High', 'Medium', 'Low'] # Explicit order for plotting, ensures consistency
                rating_counts = rating_counts.reindex(rating_order, fill_value=0).reset_index()
                rating_counts.columns = ['Rating', 'Count']
                # Filter out ratings with 0 count after reindexing if they were not present originally
                rating_counts = rating_counts[rating_counts['Count'] > 0]

                rating_counts['Percentage'] = (rating_counts['Count'] / total_ratings_q7 * 100).round(1)
                rating_counts['text_label'] = rating_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

                if not rating_counts.empty:
                    fig = px.bar(rating_counts, x='Rating', y='Count', text='text_label', color='Rating',
                                 color_discrete_map=AGENT_RATING_COLOR_MAP, title="Agent Performance in Sales")
                    fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
                    fig.update_layout(height=400, showlegend=False, yaxis_range=[0, rating_counts['Count'].max() * 1.25],
                                      xaxis_title_font_size=16, yaxis_title_font_size=16,
                                      xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                      bargap=1.0) # Apply bargap here
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    display_placeholder_outbound("No 'AgentRating' data (excluding Unknown) for Sales calls.")
            else:
                display_placeholder_outbound("No 'AgentRating' data (excluding Unknown) for Sales calls.")

        else:
            display_placeholder_outbound("Column 'AgentRating' not found or no Sales calls.")

    st.markdown("---")

    # --- Service Insights (Original from Outbound code) ---
    st.header("Service Insights")

    display_business_question_outbound("Q2.What is the sentiment of Service Calls?")
    if col_sentiment and not service_df.empty:
        sentiment_counts = service_df[col_sentiment].value_counts()
        # Filter out 'Unknown' sentiment
        sentiment_counts = sentiment_counts[sentiment_counts.index != 'Unknown']

        total_sentiment_service = sentiment_counts.sum()
        if total_sentiment_service > 0:
            sentiment_display_order = ['Positive', 'Neutral', 'Negative'] # Consistent order
            sentiment_counts = sentiment_counts.reindex(sentiment_display_order, fill_value=0).reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            # Filter out sentiments with 0 count after reindexing
            sentiment_counts = sentiment_counts[sentiment_counts['Count'] > 0]

            sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total_sentiment_service * 100).round(1)
            sentiment_counts['text_label'] = sentiment_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not sentiment_counts.empty:
                fig = px.bar(sentiment_counts, x='Sentiment', y='Count', text='text_label', color='Sentiment',
                             color_discrete_map=SENTIMENT_COLOR_MAP, category_orders={"Sentiment": sentiment_display_order})
                fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
                fig.update_layout(height=400, title="Sentiment Distribution in Service Calls", showlegend=False,
                                  yaxis_range=[0, sentiment_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  bargap=1.0) # Apply bargap here
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder_outbound("No sentiment data available for Service calls.")
        else:
            display_placeholder_outbound("No sentiment data available for Service calls.")

    else:
        display_placeholder_outbound("Column 'CallSentiment' not found or no Service calls.")

    display_business_question_outbound("Q3. What is the sentiment of Service Calls, broken down by Intent?")
    if col_intent and col_sentiment and not service_df.empty:
        # Get raw counts first
        raw_counts_cross_tab = pd.crosstab(index=service_df[col_intent], columns=service_df[col_sentiment])

        # Get percentages (normalize by index means row-wise percentage)
        pct_cross_tab = pd.crosstab(index=service_df[col_intent], columns=service_df[col_sentiment], normalize='index').mul(100).round(1)

        sentiment_display_order = ['Positive', 'Neutral', 'Negative']
        intent_order_for_plot = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

        # Reindex both tables to ensure consistent columns and row order
        raw_counts_cross_tab = raw_counts_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in raw_counts_cross_tab.index],
                                                            columns=sentiment_display_order, fill_value=0)
        pct_cross_tab = pct_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in pct_cross_tab.index],
                                              columns=sentiment_display_order, fill_value=0)

        # Melt both into long format
        plot_df_counts = raw_counts_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Count')
        plot_df_pct = pct_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Percentage')

        # Merge to combine counts and percentages
        plot_df = pd.merge(plot_df_counts, plot_df_pct, on=[col_intent, 'Sentiment'])

        # Create the combined text label
        plot_df['text_label'] = plot_df.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)" if row['Count'] > 0 else "", axis=1)

        # Filter out rows where Percentage is 0 if text is empty (to avoid clutter if there's no data for that segment)
        plot_df = plot_df[plot_df['Count'] > 0] # Only plot segments that actually have calls

        fig = px.bar(plot_df, x=col_intent, y='Percentage', color='Sentiment', barmode='group',
                     text='text_label', # Use the new combined label
                     color_discrete_map=SENTIMENT_COLOR_MAP,
                     category_orders={col_intent: intent_order_for_plot, "Sentiment": sentiment_display_order},
                     labels={'Percentage': 'Percentage of Calls', col_intent: 'User Intent to Buy', 'Count': 'Number of Calls'})
        fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=14)
        fig.update_layout(height=500, yaxis_title='Sentiment Distribution (%)', legend_title_text='Sentiment', yaxis_range=[0, 115],
                          xaxis_title_font_size=16, yaxis_title_font_size=16,
                          xaxis_tickfont_size=14, yaxis_tickfont_size=14
                          )
        st.plotly_chart(fig, use_container_width=True)

    else:
        display_placeholder_outbound("Cannot generate Intent vs. Sentiment chart. Check for 'UserIntentToBuy', 'CallSentiment', and Service calls.")

    display_business_question_outbound("Q4.What are the Top Issues in Service Calls?")
    if col_barrier and not service_df.empty:
        barrier_counts = service_df[col_barrier].value_counts().reset_index()
        barrier_counts.columns = ['Issue', 'Count']
        # Filter out non-descriptive issues
        barrier_counts = barrier_counts[~barrier_counts['Issue'].isin(['Not Specified', 'No Barrier / Info Unavailable'])]

        total_service_barriers = barrier_counts['Count'].sum()
        if total_service_barriers > 0:
            barrier_counts['Percentage'] = (barrier_counts['Count'] / total_service_barriers * 100).round(1)
            barrier_counts['text_label'] = barrier_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not barrier_counts.empty:
                fig = px.bar(barrier_counts.head(10).sort_values('Count', ascending=True),
                             x='Count', y='Issue', orientation='h', text='text_label', title="Top Issues Raised in Service Calls",
                             color_discrete_sequence=[CUSTOM_COLORS['Biscay']]) # Use Biscay for barriers
                fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
                fig.update_layout(height=400, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, barrier_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  yaxis_automargin=True
                                  )
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder_outbound("No significant issue data found for Service calls.")
        else:
            display_placeholder_outbound("No significant issue data found for Service calls.")

    else:
        display_placeholder_outbound("Column 'Major Purchase Barrier Theme' not found or no Service calls.")

    display_business_question_outbound("Q5.What are the Talking Points in a Service Call?")
    if col_top3themes and not service_df.empty:
        themes = service_df[col_top3themes].str.split(',').explode().str.strip()
        # Filter out non-descriptive themes
        themes = themes.dropna()[~themes.isin(['Not Specified', 'No Barrier / Info Unavailable', 'Other'])]

        total_service_themes = themes.shape[0] # Total mentions
        if total_service_themes > 0:
            theme_counts = themes.value_counts().reset_index()
            theme_counts.columns = ['Theme', 'Count']
            theme_counts['Percentage'] = (theme_counts['Count'] / total_service_themes * 100).round(1)
            theme_counts['text_label'] = theme_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not theme_counts.empty:
                fig = px.bar(theme_counts.head(10).sort_values('Count', ascending=True),
                             x='Count', y='Theme', orientation='h', text='text_label', title="Top 10 Talking Points in Service Calls",
                             color_discrete_sequence=[CUSTOM_COLORS['Neptune']]) # Use Neptune for service themes
                fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
                fig.update_layout(height=450, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, theme_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  yaxis_automargin=True
                                  )
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder_outbound("No talking points data available for Service calls.")
        else:
            display_placeholder_outbound("No talking points data available for Service calls.")

    else:
        display_placeholder_outbound("Column 'Top3Themes' not found or no Service calls.")

    # Service - Q6 and Q7 in columns
    col3, col4 = st.columns(2)
    with col3: # Wrapped in with col3
        display_business_question_outbound("Q6.How many Service Calls Require a Follow-up?")
        if col_nextaction and not service_df.empty:
            action_counts = service_df[col_nextaction].value_counts()
            # Filter out 'Unknown'
            action_counts = action_counts[action_counts.index != 'Unknown']

            total_service_actions = action_counts.sum()
            if total_service_actions > 0:
                action_counts = action_counts.reset_index()
                action_counts.columns = ['Follow-up', 'Count']
                action_counts['Percentage'] = (action_counts['Count'] / total_service_actions * 100).round(1)
                action_counts['text_label'] = action_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

                if not action_counts.empty:
                    fig = px.bar(action_counts, x='Follow-up', y='Count', text='text_label', color='Follow-up',
                                 color_discrete_map={'Yes': CUSTOM_COLORS['Neptune'], 'No': CUSTOM_COLORS['Biscay']})
                    fig.update_traces(textposition='outside', cliponaxis=False, width=0.4, textfont_size=16)
                    fig.update_layout(height=400, title="Agent Follow-up for Service", showlegend=False, yaxis_range=[0, action_counts['Count'].max() * 1.25],
                                      xaxis_title_font_size=16, yaxis_title_font_size=16,
                                      xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                      bargap=1.0) # Apply bargap here
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    display_placeholder_outbound("No 'AgentNextAction' data (excluding Unknown) for Service calls.")
            else:
                display_placeholder_outbound("No 'AgentNextAction' data (excluding Unknown) for Service calls.")

        else:
            display_placeholder_outbound("Column 'AgentNextAction' not found or no Service calls.")
    with col4: # Wrapped in with col4
        display_business_question_outbound("How have Agents performed in Service Calls?")
        if col_agent_rating and not service_df.empty:
            rating_counts = service_df[col_agent_rating].value_counts()
            # Filter out 'Unknown'
            rating_counts = rating_counts[rating_counts.index != 'Unknown']

            total_service_ratings = rating_counts.sum()
            if total_service_ratings > 0:
                rating_order = ['High', 'Medium', 'Low']
                rating_counts = rating_counts.reindex(rating_order, fill_value=0).reset_index()
                rating_counts.columns = ['Rating', 'Count']
                rating_counts = rating_counts[rating_counts['Count'] > 0]

                rating_counts['Percentage'] = (rating_counts['Count'] / total_service_ratings * 100).round(1)
                rating_counts['text_label'] = rating_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

                if not rating_counts.empty:
                    fig = px.bar(rating_counts, x='Rating', y='Count', text='text_label', color='Rating',
                                 color_discrete_map=AGENT_RATING_COLOR_MAP, title="Agent Performance in Service")
                    fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
                    fig.update_layout(height=400, showlegend=False, yaxis_range=[0, rating_counts['Count'].max() * 1.25],
                                      xaxis_title_font_size=16, yaxis_title_font_size=16,
                                      xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                      bargap=1.0) # Apply bargap here
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    display_placeholder_outbound("No 'AgentRating' data (excluding Unknown) for Service calls.")
            else:
                display_placeholder_outbound("No 'AgentRating' data (excluding Unknown) for Service calls.")

        else:
            display_placeholder_outbound("Column 'AgentRating' not found or no Service calls.")

    st.markdown("---")

    # --- Raw Data Expander --- (Original from Outbound code)
    with st.expander("Show Filtered Raw Data (Outbound)"):
        st.dataframe(filtered_df)

    st.markdown("---")

# --- Dashboard Function for Inbound Call Analytics (Exact original code) ---
def run_inbound_dashboard(uploaded_file):
    # --- Custom Color Palette --- (Original from Inbound code)
    CUSTOM_COLORS = {
        'Biscay': '#183F5E',    # Darkest Blue
        'Edgewater': '#BADED7', # Lightest Blue-Green
        'Neptune': '#73B4B6',   # Mid Blue-Green
        'Mariner': '#267EBB'    # Medium Blue
    }

    # Mapping for Sentiment labels (Original from Inbound code)
    SENTIMENT_COLOR_MAP = {
        'Positive': CUSTOM_COLORS['Edgewater'], # Lightest for positive
        'Neutral': CUSTOM_COLORS['Neptune'],    # Mid-tone for neutral
        'Negative': CUSTOM_COLORS['Biscay'],    # Darkest for negative
        'Unknown': '#CCCCCC' # A neutral grey for unknown/unspecified
    }

    # Custom sequential colorscale for heatmaps (Original from Inbound code)
    HEATMAP_COLORSCALE = [
        [0.0, CUSTOM_COLORS['Edgewater']],   # Lightest blue-green
        [0.33, CUSTOM_COLORS['Neptune']],    # Mid blue-green
        [0.66, CUSTOM_COLORS['Mariner']],    # Medium blue
        [1.0, CUSTOM_COLORS['Biscay']]       # Darkest blue
    ]

    # Helper functions (Original from Inbound code)
    @st.cache_data
    def load_data_inbound(file): # Renamed to avoid conflict
        """Loads and preprocesses data from an uploaded CSV file."""
        try:
            # Original Inbound code
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip() # Strip whitespace from column names

            nan_fill_defaults = {
                'UserIntentToBuy': 'Not Specified',
                'CallSentiment': 'Unknown',
                'CallObjective': 'Unknown',
                'NextAction': 'Unknown',
                'Customer_Language': 'Unknown',
                'CallType': 'Unknown'
            }

            for col, default_val in nan_fill_defaults.items():
                if col in df.columns:
                    # Original Inbound logic: only replace 'nan' string
                    df[col] = df[col].astype(str).replace('nan', default_val, regex=False)
                    df[col] = df[col].fillna(default_val).str.strip()
                else:
                    df[col] = default_val

            return df
        except Exception as e:
            st.error(f"Error processing uploaded file for Inbound: {e}. Please ensure it's a valid CSV.")
            return pd.DataFrame()

    def find_col_inbound(df_columns, possible_names): # Renamed
        """Finds a column name, trying exact match, then case-insensitive."""
        for name in possible_names:
            if name in df_columns:
                return name
        for name in possible_names:
            for c in df_columns:
                if c.lower() == name.lower():
                    return c
        return None

    def ensure_datetime_inbound(df, col_name_candidates): # Renamed
        """Tries to find and convert a column to datetime, returning series and column name."""
        for c in col_name_candidates:
            if c and c in df.columns:
                try:
                    dt = pd.to_datetime(df[c], errors='coerce')
                    if dt.notna().sum() > 0:
                        return dt, c
                except Exception:
                    pass
        return None, None

    def top_n_with_pct_inbound(series, n=5): # Renamed, no `exclude_labels` parameter here to match original
        """Calculates top N values and their percentages from a Series."""
        vc = series.value_counts(dropna=False)
        total = vc.sum()

        if vc.empty or total == 0:
            return pd.DataFrame(), 0

        top = vc.head(n).reset_index()
        top.columns = ['label', 'count']
        top['pct'] = (top['count'] / total * 100).round(1)
        top['text_label'] = top.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)

        return top, total

    def ordered_intent_counts_inbound(df_filtered, col, order=None): # Renamed
        """Orders user intent counts by a predefined order and prepares data for plotting."""
        if order is None:
            order = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Not Specified']

        if df_filtered.empty or col not in df_filtered.columns:
            return pd.DataFrame()

        counts = df_filtered[col].value_counts(dropna=False)
        counts_reindexed = counts.reindex(order, fill_value=0).astype(int)

        relevant_intents = [
            intent for intent in order
            if counts_reindexed.get(intent, 0) > 0 or (intent == 'Not Specified' and intent in counts_reindexed.index)
        ]

        final_ordered_series = counts_reindexed.loc[relevant_intents].reindex(order).dropna()

        if final_ordered_series.empty or final_ordered_series.sum() == 0:
            return pd.DataFrame()

        out_df = pd.DataFrame({
            'intent': final_ordered_series.index.tolist(),
            'count': final_ordered_series.values
        })

        total_count = out_df['count'].sum()
        out_df['pct'] = (out_df['count'] / total_count * 100).round(1).fillna(0)
        out_df['text_label'] = out_df.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)

        out_df['intent'] = pd.Categorical(out_df['intent'], categories=order, ordered=True)
        out_df = out_df.sort_values('intent')

        return out_df

    def display_placeholder_inbound(message, height_px): # Renamed
        """Displays an info message followed by a spacer to match chart height."""
        st.info(message)
        spacer_height = max(0, height_px - 80) # Adjust 80px if st.info height varies
        st.markdown(f"<div style='height: {spacer_height}px;'></div>", unsafe_allow_html=True)

    def display_business_question_inbound(question_text): # Renamed
        """Displays a business question in a wide, styled horizontal strip."""
        # Note: The CSS for .question-strip is now global for consistency.
        st.markdown(
            f'<div class="question-strip"><h3>{question_text}</h3></div>',
            unsafe_allow_html=True
        )

    @st.cache_data
    def get_hourly_data_for_heatmap_inbound(df_subset, day_type_filter=None): # Renamed
        """
        Prepares hourly call data for heatmaps, optionally filtering by day type.
        Includes percentages and returns only hours with a count > 0.
        """
        if df_subset.empty or '__hour' not in df_subset.columns or df_subset['__hour'].isna().all():
            # If no data or no valid hours, return an empty DataFrame
            return pd.DataFrame()

        current_df = df_subset.copy()
        if day_type_filter and day_type_filter != 'Overall' and 'DayType' in current_df.columns:
            current_df = current_df[current_df['DayType'] == day_type_filter]

        hourly_counts = current_df['__hour'].dropna().astype(int).value_counts().reindex(range(24), fill_value=0)

        df_result = pd.DataFrame({'Hour': hourly_counts.index, 'Count': hourly_counts.values})

        total_calls_in_subset = df_result['Count'].sum()

        if total_calls_in_subset > 0:
            df_result['Percentage'] = (df_result['Count'] / total_calls_in_subset * 100).round(1)
        else:
            df_result['Percentage'] = 0.0

        df_result['Hour_Label'] = df_result['Hour'].apply(lambda h: f"{h:02d}:00")

        # Only return rows where Count is greater than 0
        return df_result[df_result['Count'] > 0]

    # --- Start of Main Dashboard Logic (Original from Inbound code) ---

    if not uploaded_file:
        st.info("Upload your Inbound Call Analysis CSV from the sidebar to view this dashboard.")
        return # Use return instead of st.stop()

    df = load_data_inbound(uploaded_file)

    if df.empty:
        st.error("The uploaded file could not be processed or is empty after processing. Please check the file content and try again.")
        return # Use return instead of st.stop()

    colnames = df.columns.tolist()

    col_calltype = find_col_inbound(colnames, ['CallType', 'Call Type', 'calltype', 'call_type'])
    col_sentiment = find_col_inbound(colnames, ['CallSentiment', 'Call Sentiment', 'Sentiment', 'call_sentiment'])
    col_intent = find_col_inbound(colnames, ['UserIntentToBuy', 'User Intent to Buy', 'Intent', 'UserIntent', 'User_Intent'])
    col_objective = find_col_inbound(colnames, ['CallObjective', 'Call Objective', 'Call_Objective', 'callobjective'])
    col_nextaction = find_col_inbound(colnames, ['NextAction', 'Next Action', 'Next_Action', 'NextActionTaken'])
    col_time = find_col_inbound(colnames, ['Time', 'CallTime', 'CreatedAt', 'CreatedDate', 'DateTime'])
    col_language = find_col_inbound(colnames, ['Customer_Language', 'Customer Language', 'Language', 'customer_language'])
    col_region = find_col_inbound(colnames, ['Region', 'region'])
    col_state = find_col_inbound(colnames, ['State', 'state'])
    col_city = find_col_inbound(colnames, ['City', 'city'])

    dt_series, dt_col_used = ensure_datetime_inbound(df, [col_time] if col_time else [])
    if dt_series is None:
        st.warning("No valid 'Time' or datetime column found. Hourly analysis will not be available.")
        df['__call_dt'] = pd.NaT
        df['__hour'] = np.nan
        df['DayOfWeek'] = 'Unknown' # Placeholder
        df['DayType'] = 'Unknown'   # Placeholder
    else:
        df['__call_dt'] = dt_series
        df['__hour'] = df['__call_dt'].dt.hour
        df['DayOfWeek'] = df['__call_dt'].dt.day_name()
        # Monday=0, Tuesday=1, ..., Sunday=6. Weekends are Sat (5) and Sun (6)
        df['DayType'] = df['__call_dt'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

    if not col_calltype or not df[col_calltype].str.lower().isin(['sales', 'service', 'unknown']).any():
        st.error("Mandatory column 'CallType' not found or does not contain 'Sales'/'Service' values after cleaning. Please check your CSV.")
        return # Use return

    # Filter Section (Original from Inbound code, using unique keys)
    st.sidebar.markdown("### Filter Inbound Data") # Changed from st.sidebar.header
    filtered_df = df.copy()

    all_regions_options = []
    if col_region and not df[col_region].empty:
        all_regions_options = df[col_region].unique().tolist()
    selected_region = st.sidebar.multiselect('Select Region (Inbound)', options=all_regions_options, default=all_regions_options, key='inbound_region_filter')

    if col_region and selected_region:
        filtered_df = filtered_df[filtered_df[col_region].isin(selected_region)]
    else:
        if col_region is None: st.sidebar.info("No 'Region' column found for filtering.")
        elif not selected_region and col_region: st.sidebar.info("No regions selected; showing all data.")

    all_states_options = []
    if col_state:
        if not filtered_df.empty:
            all_states_options = filtered_df[col_state].unique().tolist()
        elif not df.empty:
            all_states_options = df[col_state].unique().tolist()

    selected_state = st.sidebar.multiselect('Select State (Inbound)', options=all_states_options, default=all_states_options, key='inbound_state_filter')

    if col_state and selected_state:
        filtered_df = filtered_df[filtered_df[col_state].isin(selected_state)]
    else:
        if col_state is None: st.sidebar.info("No 'State' column found for filtering.")
        elif not selected_state and col_state: st.sidebar.info("No states selected; showing all data.")

    all_cities_options = []
    if col_city:
        if not filtered_df.empty:
            all_cities_options = filtered_df[col_city].unique().tolist()
        elif not df.empty:
            all_cities_options = df[col_city].unique().tolist()

    selected_city = st.sidebar.multiselect('Select City (Inbound)', options=all_cities_options, default=all_cities_options, key='inbound_city_filter')

    if col_city and selected_city:
        filtered_df = filtered_df[filtered_df[col_city].isin(selected_city)]
    else:
        if col_city is None: st.sidebar.info("No 'City' column found for filtering.")
        elif not selected_city and col_city: st.sidebar.info("No cities selected; showing all data.")

    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your filter selections or upload a different file.")
        return # Use return

    # KPI calculations (Original from Inbound code)
    total_calls = len(filtered_df)

    sales_df = filtered_df[filtered_df[col_calltype].str.lower() == 'sales']
    service_df = filtered_df[filtered_df[col_calltype].str.lower() == 'service']
    num_sales = len(sales_df)
    num_service = len(service_df)
    pct_sales = (num_sales / total_calls * 100) if total_calls > 0 else 0
    pct_service = (num_service / total_calls * 100) if total_calls > 0 else 0

    # Sentiment distribution data
    sentiment_order_for_plot = ['Positive', 'Negative', 'Neutral', 'Unknown']

    # Customer Final Interest (Sales) - distribution for UserIntentToBuy
    intent_df_sales = pd.DataFrame()
    if col_intent and not sales_df.empty:
        intent_df_sales = ordered_intent_counts_inbound(sales_df, col_intent)

    # Customer Final Interest (Service) - distribution for UserIntentToBuy (NEW)
    intent_df_service = pd.DataFrame()
    if col_intent and not service_df.empty:
        intent_df_service = ordered_intent_counts_inbound(service_df, col_intent)

    # Top 5 Call Objective Themes - Sales & Service
    top5_sales_obj = pd.DataFrame()
    top5_service_obj = pd.DataFrame()
    if col_objective:
        if not sales_df.empty:
            top5_sales_obj, _ = top_n_with_pct_inbound(sales_df[col_objective], n=5)
        if not service_df.empty:
            top5_service_obj, _ = top_n_with_pct_inbound(service_df[col_objective], n=5)

    # Top 3 Next Actions - Sales & Service
    top3_sales_next = pd.DataFrame()
    top3_service_next = pd.DataFrame()
    if col_nextaction:
        if not sales_df.empty:
            top3_sales_next, _ = top_n_with_pct_inbound(sales_df[col_nextaction], n=3)
        if not service_df.empty:
            top3_service_next, _ = top_n_with_pct_inbound(service_df[col_nextaction], n=3)

    # Customer Language Volume & % for overall, sales, and service
    lang_df_overall = pd.DataFrame()
    lang_df_sales = pd.DataFrame()
    lang_df_service = pd.DataFrame()

    if col_language and not filtered_df.empty:
        vc_overall = filtered_df[col_language].value_counts(dropna=False)
        if not vc_overall.empty:
            lang_df_overall = pd.DataFrame({'language': vc_overall.index, 'count': vc_overall.values})
            lang_df_overall['pct'] = (lang_df_overall['count'] / lang_df_overall['count'].sum() * 100).round(1)
            lang_df_overall['text_label'] = lang_df_overall.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
            lang_df_overall = lang_df_overall.sort_values('count', ascending=False)

    if col_language and not sales_df.empty:
        vc_sales = sales_df[col_language].value_counts(dropna=False)
        if not vc_sales.empty:
            lang_df_sales = pd.DataFrame({'language': vc_sales.index, 'count': vc_sales.values})
            lang_df_sales['pct'] = (lang_df_sales['count'] / lang_df_sales['count'].sum() * 100).round(1)
            lang_df_sales['text_label'] = lang_df_sales.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
            lang_df_sales = lang_df_sales.sort_values('count', ascending=False)

    if col_language and not service_df.empty:
        vc_service = service_df[col_language].value_counts(dropna=False)
        if not vc_service.empty:
            lang_df_service = pd.DataFrame({'language': vc_service.index, 'count': vc_service.values})
            lang_df_service['pct'] = (lang_df_service['count'] / lang_df_service['count'].sum() * 100).round(1)
            lang_df_service['text_label'] = lang_df_service.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
            lang_df_service = lang_df_service.sort_values('count', ascending=False)

    # Layout / Visuals (Original from Inbound code)
    st.markdown("<h1 style='margin-bottom:6px'>Inbound Call Analytics Dashboard</h1>", unsafe_allow_html=True) # Changed from `st.title`
    st.markdown("All insights from store manager calls across India — sales and service performance at a glance.")

    # --- Overall KPIs (Top Row) ---
    col1, col2 = st.columns([1.2, 2])

    with col1:
        st.metric(label="Total Calls Analysed (Inbound)", value=f"{total_calls:,}", delta=f"Sales {num_sales:,} • Service {num_service:,}")

    # Q1. What type of Calls do we get? - Single Horizontal Stacked Bar Chart
    display_business_question_inbound("Q1. What type of Calls do we get?")
    st.markdown("<h5>Call Type Distribution</h5>", unsafe_allow_html=True)

    if total_calls > 0:
        # Prepare data (Original Inbound includes 'Overall' bar)
        call_type_data = pd.DataFrame({
            'Call Type': ['Sales', 'Service', 'Overall'],
            'Volume': [num_sales, num_service, num_sales + num_service]
        })

        # Calculate percentage and add label
        call_type_data['Percentage'] = (call_type_data['Volume'] / total_calls * 100).round(1)
        call_type_data['Label'] = call_type_data.apply(
            lambda row: f"{row['Volume']} ({row['Percentage']}%)",
            axis=1
        )

        # Custom color mapping
        color_map = {
            'Sales': CUSTOM_COLORS['Biscay'],
            'Service': CUSTOM_COLORS['Neptune'],
            'Overall': CUSTOM_COLORS['Mariner']
        }

        # Plot
        fig_call_type = px.bar(
            call_type_data,
            x='Volume',
            y='Call Type',
            orientation='h',
            text='Label',
            color='Call Type',
            color_discrete_map=color_map
        )

        # Trace formatting
        fig_call_type.update_traces(
            textposition='outside',
            cliponaxis=False,
            width=0.6
        )

        # Layout
        fig_call_type.update_layout(
            bargap=0.4,
            barmode='group',
            height=220,  # Slightly taller for text clarity
            margin=dict(t=40, b=40, l=80, r=40),
            showlegend=False,
            xaxis_title="Number of Calls",
            yaxis_title="",
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=14),
            yaxis_tickfont=dict(size=14),
            xaxis_range=[0, call_type_data['Volume'].max() * 1.15]
        )

        st.plotly_chart(fig_call_type, use_container_width=True)

    else:
        display_placeholder_inbound("No calls to analyze Call Type distribution.", height_px=200)

    st.markdown("---")

    # --- SALES INSIGHTS ---
    st.header("Sales Insights")

    # Q2. What are the Top Reasons Customers Call Us? (Sales)
    display_business_question_inbound("Q2. What are the Top Reasons Customers Call Us? (Sales)")
    st.markdown("<h5>Top 5 Call Objective Themes — Sales Calls</h5>", unsafe_allow_html=True)
    if not top5_sales_obj.empty:
        max_count_sales_obj = top5_sales_obj['count'].max()
        fig = px.bar(top5_sales_obj[::-1], x='count', y='label', orientation='h',
                     labels={'count':'Volume','label':'Call Objective'}, text='text_label',
                     color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=16,
            textangle=0,
            cliponaxis=False
        )
        fig.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Volume',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            yaxis_tickangle=0, # Set Y-axis labels horizontal
            yaxis_automargin=True, # Rely on automargin for proper spacing
            height=350, # Keep generous height
            xaxis_range=[0, max_count_sales_obj * 1.3]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        display_placeholder_inbound("No 'CallObjective' data for Sales calls.", height_px=350)

    # Q3. How Serious are the Callers? (Sales)
    display_business_question_inbound("Q3. How Serious are the Callers? (Sales)")
    st.markdown("<h5>Customer Final Interest (Sales) — Volume & %</h5>", unsafe_allow_html=True)
    if not intent_df_sales.empty: # Changed from intent_df to intent_df_sales
        max_count_intent = intent_df_sales['count'].max()
        fig = px.bar(intent_df_sales, x='intent', y='count', text='text_label', labels={'intent':'Intent Level','count':'Volume'},
                     title="", color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig.update_traces(
            textposition='outside',
            textfont_size=16,
            textangle=0,
            cliponaxis=False
        )
        fig.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(t=40, b=80, l=40, r=40),
            yaxis_title='Volume',
            xaxis_title='',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            height=250,
            yaxis_range=[0, max_count_intent * 1.3]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        display_placeholder_inbound("No 'UserIntentToBuy' data for sales calls, or the column is missing/empty after filtering.", height_px=250)

    # Q4. What is the Customer Sentiment for the Sales Calls?
    display_business_question_inbound("Q4. What is the Customer Sentiment for the Sales Calls?")
    st.markdown("<h5>Sales Sentiment Distribution</h5>", unsafe_allow_html=True)
    if col_sentiment:
        sentiment_order_for_plot = ['Positive', 'Negative', 'Neutral', 'Unknown'] # Define locally
        if not sales_df.empty and sales_df[col_sentiment].notna().any():
            sentiment_dist_sales = sales_df[col_sentiment].value_counts(dropna=False).reindex(sentiment_order_for_plot, fill_value=0)
            sentiment_dist_sales = sentiment_dist_sales[sentiment_dist_sales > 0] # Filter out categories with zero count

            if not sentiment_dist_sales.empty:
                sentiment_df_sales = sentiment_dist_sales.reset_index()
                sentiment_df_sales.columns = ['Sentiment', 'Count']
                sentiment_df_sales['Percentage'] = (sentiment_df_sales['Count'] / sentiment_df_sales['Count'].sum() * 100).round(1)
                sentiment_df_sales['Text'] = sentiment_df_sales.apply(lambda r: f"{r['Count']} ({r['Percentage']:.1f}%)", axis=1)
                max_count_sales_sent = sentiment_df_sales['Count'].max()

                fig_sales_sent = px.bar(
                    sentiment_df_sales,
                    x='Sentiment',
                    y='Count',
                    text='Text',
                    labels={'Sentiment': 'Sentiment', 'Count': 'Number of Calls'},
                    color='Sentiment',
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                    category_orders={"Sentiment": sentiment_order_for_plot}
                )

                fig_sales_sent.update_traces(
                    textposition='outside',
                    textfont_size=16,
                    cliponaxis=False,
                    width=0.6
                )

                fig_sales_sent.update_layout(
                    uniformtext_minsize=12,
                    uniformtext_mode='hide',
                    margin=dict(t=40, b=80, l=40, r=40, autoexpand=True),
                    height=400,
                    xaxis_title_font_size=16,
                    yaxis_title_font_size=16,
                    xaxis_tickfont_size=16,
                    yaxis_tickfont_size=16,
                    yaxis_range=[0, max_count_sales_sent * 1.4],  # more padding for text
                    bargap=1.0,
                    xaxis_type='category'
                )


                st.plotly_chart(fig_sales_sent, use_container_width=True)
            else:
                display_placeholder_inbound("No sales sentiment data available for plotting.", height_px=250)
        else:
            display_placeholder_inbound("No sales calls or sentiment data available.", height_px=250)
    else:
        display_placeholder_inbound("Column 'CallSentiment' not found. Detailed sentiment analysis is not available.", height_px=250)

    # Q5. What are the Languages in which Customers Call? (Sales)
    display_business_question_inbound("Q5. What are the Languages in which Customers Call? (Sales)")
    st.markdown("<h5>Sales Customer Language — Volume & %</h5>", unsafe_allow_html=True)
    if not lang_df_sales.empty:
        max_count_lang_sales = lang_df_sales['count'].max()
        fig_lang_sales = px.bar(lang_df_sales.head(10), x='count', y='language', orientation='h', text='text_label',
                                labels={'count':'Volume','language':'Language'}, title="",
                                color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig_lang_sales.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=16,
            textangle=0, # Keep text horizontal on the bar
            cliponaxis=False
        )
        fig_lang_sales.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(l=150, t=40, b=80, r=40), # Adjusted left margin
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Volume',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            yaxis_tickangle=0, # Set Y-axis labels horizontal
            yaxis_automargin=True, # Enable auto margin to ensure labels fit
            height=350, # Keep generous height
            xaxis_range=[0, max_count_lang_sales * 1.3]
        )
        st.plotly_chart(fig_lang_sales, use_container_width=True)
    else:
        display_placeholder_inbound("No 'Customer_Language' data for sales calls found or is empty after filtering.", height_px=350)

    # Q6. What are the Next Action post Calls? (Sales)
    display_business_question_inbound("Q6. What are the Next Action post Calls? (Sales)")
    st.markdown("<h5>Top 3 Next Actions — Sales Calls</h5>", unsafe_allow_html=True)
    if not top3_sales_next.empty:
        max_count_sales_next = top3_sales_next['count'].max()
        fig3 = px.bar(top3_sales_next[::-1], x='count', y='label', orientation='h', text='text_label',
                      labels={'label':'Next Action','count':'Volume'},
                      color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig3.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=16,
            textangle=0,
            cliponaxis=False
        )
        fig3.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Volume',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            yaxis_tickangle=0, # Set Y-axis labels horizontal
            yaxis_automargin=True, # Rely on automargin for proper spacing
            height=350, # Keep generous height
            xaxis_range=[0, max_count_sales_next * 1.3]
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        display_placeholder_inbound("No 'NextAction' data for Sales calls.", height_px=350)

    # Q7. What are the Peak Call Hours? (Sales)
    display_business_question_inbound("Q7. What are the Peak Call Hours? (Sales)")
    st.markdown("<h5>Sales Calls Hourly Volume Heatmap</h5>", unsafe_allow_html=True)
    sales_day_type = st.selectbox(
        "Select Day Type for Sales Calls:",
        options=['Overall', 'Weekday', 'Weekend'],
        index=0,
        key='inbound_sales_day_type_selector'
    )
    current_sales_df_for_heatmap = sales_df
    if sales_day_type != 'Overall':
        if 'DayType' not in sales_df.columns or sales_df['DayType'].isna().all():
            st.warning(f"Cannot filter sales calls by '{sales_day_type}': 'DayType' column not available or empty.")
            sales_day_type = 'Overall'
        else:
            current_sales_df_for_heatmap = sales_df[sales_df['DayType'] == sales_day_type]

    sales_hourly_data = get_hourly_data_for_heatmap_inbound(current_sales_df_for_heatmap, sales_day_type)

    if not sales_hourly_data.empty and sales_hourly_data['Count'].sum() > 0:
        heatmap_fig_sales = go.Figure(data=go.Heatmap(
            z=sales_hourly_data['Count'].values.reshape(1, -1),
            x=sales_hourly_data['Hour_Label'].tolist(),
            y=['Sales Calls'],
            colorscale=HEATMAP_COLORSCALE,
            hovertemplate="Hour: %{x}<br>Count: %{z}<br>Percentage: %{customdata:.1f}%<extra></extra>",
            customdata=[sales_hourly_data['Percentage'].values.tolist()]
        ))
        heatmap_fig_sales.update_layout(
            height=150,
            margin=dict(t=40,b=80,l=40,r=40),
            xaxis_nticks=len(sales_hourly_data['Hour_Label']), # Display ticks only for hours with data
            xaxis_title="Hour of Day",
            yaxis_title="",
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=16,
        )
        st.plotly_chart(heatmap_fig_sales, use_container_width=True)
        st.markdown("<h6>Sales Calls Hourly Data:</h6>", unsafe_allow_html=True)
        st.dataframe(sales_hourly_data[['Hour_Label', 'Count', 'Percentage']].rename(columns={'Hour_Label': 'Hour', 'Count': 'Calls', 'Percentage': 'Percentage (%)'}))
    else:
        display_placeholder_inbound(f"No sales calls data to display hourly heatmap on {sales_day_type}.", height_px=150)

    st.markdown("---")

    # --- SERVICE INSIGHTS ---
    st.header("Service Insights")

    # Q2. What are the Top Reasons Customers Call Us? (Service)
    display_business_question_inbound("Q2. What are the Top Reasons Customers Call Us? (Service)")
    st.markdown("<h5>Top 5 Call Objective Themes — Service Calls</h5>", unsafe_allow_html=True)
    if not top5_service_obj.empty:
        max_count_service_obj = top5_service_obj['count'].max()
        fig2 = px.bar(top5_service_obj[::-1], x='count', y='label', orientation='h', text='text_label',
                      labels={'count':'Volume','label':'Call Objective'},
                      color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig2.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=16,
            textangle=0,
            cliponaxis=False
        )
        fig2.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Volume',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            yaxis_tickangle=0, # Set Y-axis labels horizontal
            yaxis_automargin=True, # Rely on automargin for proper spacing
            height=350, # Keep generous height
            xaxis_range=[0, max_count_service_obj * 1.3]
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        display_placeholder_inbound("No 'CallObjective' data for Service calls.", height_px=350)

    # Q3. How Serious are the Callers? (Service) - NEW CHART FOR SERVICE
    display_business_question_inbound("Q3. How Serious are the Callers? (Service)")
    st.markdown("<h5>Customer Final Interest (Service) — Volume & %</h5>", unsafe_allow_html=True)
    if not intent_df_service.empty:
        max_count_intent_service = intent_df_service['count'].max()
        fig_service_intent = px.bar(intent_df_service, x='intent', y='count', text='text_label', labels={'intent':'Intent Level','count':'Volume'},
                                     title="", color_discrete_sequence=[CUSTOM_COLORS['Neptune']]) # Use a service-specific color
        fig_service_intent.update_traces(
            textposition='outside',
            textfont_size=16,
            textangle=0,
            cliponaxis=False
        )
        fig_service_intent.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(t=40, b=80, l=40, r=40),
            yaxis_title='Volume',
            xaxis_title='',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            height=250,
            yaxis_range=[0, max_count_intent_service * 1.3]
        )
        st.plotly_chart(fig_service_intent, use_container_width=True)
    else:
        display_placeholder_inbound("No 'UserIntentToBuy' data for service calls, or the column is missing/empty after filtering.", height_px=250)


    # Q4. What is the Customer Sentiment for the Service Calls?
    display_business_question_inbound("Q4. What is the Customer Sentiment for the Service Calls?")
    st.markdown("<h5>Service Sentiment Distribution</h5>", unsafe_allow_html=True)
    if col_sentiment:
        sentiment_order_for_plot = ['Positive', 'Negative', 'Neutral', 'Unknown'] # Define locally
        if not service_df.empty and service_df[col_sentiment].notna().any():
            sentiment_dist_service = service_df[col_sentiment].value_counts(dropna=False).reindex(sentiment_order_for_plot, fill_value=0)
            sentiment_dist_service = sentiment_dist_service[sentiment_dist_service > 0]

            if not sentiment_dist_service.empty:
                sentiment_df_service = sentiment_dist_service.reset_index()
                sentiment_df_service.columns = ['Sentiment', 'Count']
                sentiment_df_service['Percentage'] = (sentiment_df_service['Count'] / sentiment_df_service['Count'].sum() * 100).round(1)
                sentiment_df_service['Text'] = sentiment_df_service.apply(lambda r: f"{r['Count']} ({r['Percentage']:.1f}%)", axis=1)
                max_count_service_sent = sentiment_df_service['Count'].max()

                fig_service_sent = px.bar(
                    sentiment_df_service,
                    x='Sentiment',
                    y='Count',
                    text='Text',
                    labels={'Sentiment': 'Sentiment', 'Count': 'Number of Calls'},
                    color='Sentiment',
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                    category_orders={"Sentiment": sentiment_order_for_plot}
                )

                fig_service_sent.update_traces(
                    textposition='outside',
                    textfont_size=16,
                    cliponaxis=False,
                    width=0.6
                )

                fig_service_sent.update_layout(
                    uniformtext_minsize=12,
                    uniformtext_mode='hide',
                    margin=dict(t=40, b=80, l=40, r=40),
                    height=500,
                    #width=100,
                    xaxis_title_font_size=16,
                    yaxis_title_font_size=16,
                    xaxis_tickfont_size=16,
                    yaxis_tickfont_size=16,
                    yaxis_range=[0, max_count_service_sent * 1.4],  # more padding for text (original range was [0,100])
                    bargap=1.0,
                    xaxis_type='category'
                )

                st.plotly_chart(fig_service_sent, use_container_width=True)
            else:
                display_placeholder_inbound("No service sentiment data available for plotting.", height_px=250)
        else:
            display_placeholder_inbound("No service calls or sentiment data available.", height_px=250)
    else:
        display_placeholder_inbound("Column 'CallSentiment' not found. Detailed sentiment analysis is not available.", height_px=250)

    # Q5. What are the Languages in which Customers Call? (Service)
    display_business_question_inbound("Q5. What are the Languages in which Customers Call? (Service)")
    st.markdown("<h5>Service Customer Language — Volume & %</h5>", unsafe_allow_html=True)
    if not lang_df_service.empty:
        max_count_lang_service = lang_df_service['count'].max()
        fig_lang_service = px.bar(lang_df_service.head(10), x='count', y='language', orientation='h', text='text_label',
                                   labels={'count':'Volume','language':'Language'}, title="",
                                   color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig_lang_service.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=16,
            textangle=0, # Keep text horizontal on the bar
            cliponaxis=False
        )
        fig_lang_service.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(l=150, t=40, b=80, r=40), # Adjusted left margin
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Volume',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            yaxis_tickangle=0, # Set Y-axis labels horizontal
            yaxis_automargin=True, # Enable auto margin to ensure labels fit
            height=350, # Keep generous height
            xaxis_range=[0, max_count_lang_service * 1.3]
        )
        st.plotly_chart(fig_lang_service, use_container_width=True)
    else:
        display_placeholder_inbound("No 'Customer_Language' data for service calls found or is empty after filtering.", height_px=350)

    # Q6. What are the Next Action post Calls? (Service)
    display_business_question_inbound("Q6. What are the Next Action post Calls? (Service)")
    st.markdown("<h5>Top 3 Next Actions — Service Calls</h5>", unsafe_allow_html=True)
    if not top3_service_next.empty:
        max_count_service_next = top3_service_next['count'].max()
        fig4 = px.bar(top3_service_next[::-1], x='count', y='label', orientation='h', text='text_label',
                      labels={'label':'Next Action','count':'Volume'},
                      color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig4.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=16,
            textangle=0,
            cliponaxis=False
        )
        fig4.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Volume',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            yaxis_tickangle=0, # Set Y-axis labels horizontal
            yaxis_automargin=True, # Rely on automargin for proper spacing
            height=350, # Keep generous height
            xaxis_range=[0, max_count_service_next * 1.3]
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        display_placeholder_inbound("No 'NextAction' data for Service calls.", height_px=350)

    # Q7. What are the Peak Call Hours? (Service)
    display_business_question_inbound("Q7. What are the Peak Call Hours? (Service)")
    st.markdown("<h5>Service Calls Hourly Volume Heatmap</h5>", unsafe_allow_html=True)
    service_day_type = st.selectbox(
        "Select Day Type for Service Calls:",
        options=['Overall', 'Weekday', 'Weekend'],
        index=0,
        key='inbound_service_day_type_selector'
    )
    current_service_df_for_heatmap = service_df
    if service_day_type != 'Overall':
        if 'DayType' not in service_df.columns or service_df['DayType'].isna().all():
            st.warning(f"Cannot filter service calls by '{service_day_type}': 'DayType' column not available or empty.")
            service_day_type = 'Overall'
        else:
            current_service_df_for_heatmap = service_df[service_df['DayType'] == service_day_type]

    service_hourly_data = get_hourly_data_for_heatmap_inbound(current_service_df_for_heatmap, service_day_type)

    if not service_hourly_data.empty and service_hourly_data['Count'].sum() > 0:
        heatmap_fig_service = go.Figure(data=go.Heatmap(
            z=service_hourly_data['Count'].values.reshape(1, -1),
            x=service_hourly_data['Hour_Label'].tolist(),
            y=['Service Calls'],
            colorscale=HEATMAP_COLORSCALE,
            hovertemplate="Hour: %{x}<br>Count: %{z}<br>Percentage: %{customdata:.1f}%<extra></extra>", # Corrected hovertemplate
            customdata=[service_hourly_data['Percentage'].values.tolist()]
        ))
        heatmap_fig_service.update_layout(
            height=150,
            margin=dict(t=40,b=80,l=40,r=40),
            xaxis_nticks=len(service_hourly_data['Hour_Label']), # Display ticks only for hours with data
            xaxis_title="Hour of Day",
            yaxis_title="",
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=16,
        )
        st.plotly_chart(heatmap_fig_service, use_container_width=True)
        st.markdown("<h6>Service Calls Hourly Data:</h6>", unsafe_allow_html=True)
        st.dataframe(service_hourly_data[['Hour_Label', 'Count', 'Percentage']].rename(columns={'Hour_Label': 'Hour', 'Count': 'Calls', 'Percentage': 'Percentage (%)'}))
    else:
        display_placeholder_inbound(f"No service calls data to display hourly heatmap on {service_day_type}.", height_px=150)

    st.markdown("---")

    # --- GENERAL INSIGHTS ---
    st.header("General Insights")

    # Q5. What are the Languages in which Customers Call? (Overall)
    display_business_question_inbound("Q5. What are the Languages in which Customers Call? (Overall)")
    st.markdown("<h5>Overall Customer Language — Volume & %</h5>", unsafe_allow_html=True)
    if not lang_df_overall.empty:
        max_count_lang_overall = lang_df_overall['count'].max()
        fig_lang_overall = px.bar(lang_df_overall.head(10), x='count', y='language', orientation='h', text='text_label',
                                   labels={'count':'Volume','language':'Language'}, title="",
                                   color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig_lang_overall.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=16,
            textangle=0, # Keep text horizontal on the bar
            cliponaxis=False
        )
        fig_lang_overall.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(l=150, t=40, b=80, r=40), # Adjusted left margin
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Volume',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16,
            yaxis_tickangle=0, # Set Y-axis labels horizontal
            yaxis_automargin=True, # Enable auto margin to ensure labels fit
            height=350, # Keep generous height
            xaxis_range=[0, max_count_lang_overall * 1.3]
        )
        st.plotly_chart(fig_lang_overall, use_container_width=True)
    else:
        display_placeholder_inbound("No 'Customer_Language' data found or is empty after filtering.", height_px=350)

    # Q7. What are the Peak Call Hours? (Overall)
    display_business_question_inbound("Q7. What are the Peak Call Hours? (Overall)")
    st.markdown("<h5>Overall Calls Hourly Volume Heatmap</h5>", unsafe_allow_html=True)
    overall_day_type = st.selectbox(
        "Select Day Type for Overall Calls:",
        options=['Overall', 'Weekday', 'Weekend'],
        index=0,
        key='inbound_overall_day_type_selector'
    )
    current_total_df_for_heatmap = filtered_df
    if overall_day_type != 'Overall':
        if 'DayType' not in filtered_df.columns or filtered_df['DayType'].isna().all():
            st.warning(f"Cannot filter overall calls by '{overall_day_type}': 'DayType' column not available or empty.")
            overall_day_type = 'Overall'
        else:
            current_total_df_for_heatmap = filtered_df[filtered_df['DayType'] == overall_day_type]

    overall_hourly_data = get_hourly_data_for_heatmap_inbound(current_total_df_for_heatmap, overall_day_type)

    if not overall_hourly_data.empty and overall_hourly_data['Count'].sum() > 0:
        heatmap_fig_total = go.Figure(data=go.Heatmap(
            z=overall_hourly_data['Count'].values.reshape(1, -1),
            x=overall_hourly_data['Hour_Label'].tolist(),
            y=['Total Calls'],
            colorscale=HEATMAP_COLORSCALE,
            hovertemplate="Hour: %{x}<br>Count: %{z}<br>Percentage: %{customdata:.1f}%<extra></extra>", # Corrected hovertemplate
            customdata=[overall_hourly_data['Percentage'].values.tolist()]
        ))
        heatmap_fig_total.update_layout(
            height=150,
            margin=dict(t=40,b=80,l=40,r=40),
            xaxis_nticks=len(overall_hourly_data['Hour_Label']), # Display ticks only for hours with data
            xaxis_title="Hour of Day",
            yaxis_title="",
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=16,
        )
        st.plotly_chart(heatmap_fig_total, use_container_width=True)
        st.markdown("<h6>Overall Calls Hourly Data:</h6>", unsafe_allow_html=True)
        st.dataframe(overall_hourly_data[['Hour_Label', 'Count', 'Percentage']].rename(columns={'Hour_Label': 'Hour', 'Count': 'Calls', 'Percentage': 'Percentage (%)'}))
    else:
        display_placeholder_inbound(f"No valid 'Time' or hour information found for overall calls on {overall_day_type}.", height_px=150)


    # Option to show raw data table
    with st.expander("Show raw data (first 200 rows - Inbound)"):
        st.dataframe(df.head(200))

    # Footer / notes
    st.markdown("---")
    st.markdown("**Notes:**")
    st.markdown("""
    - The dashboard adapts to your uploaded CSV. Column names are detected case-insensitively and whitespace is stripped.
    - Missing values in key categorical columns are filled with 'Not Specified' or 'Unknown'.
    - Intent order used: Very Low → Low → Medium → High → Very High → Not Specified.
    - Sentiment detection expects `CallSentiment` to contain 'Positive', 'Negative', 'Neutral', or 'Unknown'. Missing sentiment is treated as 'Unknown'.
    - Hours for the heatmap are extracted from the detected datetime column (e.g., `Time`, `CreatedDate`).
    """)


# --- Main Application Execution Flow ---

# Global title for the entire application
st.title("Mattress Company Call Analytics")
st.markdown("Welcome! Upload your call data files in the sidebar to explore the Outbound and Inbound call analytics dashboards.")

st.sidebar.title("Upload Call Data Files")
st.sidebar.markdown("Upload the corresponding CSV files for each dashboard to enable its view.")

# File uploaders for each dashboard - must have unique keys
uploaded_file_outbound = st.sidebar.file_uploader("Upload Outbound Call Data CSV", type=["csv"], key="outbound_uploader")
uploaded_file_inbound = st.sidebar.file_uploader("Upload Inbound Call Data CSV", type=["csv"], key="inbound_uploader")

st.sidebar.markdown("---") # Separator for clarity
st.sidebar.markdown("### Outbound Expected Columns:")
st.sidebar.markdown("""
- **CallID**
- **Call Type** (Sale / Service)
- **UserIntentToBuy** (Very Low, Low, Medium, High, Very High)
- **CallSentiment** (Positive, Neutral, Negative)
- **Major Purchase Barrier Theme**
- **Top3Themes** (comma-separated, e.g., "Price, Discount, Delivery")
- **AgentNextAction** (Yes / No)
- **AgentRating** (High, Medium, Low)
- **City** (for geographical filtering)
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### Inbound Expected Columns:")
st.sidebar.markdown("""
- **CallType** (e.g., Sales / Service) - Mandatory
- **CallSentiment** (e.g., Positive/Negative/Neutral)
- **UserIntentToBuy** (e.g., Very Low / Low / Medium / High / Very High)
- **CallObjective** (text / theme)
- **NextAction** (text)
- **Time** or other datetime column (timestamp of call)
- **Customer_Language**
- **Region, State, City** (for geographical filters)
""")


# Create tabs for the two dashboards
tab_outbound, tab_inbound = st.tabs(["Outbound Call Analytics", "Inbound Call Analytics"])

with tab_outbound:
    run_outbound_dashboard(uploaded_file_outbound)

with tab_inbound:
    run_inbound_dashboard(uploaded_file_inbound)