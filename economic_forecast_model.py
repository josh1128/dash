import sys
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# === Confirm Running File ===
print("Running file:", os.path.abspath(__file__))

# === Define File Path ===
file_path = r'C:\Users\AC03537\OneDrive - Alberta Central\Desktop\test.xlsx'
sheet_main = 'Potential Output'
sheet_hours = 'Hours'
sheet_is_curve = 'IS Curve'
sheet_phillips = 'Phillips Curve'
sheet_taylor = 'Taylor Rule'

# === File and Sheet Checks ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found: {file_path}")

xls = pd.ExcelFile(file_path)
required_sheets = [sheet_main, sheet_hours, sheet_is_curve, sheet_phillips, sheet_taylor]
for sheet in required_sheets:
    if sheet not in xls.sheet_names:
        raise ValueError(f"Sheet '{sheet}' not found. Available sheets: {xls.sheet_names}")

# === Load Potential Output Data ===
data = pd.read_excel(xls, sheet_name=sheet_main, na_values=["NA"])
cols = ['Date', 'Population', 'Labour Force Participation', 'Labour Productivity', 'NAIRU',
        'Output_Gap_multivariate', 'Output_Gap_Integrated', 'Output_Gap_Internal']
data[cols[1:]] = data[cols[1:]].apply(pd.to_numeric, errors='coerce')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
data['Year'] = data['Date'].dt.year

# === Load Hours Worked Data ===
hours_df = pd.read_excel(xls, sheet_name=sheet_hours, na_values=["NA"])
hours_df['Date'] = pd.to_datetime(hours_df['Date'], format='%Y-%m', errors='coerce')
hours_df = hours_df.dropna(subset=['Date', 'Hours Worked'])

# === Merge Hours Worked ===
data = pd.merge_asof(data.sort_values('Date'), hours_df.sort_values('Date'), on='Date', direction='backward')

# === Calculate Potential Output ===
data['LFP_decimal'] = data['Labour Force Participation'] / 100
data['NAIRU_decimal'] = data['NAIRU'] / 100
data['Productivity_index'] = data['Labour Productivity'] / 100
data['Hours_total'] = data['Hours Worked'] * 1000
data['Potential Output'] = (
    data['LFP_decimal'] * (1 - data['NAIRU_decimal']) * data['Hours_total'] * data['Productivity_index']
)
data['Potential Output'] = data['Potential Output'].clip(lower=0)

# === Load IS Curve Data ===
is_curve_df = pd.read_excel(xls, sheet_name=sheet_is_curve, na_values=["NA"])
is_curve_df.columns = is_curve_df.columns.str.strip()
is_curve_df['Date'] = pd.to_datetime(is_curve_df['Date'], errors='coerce')
is_curve_df = is_curve_df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
is_curve_df['Year'] = is_curve_df['Date'].dt.year
is_curve_cols = [
    'Real Interest Rate', 'Nominal interest rate', 'Real Effective Exchange Rate',
    'Commodity Price Index', 'Foreign Demand', 'Output Gap', 'Inflation Rate'
]
is_curve_df[is_curve_cols] = is_curve_df[is_curve_cols].apply(pd.to_numeric, errors='coerce')

# === Load Phillips Curve Data ===
phillips_df = pd.read_excel(xls, sheet_name=sheet_phillips, na_values=["NA"])
phillips_df.columns = phillips_df.columns.str.strip()
phillips_df['Date'] = pd.to_datetime(phillips_df['Date'], errors='coerce')
phillips_df = phillips_df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
phillips_df['Year'] = phillips_df['Date'].dt.year
phillips_cols = [
    'Inflation rate', 'Real effective exchange rate', 'Brent Crude',
    'WTI Crude', 'Commodity price index'
]
phillips_df[phillips_cols] = phillips_df[phillips_cols].apply(pd.to_numeric, errors='coerce')

# === Load Taylor Rule Data ===
taylor_df = pd.read_excel(xls, sheet_name=sheet_taylor, na_values=["NA"])
taylor_df.columns = taylor_df.columns.str.strip()
taylor_df['Date'] = pd.to_datetime(taylor_df['Date'], errors='coerce')
taylor_df = taylor_df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
taylor_df['Year'] = taylor_df['Date'].dt.year
taylor_cols = [
    'Nominal Interest rate', 'Real Interest Rate', 'Inflation rate',
    'Target inflation rate', 'Real GDP', 'Potential GDP'
]
taylor_df[taylor_cols] = taylor_df[taylor_cols].apply(pd.to_numeric, errors='coerce')

# === HP Filter Function ===
def apply_hp_filter(df, column, prefix=None, log_transform=False, exp_transform=False):
    if column not in df.columns:
        print(f"Column {column} not found.")
        return df
    prefix = prefix or column
    series = df[column].replace(0, np.nan).dropna()
    if len(series) < 10:
        print(f"Skipping HP filter for {column} due to insufficient data.")
        return df
    clean_series = np.log(series) if log_transform else series
    cycle, trend = hpfilter(clean_series, lamb=1600)
    if exp_transform:
        trend = np.exp(trend)
    full_trend = pd.Series(index=df.index, dtype='float64')
    full_cycle = pd.Series(index=df.index, dtype='float64')
    full_trend.loc[series.index] = trend
    full_cycle.loc[series.index] = cycle
    df[f"{prefix}_Trend"] = full_trend
    df[f"{prefix}_Cycle"] = full_cycle
    return df

# === Apply HP Filters ===
for col in ['Labour Force Participation', 'Labour Productivity', 'Potential Output', 'Hours Worked', 'Unemployment']:
    apply_hp_filter(data, col, log_transform=(col == 'Potential Output'), exp_transform=(col == 'Potential Output'))
for col in is_curve_cols:
    apply_hp_filter(is_curve_df, col)
for col in phillips_cols:
    apply_hp_filter(phillips_df, col)
for col in taylor_cols:
    apply_hp_filter(taylor_df, col)

# === Year Range for Slider ===
year_min = min(data['Year'].min(), is_curve_df['Year'].min(), phillips_df['Year'].min(), taylor_df['Year'].min())
year_max = max(data['Year'].max(), is_curve_df['Year'].max(), phillips_df['Year'].max(), taylor_df['Year'].max())

# === Dash App Layout ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Economic Forecast Dashboard"

app.layout = dbc.Container([
    html.H2("Economic Forecast Dashboard", className="text-center my-4 fw-bold"),

    dbc.Row([
        dbc.Col([
            html.Label("Select General Variable", className="fw-semibold"),
            dcc.Dropdown(
                id='variable-dropdown',
                options=[{'label': col, 'value': col} for col in sorted(
                    [col for col in data.columns if col not in ['Date', 'Year']]
                )],
                value='Potential Output',
                clearable=True,
                className="mb-3"
            ),

            html.Label("Select IS Curve Variable", className="fw-semibold"),
            dcc.Dropdown(
                id='is-curve-dropdown',
                options=[{'label': col, 'value': col} for col in is_curve_cols],
                value=None,
                clearable=True,
                placeholder="Optional: Select IS Curve Variable",
                className="mb-3"
            ),

            html.Label("Select Phillips Curve Variable", className="fw-semibold"),
            dcc.Dropdown(
                id='phillips-dropdown',
                options=[{'label': col, 'value': col} for col in phillips_cols],
                value=None,
                clearable=True,
                placeholder="Optional: Select Phillips Curve Variable",
                className="mb-3"
            ),

            html.Label("Select Taylor Rule Variable", className="fw-semibold"),
            dcc.Dropdown(
                id='taylor-dropdown',
                options=[{'label': col, 'value': col} for col in taylor_cols],
                value=None,
                clearable=True,
                placeholder="Optional: Select Taylor Rule Variable",
                className="mb-3"
            ),

            html.Label("Select Data Type", className="fw-semibold"),
            dcc.RadioItems(
                id='data-type-selector',
                options=[
                    {'label': 'Raw', 'value': 'raw'},
                    {'label': 'Trend', 'value': 'trend'},
                    {'label': 'Cycle', 'value': 'cycle'},
                    {'label': 'Raw + Trend', 'value': 'raw_trend'}
                ],
                value='raw',
                inline=True,
                className="mb-4"
            ),

            html.Label("Select Year Range", className="fw-semibold"),
            dcc.RangeSlider(
                id='year-range-slider',
                min=year_min,
                max=year_max,
                value=[year_min, year_max],
                marks={str(year): str(year) for year in range(year_min, year_max + 1, 5)},
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ], md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='forecast-graph')
                ])
            ], className="shadow-sm border-0")
        ], md=8)
    ], align="start")
], fluid=True)

# === Callback ===
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('variable-dropdown', 'value'),
    Input('is-curve-dropdown', 'value'),
    Input('phillips-dropdown', 'value'),
    Input('taylor-dropdown', 'value'),
    Input('data-type-selector', 'value'),
    Input('year-range-slider', 'value')
)
def update_graph(selected_general, selected_is_curve, selected_phillips, selected_taylor, selected_type, selected_year_range):
    start_year, end_year = selected_year_range

    if selected_taylor:
        selected_var = selected_taylor
        df = taylor_df
    elif selected_phillips:
        selected_var = selected_phillips
        df = phillips_df
    elif selected_is_curve:
        selected_var = selected_is_curve
        df = is_curve_df
    elif selected_general:
        selected_var = selected_general
        df = data
    else:
        return go.Figure()

    filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

    fig = go.Figure()
    if selected_type == 'trend':
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(f"{selected_var}_Trend"),
            mode='lines',
            name=f"{selected_var} - Trend",
            line=dict(color='green', dash='dot')
        ))
    elif selected_type == 'cycle':
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(f"{selected_var}_Cycle"),
            mode='lines',
            name=f"{selected_var} - Cycle",
            line=dict(color='purple', dash='dot')
        ))
    elif selected_type == 'raw_trend':
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(selected_var),
            mode='lines',
            name=f"{selected_var} - Raw",
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(f"{selected_var}_Trend"),
            mode='lines',
            name=f"{selected_var} - Trend",
            line=dict(color='green', dash='dot')
        ))
    else:
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df.get(selected_var),
            mode='lines',
            name=f"{selected_var} - Raw",
            line=dict(color='blue')
        ))

    fig.update_layout(
        title={'text': f"{selected_var} ({start_year} to {end_year})", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title=selected_var,
        legend_title="Data Type",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    return fig

# === Run the App ===
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
