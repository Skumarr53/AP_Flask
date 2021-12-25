import os
import re
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

from app_config import app
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output, State
from pdb import set_trace


input_path = './static'

Flag_cols = ['BU_Is_Multi_VendorAddress',
             'BU_Is_Multi_AccCat',
             'BU_Is_Address_MultiVendor']


flag_lab2name = {'BU_Is_Multi_VendorAddress': 'Transaction Volume where vendor is connected to Multiple Addresses under same BU in a month',
                 'BU_Is_Multi_AccCat': 'Transaction Volume where Vendor is connected to Multiple Accounts under same BU in a month',
                 'BU_Is_Address_MultiVendor': 'Transaction Volume where Address is connected to Multiple vendors under same BU in a month'}

colorlist = ['#636EFA',
             '#EF553B',
             '#00CC96']

color_dict = {x: y for x, y in zip(flag_lab2name.values(), colorlist)}

n_top = 3


def load_data(input_path, n_months=12):
    '''
     loads last n_months data and returns combined dataframe
     for both Fraud and GraphSIF

    input:
        n_months: number of previous months results to be load
    '''
    start_date = datetime.datetime.today()
    fraud_df_list = []
    GraphSIF_df_list = []

    for i in range(n_months):
        start_date += relativedelta(months=-1)
        mon = f'{str(start_date.month).zfill(2)}{start_date.year}'

        if os.path.exists(f'{input_path}/Fraud1_Flags_{mon}.csv'):
            fraud_df_list.append(pd.read_csv(
                f'{input_path}/Fraud1_Flags_{mon}.csv'))
        if os.path.exists(f'{input_path}/GraphSIFresults_{mon}.csv'):
            GraphSIF_df_list.append(pd.read_csv(
                f'{input_path}/GraphSIFresults_{mon}.csv'))

    fraud_df = pd.concat(fraud_df_list)
    graphSIF_df = pd.concat(GraphSIF_df_list)
    return fraud_df, graphSIF_df

# Functions


def get_GraphSIF_stats(BU, month):
    if BU != '':
        graphSIF_sub = graphSIF_df[(graphSIF_df.Month == month) &
                                   (graphSIF_df.GraphSIF_BUs == BU)].squeeze()

        # get top feature stats
        top_feat_values = {graphSIF_sub[f'top_feature_{i}']: [
            graphSIF_sub[f'Hist_average_{i}'], graphSIF_sub[f'Test_hstgm_{i}']] for i in range(1, n_top+1)}

        stats_list = [
            f'Changes observed in the top 3 features for test month (histogram) compared to historical 24 month average for BU {BU} in {month} month']
        for key, value in top_feat_values.items():
            feat, avg, test_hist = key, round(value[0], 2), round(value[1], 2)
            feat_split = feat.split('_')
            n_acc, n_add, n_ven, feat_key = feat_split[0], feat_split[1], feat_split[2], '_'.join(
                feat_split[3:])
            prct_chng = (test_hist-avg)/avg*100
            stats_list.append(html.Br())
            stats_list.append(html.Br())
            stats_list.append(
                f" - {feat} :  {str(abs(round(prct_chng,2)))}% {'decrease' if prct_chng < 0 else 'increase'} with respect to past 24 months average ({avg}) in the {feat_labels[feat_key]} with the pattern where {n_ven} vendor connected has {n_add} addresses and is transacting under {n_acc} accounts. current month {feat_labels[feat_key]} is {test_hist}")
        return stats_list
    return ''


def monthly_flags_chart(BU, month):
    flags_df_sub = flags_df.loc[(flags_df.GraphSIF_BUs == BU) & (
        flags_df.Month == month), Flag_cols]

    flags_count = flags_df_sub[Flag_cols].sum()

    flags_count_df = flags_count.to_frame().reset_index()
    flags_count_df.columns = ['flag_label', 'count']

    flags_count_df['label_name'] = flags_count_df.flag_label.replace(
        flag_lab2name)

    fig = px.bar(flags_count_df, y='count', x='label_name',
                 labels={'label_name': 'BU flags',
                         'count': 'Transaction count'},
                 color='label_name',
                 color_discrete_map=color_dict)

    fig.update_layout(transition_duration=500,
                      legend=dict(yanchor='bottom',
                                  y=1.0,
                                  x=-0.1),
                      title=f"Monthly flags for BU {BU} in {month} month",
                      autosize=True,
                      xaxis=dict(visible=False, showgrid=False,
                                 zeroline=False, showticklabels=False),
                      margin=dict(t=20, l=0, r=0, b=0),
                      width=600)
    return fig


flags_df, graphSIF_df = load_data(input_path)

# Months to be displayed in the dropdown menu
Mnths = graphSIF_df['Month'].unique()
Mnths.sort()
Month_options = [{'label': i, 'value': i} for i in Mnths[::-1]]

# BUs to be displayed in the dropdown menu
BUs = graphSIF_df['GraphSIF_BUs'].unique()
BUs.sort()
BU_options = [{'label': i, 'value': i} for i in BUs]

# Anoumolus select options to be displayed in the dropdown menu
Anom_options = [{'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}]

# feature lables
feat_labels = {'Pattern_Count': "transaction volume",
               'Total_Amt': "total amount"}

low_trans_BUs = ['Bu_Monthly_0_10', 'Bu_Monthly_10_20',
                 'Bu_Monthly_20_30', 'Low_window_Bu']

layout = html.Div([
    dbc.Container([
        dbc.Row([html.H3(children='GraphSIF Analysis')]),
        dbc.Row([
            dbc.Col([
                html.Div([html.Label('Select Month'),
                          dcc.Dropdown(id='select-month',
                                       options=Month_options,
                                       value=Mnths[-1],
                                       clearable=False,
                                       style={'width': '300px'}),
                          dcc.Graph(id='monthly-bu-flags')])
            ], id='top-vendors', width=4),
            dbc.Col([
                html.Div([
                    dbc.Row([
                        html.Div([
                            html.Label('Select BU'),
                            dcc.Dropdown(
                                id='select-BU',
                                options=BU_options,
                                value=BUs[0],
                                clearable=False)],
                            style={'paddingLeft': '15px',
                                   'paddingRight': '15px',
                                   'width': '300px'}),
                        html.Div([
                            html.Label('Is Anomalous'),
                            dcc.Dropdown(
                                id='is-anomalous',
                                options=Anom_options,
                                value=0,
                                clearable=False,
                                style={'width': '300px'})])
                    ]),
                    dbc.Row([html.H6(id='graph-sif-writeup',
                                     style={"margin-top": "5px"})]),
                    dbc.Row([html.H6(id='low-trans-bus-list',
                                     style={"margin-top": "5px"})])
                ], style={"margin-left": "45px"})
            ])
        ]),

    ]),
    dcc.Download(id="download-monthly-flags-csv"),
])


@ app.callback(
    Output('is-anomalous', 'value'),
    Input('select-month', 'value'))
def update_anom_value(month):
    return 0


@ app.callback(
    Output('select-BU', 'options'),
    Output('select-BU', 'value'),
    Input('is-anomalous', 'value'),
    State('select-month', 'value'))
def update_bu_option(is_anomulus, month):
    BU_list = (graphSIF_df[(graphSIF_df.Is_anomolous == is_anomulus) &
                           (graphSIF_df.Month == month)]
               .GraphSIF_BUs.values)
    BU_list.sort()
    if len(BU_list) > 0:
        return [{'label': BU, 'value': BU} for BU in BU_list], BU_list[0]
    else:
        return [], ''


@ app.callback(
    Output('graph-sif-writeup', 'children'),
    Input('select-BU', 'value'),
    State('select-month', 'value'))
def display_GraphSIF_stats(BU, month):
    return html.Div(html.P(get_GraphSIF_stats(BU, month)))


@ app.callback(
    Output('monthly-bu-flags', 'figure'),
    Input('select-BU', 'value'),
    State('select-month', 'value'))
def create_monthly_flags_chart(BU, month):
    return monthly_flags_chart(BU, month)


@app.callback(
    Output("download-monthly-flags-csv", "data"),
    Input('monthly-bu-flags', 'clickData'),
    State('select-BU', 'value'),
    State('select-month', 'value'))
def click_BUflags_data_download(clic_dat, BU, month):
    if clic_dat is None:
        raise PreventUpdate
    flag_scen = clic_dat['points'][0]['label']
    scen2colnm = {y: x for x, y in flag_lab2name.items()}
    colnm = scen2colnm[flag_scen]
    flags_sub = flags_df[(flags_df[colnm] == 1) &
                         (flags_df.GraphSIF_BUs == BU) &
                         (flags_df.Month == month)]
    return dcc.send_data_frame(flags_sub.to_csv, f"{flag_scen.replace(' ','_')}_{BU}_{month}.csv")


@ app.callback(
    Output('low-trans-bus-list', 'children'),
    Input('select-BU', 'value'),
    State('select-month', 'value'))
def get_lowTrans_BUs(BU, month):
    if BU in low_trans_BUs:
        bus = flags_df[(flags_df.GraphSIF_BUs == BU) &
                       (flags_df.Month == month)].BUSINESS_UNIT.unique()
        bus.sort()
        return f"list of BUs under {BU}: {', '.join(bus)}"
    return ''
