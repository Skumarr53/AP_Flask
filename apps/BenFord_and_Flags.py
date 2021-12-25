import os
import re
import datetime
from dateutil.relativedelta import relativedelta
from app_config import app
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import _pickle as cPickle
import pandas as pd
import numpy as np
import calendar
from pdb import set_trace


# User Inputs

# 'Is_Multi_VendorId',
# 'Is_Multi_VendorAddress',
# 'Is_Invoice_Sequential',
# 'Is_Inv_inconsistent',
# 'Is_pymt_withinSevenDays',
# 'Is_Paid_grtn_InvAmt',
# 'is_roundedAmount_prcnt_sig50',
# 'is_roundedAmount_prcnt_sig80',
# 'Is_NegBalance',
# 'Is_duplicateInvID',
# 'Is_duplicate_InvDt',
# 'Is_SingleApprover',
# 'Is_Vendor_UsingMultiAcc',
# 'Is_pymt_priorToInvoice',
# 'Is_MultiVendor_SameAdd',
# 'is_approvalLimit_prcnt_sig',
# 'Is_PGvar_significant'

Flag_cols = ['Is_Multi_VendorId',
             'Is_Multi_VendorAddress',
             'Is_Invoice_Sequential',
             'Is_Inv_inconsistent',
             'Is_pymt_withinSevenDays',
             'Is_Paid_grtn_InvAmt',
             'is_roundedAmount_prcnt_sig50',
             'is_roundedAmount_prcnt_sig80',
             'Is_NegBalance',
             'Is_duplicateInvID',
             'Is_duplicate_InvDt',
             'Is_SingleApprover',
             'Is_Vendor_UsingMultiAcc',
             'Is_pymt_priorToInvoice',
             'Is_MultiVendor_SameAdd',
             'is_approvalLimit_prcnt_sig',
             'Is_PGvar_significant',
             'Is_VendorAccChanged_3M',
             'Is_VendorAccChanged_1Yr',
             'Rulesbaed_outliers']

flag_lab2name = {'Is_Multi_VendorId': 'Transactions  from vendor with multiple vendor ids',
                 'Is_Multi_VendorAddress': 'Transaction from vendor with multiple Addresses',
                 'Is_Invoice_Sequential': 'Transactions from vendor with Sequential Invoices',
                 'Is_Inv_inconsistent': 'Transactions with Inconsistent Invoices',
                 'Is_pymt_withinSevenDays': 'Transactions payment within Seven days',
                 'Is_Paid_grtn_InvAmt': 'Payment higher than invoice amount transactions',
                 'is_roundedAmount_prcnt_sig50': 'Transactions with more than 50% rounded amounts (last 3 months)',
                 'is_roundedAmount_prcnt_sig80': 'Transactions with more than 80% rounded amounts (last 3 months)',
                 'Is_NegBalance': 'Negative Balances transactions',
                 'Is_duplicateInvID': 'Duplicate invoices transactions by Invoice no',
                 'Is_duplicate_InvDt': 'Duplicate invoices transactions by Invoice date',
                 'Is_SingleApprover': 'Transactions from vendor with Single Approver',
                 'Is_Vendor_UsingMultiAcc': 'Transactions from vendor using multiple accounts',
                 'Is_pymt_priorToInvoice': 'Transactions  with payment Prior to Invoice date',
                 'Is_MultiVendor_SameAdd': 'Transactions  from multiple vendors with same Address',
                 'is_approvalLimit_prcnt_sig': 'Transactions within approval limit from the same approver id (last 3 months)',
                 'Is_PGvar_significant': 'Transactions from Vendor with significant PageRank variation(>20%)',
                 'Is_VendorAccChanged_3M': 'Transactions from Vendor who changed account in past 3 months',
                 'Is_VendorAccChanged_1Yr': 'Transactions from Vendor who changed account in past 1 year',
                 'Rulesbaed_outliers': 'Rule based Outlier transactions'
                 }

colorlist = ['#636EFA',
             '#EF553B',
             '#00CC96',
             '#AB63FA',
             '#FFA15A',
             '#19D3F3',
             '#FF6692',
             '#B6E880',
             '#FF97FF',
             '#FECB52',
             '#0048BA',
             '#C46210',
             '#A67B5B',
             '#EE82EE',
             '#7FFFD4',
             '#E9DCD3',
             '#FFA07A']

# Import Datasets
# Load data
# Load data

input_path = './static'


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


def Benfords_plot(BU, start, end):
    if BU != 'ALL':
        Lead_ser = flags_df.lead_digit[(flags_df.lead_digit != '0') & (
            flags_df.lead_digit != '-') & (flags_df.BUSINESS_UNIT == BU) &
            (flags_df.Month >= start) & (flags_df.Month <= end)]
    else:
        Lead_ser = flags_df.lead_digit[(
            flags_df.lead_digit != '0') & (flags_df.lead_digit != '-') &
            (flags_df.Month >= start) & (flags_df.Month <= end)]

    lead_df = Lead_ser.value_counts(normalize=True).sort_index().reindex(
        [str(i) for i in range(1, 10)]).to_frame().reset_index()
    lead_df.columns = ['digit', 'freq']
    lead_df['freq'] = lead_df['freq']*100
    lead_df = lead_df.sort_values('digit').round(2)

    BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]

    fig = px.bar(lead_df, y='freq', x='digit',
                 labels={'digit': 'First digit of Gross Amount',
                         'freq': 'Relative Frequency (%)'},
                 text='freq')
    fig.add_trace(go.Scatter(y=BENFORD, x=list(range(1, 10))))
    fig.update_layout(transition_duration=500,
                      autosize=True,
                      title=f"Benford's law distribution for Bussiness Unit {BU} <br>between {start} & {end} | transactions: {len(Lead_ser)}" if BU != 'ALL' else f"Benford's law distribution across Bussiness Units <br>between {start} & {end} | transactions: {len(Lead_ser)}",
                      showlegend=False,
                      width=450,
                      height=450,
                      margin=dict(t=50, l=0, r=0, b=0))
    return fig


def Flag_prop_plot(BU, start, end):
    if BU != 'ALL':
        flags_df_sub = flags_df[(flags_df.BUSINESS_UNIT == BU) &
                                (flags_df.Month >= start) &
                                (flags_df.Month <= end)]
    else:
        flags_df_sub = flags_df[(flags_df.Month >= start) &
                                (flags_df.Month <= end)]

    flags_prop = flags_df_sub[Flag_cols].sum()

    flags_prop_df = flags_prop.to_frame().reset_index()
    flags_prop_df.columns = ['flag_label', 'count']
    flags_prop_df['count'] = flags_prop_df['count']
    flags_prop_df = flags_prop_df.round(2)
    flags_prop_df['label_name'] = flags_prop_df.flag_label.replace(
        flag_lab2name)

    fig = px.bar(flags_prop_df, y='count', x='label_name',
                 labels={'label_name': 'Scenario',
                         'count': 'Transaction count'},
                 color='label_name',
                 color_discrete_map=color_dict)

    fig.update_layout(transition_duration=500,
                      title=f"Flagged transactions for BU {BU} between {start} and {end}" if BU != 'ALL' else f"Flagged transactions across BUs between {start} and {end}",
                      autosize=True,
                      legend=dict(title_text=''),
                      xaxis=dict(visible=False, showgrid=False,
                                 zeroline=False, showticklabels=False),
                      margin=dict(t=30, l=30, r=0, b=0),  # width=1250
                      width=1400,
                      height=450)
    return fig

# Vendor account changes chart


def Top_Vendors_AccChange3M_plot():
    top_vend_acc = (flags_df[['VENDOR_ID', 'VENDOR_NAME_SHORT', 'n_AccChanges_ByVendor_3M']]
                    .groupby('VENDOR_ID').max().reset_index()
                    .sort_values('n_AccChanges_ByVendor_3M', ascending=False)).head(10)
    fig = px.bar(top_vend_acc, y='n_AccChanges_ByVendor_3M', x='VENDOR_NAME_SHORT',
                 labels={'VENDOR_NAME_SHORT': 'Vendor',
                         'n_AccChanges_ByVendor_3M': 'Changes count'})

    fig.update_traces(marker_color='#00CC96')

    fig.update_layout(transition_duration=500,
                      title="Top vendors by Account changes in past 3 Months",
                      autosize=True,
                      showlegend=False,
                      xaxis=dict(visible=True, showgrid=False,
                                 zeroline=False, showticklabels=True),
                      margin=dict(t=30, l=30, r=0, b=0))
    return fig


def Top_Vendors_AccChange1Y_plot():
    top_vend_acc = (flags_df[['VENDOR_ID', 'VENDOR_NAME_SHORT', 'n_AccChanges_ByVendor_1Yr']]
                    .groupby('VENDOR_ID').max().reset_index()
                    .sort_values('n_AccChanges_ByVendor_1Yr', ascending=False)).head(10)
    fig = px.bar(top_vend_acc, y='n_AccChanges_ByVendor_1Yr', x='VENDOR_NAME_SHORT',
                 labels={'VENDOR_NAME_SHORT': 'Vendor',
                         'n_AccChanges_ByVendor_1Yr': 'Changes count'})
    fig.update_traces(marker_color='#FFA15A')
    fig.update_layout(transition_duration=500,
                      title="Top vendors by Account changes in past 1 year",
                      autosize=True,
                      showlegend=False,
                      xaxis=dict(visible=True, showgrid=False,
                                 zeroline=False, showticklabels=True),
                      margin=dict(t=30, l=30, r=0, b=0))
    return fig


# Loading Datasets and preping inputs
flags_df, _ = load_data(input_path)
flags_df.lead_digit = flags_df.lead_digit.astype('str')

req_cols = [col for col in flags_df.columns if col not in Flag_cols+['RuleBased_scores', 'Outliers_k20',
                                                                     'Outliers_k40', 'n_transaction_last3mnth_byApprover', 'n_transaction_last3mnth_byVendor']]

flags_df.INVOICE_DT = pd.to_datetime(flags_df.INVOICE_DT)
# flags_df['trans_month'] = flags_df['INVOICE_DT'].dt.month

# month_num_to_name = {month: index for index,
#                      month in enumerate(calendar.month_abbr) if month}
# flags_df['lead_digit'] = flags_df['GROSS_AMT'].astype(str).str[0]

# flags_df = flags_df.drop('Is_MultiVendor_SameAcc', axis=1)

# month_num_to_name_rev = {y: x for x, y in month_num_to_name.items()}


BU_options = [{'label': 'ALL', 'value': 'ALL'}]
for bu in flags_df.BUSINESS_UNIT.value_counts().index:
    BU_options.append({'label': bu, 'value': bu})


mnths = flags_df.Month.unique()
mnths.sort()

Month_options = []
for mnth in mnths:
    Month_options.append({'label': mnth, 'value': mnth})


color_dict = {x: y for x, y in zip(flag_lab2name.values(), colorlist)}


# Layout
layout = html.Div([
    dbc.Container([
        dbc.Row([html.H3(children='Benford and Flag Analysis')]),
        dbc.Row([
            dbc.Col([
                html.Div([html.Label('Select Bussiness unit'),
                          dcc.Dropdown(id='select-bu',
                                       options=BU_options,
                                       value='ALL',
                                       clearable=False,
                                       style={'width': '150px'}),
                          dcc.Graph(id='benford-bar-graph')])
            ], id='top-vendors', width=3),
            dbc.Col([
                html.Div([
                    dbc.Row([
                        html.Div([
                            html.Label('Start Month:'),
                            dcc.Dropdown(
                                id='start-month',
                                options=Month_options,
                                value=mnths[0],
                                clearable=False,
                                style={'width': '100px'}), ],
                            style={'paddingLeft': '15px',
                                   'paddingRight': '15px'}),
                        html.Div([
                            html.Label('End Month:'),
                            dcc.Dropdown(
                                id='end-month',
                                options=Month_options,
                                value=mnths[-1],
                                clearable=False,
                                style={'width': '100px'})])
                    ]),
                    dbc.Row([dcc.Graph(id='flag-prop-graph',
                                       style={"width": '65%'})])
                ], style={"margin-left": "45px"})
            ], width=9)
        ]),
        html.Hr(style={"width": "5"}),
        dbc.Row([dcc.Graph(id='vend-acc-changes-1Y-graph',
                           style={"margin-top": "10px",
                                  "width": '50%'}),
                 dcc.Graph(id='vend-acc-changes-3M-graph',
                           style={"margin-top": "10px",
                                  "width": '50%'})])
    ]),
    dcc.Download(id="download-dataframe-csv"),
    dcc.Download(id="Benford-download-dataframe-csv")
])


@app.callback(
    Output("Benford-download-dataframe-csv", "data"),
    Input('benford-bar-graph', 'clickData'),
    State('select-bu', 'value'),
    State('start-month', 'value'),
    State('end-month', 'value'))
def Benford_click_data_download(clic_dat, BU, start, end):
    if clic_dat is None:
        raise PreventUpdate
    digit = clic_dat['points'][0]['label']
    flags_df_sub = flags_df[(flags_df['lead_digit'] == digit) &
                            (flags_df.BUSINESS_UNIT == BU) &
                            (flags_df.Month >= start) &
                            (flags_df.Month <= end)]
    return dcc.send_data_frame(flags_df_sub.to_csv, f"Benford_LeadDigit{digit}_{BU}_{start}_{end}.csv.csv")


@app.callback(
    Output('benford-bar-graph', 'figure'),
    Input('select-bu', 'value'),
    Input('start-month', 'value'),
    Input('end-month', 'value'))
def create_benford_plot(bu, start, end):
    return Benfords_plot(bu, start, end)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input('flag-prop-graph', 'clickData'),
    State('select-bu', 'value'),
    State('start-month', 'value'),
    State('end-month', 'value'))
def click_data_download(clic_dat, BU, start, end):
    if clic_dat is None:
        raise PreventUpdate
    scen = clic_dat['points'][0]['label']
    scen2colnm = {y: x for x, y in flag_lab2name.items()}
    colnm = scen2colnm[scen]
    flags_df_sub = flags_df[(flags_df[colnm] == 1) &
                            (flags_df.BUSINESS_UNIT == BU) &
                            (flags_df.Month >= start) &
                            (flags_df.Month <= end)]
    return dcc.send_data_frame(flags_df_sub.to_csv,
                               f"{scen.replace(' ','_')}_{BU}_{start}_{end}.csv")


@ app.callback(
    Output('flag-prop-graph', 'figure'),
    Input('select-bu', 'value'),
    Input('start-month', 'value'),
    Input('end-month', 'value'))
def create_Flag_prop_plot(BU, start, end):
    return Flag_prop_plot(BU, start, end)


@ app.callback(
    Output('vend-acc-changes-3M-graph', 'figure'),
    Input('select-bu', 'value'))
def create_Top_Vendors_AccChange3M_plot(BU):
    return Top_Vendors_AccChange3M_plot()


@ app.callback(
    Output('vend-acc-changes-1Y-graph', 'figure'),
    Input('select-bu', 'value'))
def create_Top_Vendors_AccChange1Y_plot(BU):
    return Top_Vendors_AccChange1Y_plot()
