import os
import re
import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
from app_config import app
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objects as go
import _pickle as cPickle
import pandas as pd
import numpy as np
import time
from pdb import set_trace


# Entitty Colors
color_dict = {'VENDOR_ID': '#87CEFA',
              'ACCOUNT':  '#00CC96',  # '#00CC96',
              'Full_Address': '#e6897a',
              'BUSINESS_UNIT': '#b590de'}  # '#AB63FA'

# Funtions

# Import Datasets
# Load data

input_path = './static'


def load_data(input_path, n_months=15):
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


def GraphInp_prep(fraud_df):

    # PageRank df preparation
    pr_stats_df = fraud_df[['Quarter', 'VENDOR_ID', 'NAME1', 'ACCOUNT_CAT', 'BUSINESS_UNIT', 'Full_Address', 'Vendor_PGRK',
                            'Account_PGRK', 'BU_PGRK', 'Address_PGRK', 'Vendor_PR_Var', 'Account_PR_Var', 'BU_PR_Var', 'Address_PR_Var']].copy()

    dic = {'VENDOR_ID': ['Quarter', 'VENDOR_ID', 'Vendor_PGRK', 'Vendor_PR_Var', 'NAME1'],
           'ACCOUNT': ['Quarter', 'ACCOUNT_CAT', 'Account_PGRK', 'Account_PR_Var', 'NAME1'],
           'BUSINESS_UNIT': ['Quarter', 'BUSINESS_UNIT', 'BU_PGRK', 'BU_PR_Var', 'NAME1'],
           'Full_Address': ['Quarter', 'Full_Address', 'Address_PGRK', 'Address_PR_Var', 'NAME1']}

    PageRank_df = pd.DataFrame()
    for x, y in dic.items():
        entity_df = pr_stats_df[y]
        if not x == 'VENDOR_ID':
            entity_df['NAME1'] = ''
        entity_df['Entity_type'] = x
        entity_df.columns = ['Quarter', 'id', 'PageRank',
                             'PGRK_PrctCng', 'NAME1', 'Entity_type']
        entity_df_agg = entity_df.groupby(
            list(entity_df.columns), dropna=False).size().reset_index(name='n_trans')
        PageRank_df = PageRank_df.append(entity_df_agg)

    PageRank_df = pd.pivot_table(PageRank_df,
                                 index=['id', 'Entity_type', 'NAME1'],
                                 columns=['Quarter'],
                                 values=['PageRank', 'PGRK_PrctCng', 'n_trans']).reset_index()
    PageRank_df.columns = ['_'.join(col).strip('_')
                           for col in PageRank_df.columns]
    PageRank_df['id'] = PageRank_df.id.apply(
        lambda x: re.sub(', ', ',<br>', x, 2))

    # Edges df preparation
    Edges_stats = fraud_df[['VENDOR_ID', 'Full_Address',
                            'ACCOUNT_CAT', 'BUSINESS_UNIT', 'Quarter', 'GROSS_AMT']].copy()

    Edges_stats = Edges_stats.melt(id_vars=['Quarter', 'VENDOR_ID', 'GROSS_AMT'], value_vars=[
                                   'ACCOUNT_CAT', 'BUSINESS_UNIT', 'Full_Address'])

    Edges_df = (Edges_stats
                .groupby(list(Edges_stats.columns.drop('GROSS_AMT')), dropna=False)['GROSS_AMT']
                .agg([('n_edges', 'size'), ('Total_Amt', 'sum')])
                .reset_index())
    Edges_df.drop('variable', axis=1, inplace=True)
    Edges_df.rename(columns={'VENDOR_ID': 'src', 'value': 'dst'}, inplace=True)
    Edges_df['dst'] = Edges_df.dst.apply(lambda x: re.sub(', ', ',<br>', x, 2))
    Edges_df['Edge_Label'] = 'Transactions: ' + \
        Edges_df.n_edges.astype(str)+'<br>'+'Total Amount: ' + \
        Edges_df.Total_Amt.round(2).astype(str)

    return PageRank_df, Edges_df


def plot_TopVendor_bar(Vendor: 'Vendor', quarter: 'Quarter'):
    df_sub = (df[(df.Entity_type == 'VENDOR_ID') & (~df[f'PGRK_PrctCng_{quarter}'].isna())]
              [['id', f'PageRank_{quarter}',
                  f'PGRK_PrctCng_{quarter}', 'id_2']]
              .sort_values(f'PGRK_PrctCng_{quarter}', ascending=True).tail(10))
    # df_sub = df_sub[~df_sub.id.str.contains('(^ACE |^CHUBB )')]
    # df_sub = df_sub.tail(10)
    df_sub['selected_Vendor'] = '0'
    df_sub.loc[df_sub.id == Vendor, 'selected_Vendor'] = '1'
    # df_sub = df_sub.sort_values(f'PGRK_PrctCng_{quarter}',ascending=True)
    fig = px.bar(df_sub,
                 y="id",
                 x=f'PGRK_PrctCng_{quarter}',
                 text='id_2',
                 category_orders={'id': df_sub.id.tolist()[::-1]},
                 color='selected_Vendor',
                 hover_data=[f'PageRank_{quarter}'],
                 color_discrete_map={'0': '#636efa'},
                 labels={'id': 'Vendor',
                         f'PGRK_PrctCng_{quarter}': 'PageRank Variation (%)',
                         f'PageRank_{quarter}': 'PageRank'},
                 orientation='h')
    fig.update_layout(transition_duration=100,
                      autosize=True,
                      title_xanchor='left',
                      title=f"Top Vendors by PageRank variation (%) in {quarter}",
                      xaxis=dict(visible=True, side='top', title=None,
                                 showgrid=True, zeroline=True, showticklabels=True),
                      yaxis=dict(visible=False, showgrid=False,
                                 zeroline=False, showticklabels=False),
                      margin=dict(t=50, l=0, r=0, b=0),
                      height=350,  # 400
                      showlegend=False)
    fig.update(layout_coloraxis_showscale=False)
    return fig


def Plot_VenGraph_new(pos_df, Edges_df_sub, quarter):
    edge_x = []
    edge_y = []
    lab_pos = []
    pos = dict(zip(pos_df.Entity, pos_df[['x', 'y']].values))

    for edge in Edges_df_sub[['src', 'dst']].values:
        x0, y0 = list(pos[edge[0]])  # list(pos[edge[0]])
        x1, y1 = list(pos[edge[1]])
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        lab_pos.append([(x1+x0)/2, (y0+y1)/2])

    print(pos_df.Entity.head())
    PageRank_df_sub = df[(df.id_2.isin(
        pos_df.Entity))]
    PageRank_df_sub = PageRank_df_sub.merge(
        pos_df, how='left', left_on='id_2', right_on='Entity')
    PageRank_df_sub['size'] = 10

    selected_cols = [col for col in PageRank_df_sub.columns if '_Q' in col]
    PageRank_df_sub.loc[:, selected_cols] = PageRank_df_sub[selected_cols].fillna(
        'NA')

    fig = (px.scatter(
        PageRank_df_sub, x="x", y="y",
        color="Entity_type",
        color_discrete_map=color_dict,
        custom_data=[f'PageRank_{quarter}', f'PGRK_PrctCng_{quarter}'],
        size='size',
        opacity=0.75,
        size_max=50,
        text='id',
        labels={'VENDOR_ID': 'Vendor',
                'ACCOUNT': 'Account',
                'Full_Address': 'Address',
                'BUSINESS_UNIT': 'Bussiness Unit'},
        hover_data=['id'],
        # textfont=dict(
        #     family="sans serif",
        #     size=18,
        #     color="LightSeaGreen"),
    ))

    fig.update_traces(
        hovertemplate="<br>".join([
            "PageRank: %{customdata[0]}",
            "PageRank Variation: %{customdata[1]}",
        ]))

    fig.add_trace(go.Scatter(

        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        opacity=0.5,
        mode='lines',
        showlegend=False))

    for i in range(len(Edges_df_sub)):
        fig.add_annotation(x=lab_pos[i][0],
                           y=lab_pos[i][1],
                           text=Edges_df_sub.Edge_Label[i],
                           showarrow=False)

    fig.update_layout(
        transition_duration=500,
        autosize=True,
        clickmode='event+select',
        title=None,
        xaxis=dict(visible=False, showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(visible=False, showgrid=False,
                   zeroline=False, showticklabels=False),
        legend=dict(orientation="h",
                    title_text='',
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=0.0),
        margin=dict(t=0, l=40, r=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        height=830, width=1400)  # 970
    return fig


def Entity_PRvar_bar(Vendor, quarter):
    Edges_df_sub = Edges_df[(Edges_df.src == Vendor) &
                            (Edges_df.Quarter == quarter)]

    get_q_ind = np.where(Quarters == quarter)[0]
    prev_quarter = Quarters[get_q_ind-1][0]

    Edges_df_sub = Edges_df_sub[['dst']].merge(
        df[['id', 'Entity_type', f'PageRank_{prev_quarter}', f'PageRank_{quarter}']], left_on='dst', right_on='id', how='left')

    agg_Ent_PR_var = (Edges_df_sub.groupby('Entity_type')
                      .agg({f'PageRank_{prev_quarter}': np.nanmean,
                            f'PageRank_{quarter}': np.nanmean}).reset_index())

    agg_Ent_PR_var[f'PGRK_PrctCng_{quarter}'] = (
        (agg_Ent_PR_var[f'PageRank_{quarter}'] - agg_Ent_PR_var[f'PageRank_{prev_quarter}'])/agg_Ent_PR_var[f'PageRank_{prev_quarter}']*100).round(2)

    fig = px.bar(agg_Ent_PR_var,
                 x="Entity_type",
                 y=f'PGRK_PrctCng_{quarter}',
                 # hover_data=[f'PGRK_PrctCng_{quarter}'],
                 hover_data=[f'PGRK_PrctCng_{quarter}',
                             f'PageRank_{prev_quarter}',
                             f'PageRank_{quarter}'],
                 color="Entity_type",
                 color_discrete_map=color_dict,
                 labels={'Entity_type': 'Entity',
                         f'PGRK_PrctCng_{quarter}': 'PageRank Variation (%)',
                         f'PageRank_{prev_quarter}': f'{prev_quarter} Avg PageRank',
                         f'PageRank_{quarter}': f'{quarter} Avg PageRank'})
    fig.update_layout(transition_duration=500,
                      autosize=True,
                      title_xanchor='left',
                      yaxis=dict(visible=True, title=None, showgrid=True,
                                 zeroline=True, showticklabels=True),
                      xaxis=dict(visible=True, title=None, showgrid=False,
                                 zeroline=False, showticklabels=True),
                      title=None,
                      margin=dict(t=0, l=40, r=0, b=0),
                      title_font_size=15,
                      height=270,  # 350
                      showlegend=False)
    return fig


def Trans_vol_jump(Vendor, quarter):
    get_q_ind = np.where(Quarters == quarter)[0]
    prev_quarter = Quarters[get_q_ind-1][0]

    cur_Qvol = df.loc[(df.id == Vendor), f'n_trans_{quarter}']
    prev_Qvol = df.loc[(df.id == Vendor), f'n_trans_{prev_quarter}']
    Valu_jump = (((cur_Qvol-prev_Qvol)/prev_Qvol)*100).max()
    # int(prev_Qvol.item())
    return int(cur_Qvol.item()), int(prev_Qvol.item()), int(Valu_jump)


fraud_df, _ = load_data(input_path)

# ## filter transactions from last 5 quarters only
# quats = fraud_df.Quarter.unique()
# quats.sort()

# fraud_df = fraud_df[fraud_df.Quarter.isin(quats[-5:])].Quarter.unique()

PageRank_df, Edges_df = GraphInp_prep(fraud_df)

# Preping Inputs
cols = [col for col in PageRank_df.columns if 'PGRK_PrctCng_' in col]
for col in cols:
    PageRank_df[col] = PageRank_df[col].astype(float)
PageRank_df = PageRank_df.round(2)

Edges_df = Edges_df.merge(
    PageRank_df[['id', 'Entity_type']], left_on='dst', right_on='id', how='left')


def shroten_name(name, upto=4):
    wrds = name.split()
    if len(wrds) > upto:
        wrds = wrds[:upto]
    return ' '.join(wrds)


PageRank_df.NAME1 = PageRank_df.NAME1.fillna(value='NA NA')
PageRank_df['NAME1'] = PageRank_df.NAME1.apply(lambda x: x.replace('THE ', ''))
PageRank_df['shorten_name'] = PageRank_df.NAME1.apply(
    lambda x: shroten_name(x))

Edges_df = Edges_df.merge(PageRank_df[['id', 'shorten_name']].copy(), left_on='src',
                          right_on='id', how='left')

PageRank_df['id_2'] = PageRank_df.id
PageRank_df.loc[PageRank_df.Entity_type == 'VENDOR_ID',
                'id_2'] = PageRank_df.loc[PageRank_df.Entity_type == 'VENDOR_ID', 'shorten_name']

# Edges_df.rename(columns={'shorten_name': 'src'}, inplace=True)
# Edges_df = Edges_df.iloc[:, 1:]

ent_type_dict = pd.Series(PageRank_df.Entity_type.values,
                          index=PageRank_df.id).to_dict()
df = PageRank_df.copy()

# Edges_df = Edges_df.merge(df[['id', 'Entity_type']].copy(),
#                           left_on='dst', right_on='id', how='left')

# Vendor id 2 name mappings
ven_inds = df.Entity_type == 'VENDOR_ID'
Ven_id2name = dict(zip(df.id[ven_inds], df.id_2[ven_inds]))

# unique sorted quarter
Quarters = Edges_df.Quarter.unique()
Quarters.sort()

Quarters_options = [{'label': q,  'value': q} for q in Quarters[1:]]

print(Edges_df.columns)
# LayOut

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(children='PageRank Analysis for top vendors',
                            style={"text-align": "center"}),
                    dbc.Row([
                        dbc.Col([html.H5('Select Quarter:')],
                                style={'text-align': 'right',
                                       'padding': '5px'}),
                        dbc.Col([dcc.Dropdown(
                            id='select-quarter',
                            options=Quarters_options,
                            value=Quarters[1],
                            clearable=False)
                        ])
                    ]),
                    dcc.Graph(id='top-vendor-bar'),
                    html.Hr(style={"width": "3"}),
                    html.H6(id='trans-jump-id',
                            style={"margin-top": "5px"}),
                    html.Hr(style={"width": "3"}),
                    html.H6(id='entity-chart-title',
                            style={"margin-top": "5px"}),
                    dcc.Graph(id='entity-prv-bar')
                ])
            ], id='top-vendors', width=3),

            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5(id='select-quarter-vendor-graph',
                                         style={'text-align': 'right',
                                                "margin-top": "10px",
                                                "margin-right": "10px",
                                                "margin-left": "45px"})], width=6),
                        dbc.Col([dcc.RadioItems(
                            id='user-selected-quarter',
                            options=Quarters_options,
                            value=Quarters[1],
                            labelStyle={'display': 'inline-block',
                                        "margin-top": "10px",
                                        "margin-right": "15px",
                                        "margin-left": "15px"}
                        )], width=6)
                    ]),
                    html.Div([dcc.Graph(id='vendor-graph')],
                             style={'text-align': 'center'})
                ])
            ], width=9)
        ])
    ]),
    dcc.Store(id='selected-vendor'),
    dcc.Store(id='entity-list'),
    dcc.Loading(dcc.Store(id="intermediate-value"),
                fullscreen=False, type="dot")
])


# App Callbacks

@ app.callback(
    Output('select-quarter-vendor-graph', 'children'),
    Input('selected-vendor', 'data'))
def update_vendor_radiotit(Vendor):
    return f"For {Vendor.get('vendor')} select quarter: "


@ app.callback(
    Output('user-selected-quarter', 'options'),
    [Input('selected-vendor', 'data')])
def update_Vend_Quarters(Vendor):
    Vendor = Vendor.get('vendor')
    cols = [col for col in df.columns if 'PageRank_' in col]
    a = ~df.loc[df.id == Vendor, cols].squeeze(axis=0).isna()
    idxs = a.index[a]
    Vend_quaters = [col.replace('PageRank_', '') for col in idxs]
    Vend_quaters.sort()
    # Vend_quaters = Vend_quaters.tolist()
    if Quarters[0] in Vend_quaters:
        Vend_quaters.remove(Quarters[0])
    options = [{'label': opt, 'value': opt} for opt in Vend_quaters]
    return options


@ app.callback(
    Output('entity-chart-title', 'children'),
    Input('selected-vendor', 'data'),
    State('select-quarter', 'value'))
def update_ent_chart_tit(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    return f"Entity PageRank variation (%) for {Vendor} in {quarter}"


@ app.callback(
    Output('selected-vendor', 'data'),
    Input('top-vendor-bar', 'clickData'),
    Input('select-quarter', 'value'))
def update_vendor(Vendor, quarter):
    if Vendor is not None:
        Vendor = Vendor['points'][0]['label']
    else:
        df_sub = (df[(df.Entity_type == 'VENDOR_ID') & (~df[f'PGRK_PrctCng_{quarter}'].isna())]
                  [['id', f'PageRank_{quarter}', f'PGRK_PrctCng_{quarter}']]
                  .sort_values(f'PGRK_PrctCng_{quarter}', ascending=False).head(1))
        # df_sub = df_sub[~df_sub.id.str.contains('(^ACE |^CHUBB )')].head(1)
        Vendor = df_sub.id.item()
    return {'vendor': Vendor}


@ app.callback(
    Output('user-selected-quarter', 'value'),
    Input('select-quarter', 'value'),
    Input('top-vendor-bar', 'clickData'))
def update_quarter_radio(quarter: 'Quarter', Vendor: 'Vendor'):
    return quarter


@ app.callback(
    Output('top-vendor-bar', 'figure'),
    Input('selected-vendor', 'data'),
    Input('select-quarter', 'value'))
def create_TopVendor_bar(Vendor, quarter: 'Quarter'):
    Vendor = Vendor.get('vendor')
    return plot_TopVendor_bar(Vendor, quarter)


@ app.callback(
    Output('top-vendor-bar', 'clickData'),
    Input('select-quarter', 'value'))
def update_hover_data(quarter):
    return None


@ app.callback(
    Output('vendor-graph', 'clickData'),
    Output('vendor-graph', 'selectedData'),
    #     Input('selected-vendor', 'data'),
    #     Input('user-selected-quarter', 'value'))
    Input("intermediate-value", "data"))
def update_graph_click_data(data):
    return None, None


@ app.callback(
    Output('entity-list', 'data'),
    Input('vendor-graph', 'selectedData'),
    [State('vendor-graph', 'clickData'),
     State('entity-list', 'data')])
def update_graph_click_data(select_data, click_data, ent_list):
    if click_data is None:
        res = {'entityList': []}
    else:
        ent = click_data['points'][0]['text']
        if ((df[df.id == ent].Entity_type.item() == 'VENDOR_ID') & (ent in ent_list['entityList'])):
            lst = []
        elif ent in ent_list['entityList']:
            lst = ent_list['entityList']
            lst.remove(ent)
        else:
            lst = ent_list['entityList']
            lst.append(ent)
        res = {'entityList': lst}
    time.sleep(0.5)
    return res


@ app.callback(
    Output('entity-prv-bar', 'figure'),
    Input('selected-vendor', 'data'),
    State('select-quarter', 'value'))
def create_Ent_bar(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    return Entity_PRvar_bar(Vendor, quarter)

# @app.callback(
#     Output('vendor-trans-vol-bar', 'figure'),
#     Input('selected-vendor', 'data'),
#     State('select-quarter', 'value'))
# def create_plot_transVol(Vendor,quarter):
#     Vendor = Vendor.get('vendor')
#     return plot_transVol(Vendor,quarter)


@ app.callback(
    Output('trans-jump-id', 'children'),
    Input('selected-vendor', 'data'),
    State('select-quarter', 'value'))
def update_trans_jump(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    get_q_ind = np.where(Quarters == quarter)[0]
    prev_quarter = Quarters[get_q_ind-1][0]
    curr_vol, prv_vol, prcnt = Trans_vol_jump(Vendor, quarter)

    # return f'Transaction volume increase for \n{Vendor} from {prev_quarter} to {quarter}: {prcnt}% <br>'
    # return f'Transaction volume increase for {Vendor} from {prev_quarter} to {quarter}: {prcnt}% <br/> Transactions for vendor {Vendor} jumped from {prv_vol} in {prev_quarter} to {curr_vol} in {quarter}'
    return (html.Div(html.P([f'Transaction volume increase for {Vendor} from {prev_quarter} to {quarter}: {prcnt}%',
                             html.Br(),
                             html.Br(),
                             f'Transactions for vendor {Vendor} jumped from {prv_vol} in {prev_quarter} to {curr_vol} in {quarter}'])))


@ app.callback(
    Output("intermediate-value", "data"),
    [Input('selected-vendor', 'data'),
     Input('user-selected-quarter', 'value')])
def get_edges(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    Edges_df_sub = Edges_df[(Edges_df.src == Vendor) &
                            (Edges_df.Quarter == quarter)].copy()

    ent_edges_df = Edges_df[Edges_df.dst.isin(Edges_df_sub.dst) & (Edges_df.Quarter == quarter) &
                            (Edges_df.Entity_type == 'Full_Address') & (Edges_df.src != Vendor)].copy()
    ent_edges_df.rename(columns={'src': 'dst', 'dst': 'src'}, inplace=True)
    ent_edges_df = ent_edges_df[Edges_df_sub.columns]
    Edges_df_sub = pd.concat(
        [Edges_df_sub, ent_edges_df], axis=0).reset_index(drop=True)

    Edges_df_sub.rename(
        columns={'src': 'mtp', 'shorten_name': 'src'}, inplace=True)
    # compute node positions
    G = nx.from_pandas_edgelist(
        Edges_df_sub, 'src', 'dst')
    pos = nx.spring_layout(G)

    pos_df = pd.DataFrame(pos).T.reset_index(drop=False)
    pos_df.columns = ['Entity', 'x', 'y']  # shorten_name
    # print(pos_df.columns)
    # print(Edges_df_sub.columns)
    return {'pos_df': pos_df.to_json(date_format='iso', orient='split'),
            'edges_df': Edges_df_sub.to_json(date_format='iso', orient='split')}
#     return {'pos_df':pos_df,
#             'edges_df':Edges_df_sub}


# @app.callback(
#     Output('vendor-graph', 'figure'),
#     Input('selected-vendor', 'data'),
#     Input('user-selected-quarter', 'value'))
# def Create_graph(Vendor, quarter):
#     Vendor = Vendor.get('vendor')
#     return Plot_VenGraph(Vendor, quarter)

@ app.callback(
    Output('vendor-graph', 'figure'),
    Input('entity-list', 'data'),
    [State('selected-vendor', 'data'),
     State('user-selected-quarter', 'value'),
     State("intermediate-value", "data")])
def Create_graph(ent_list, Vendor, quarter, json_data):
    entList = ent_list['entityList']
    Vendor = Vendor.get('vendor')

    Vendor_name = Ven_id2name[Vendor]
    # json_data['pos_df']
    pos_df = pd.read_json(json_data['pos_df'], orient='split')
    edges_df_sub = pd.read_json(
        json_data['edges_df'], orient='split')  # json_data['edges_df']

    mod_ent_list = [Ven_id2name[key]
                    if key in Ven_id2name else key for key in entList]
    print(mod_ent_list)
    ent_edges = edges_df_sub[edges_df_sub.src.isin(mod_ent_list)]
    ent_pos_df = pos_df[pos_df.Entity.isin(
        set(ent_edges.src.tolist()+ent_edges.dst.tolist()+[Vendor_name]))]
    print(pos_df.shape, ent_pos_df.shape)
    return Plot_VenGraph_new(ent_pos_df, ent_edges, quarter)
