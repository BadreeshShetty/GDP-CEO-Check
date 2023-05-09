import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import random

# Importing GDP, GDP/Capita and Population Data


def gdp_gdp_capita_population():
    # GDP, GDP/Capita, Population Data
    imf_data_gdp = pd.read_excel("IMF-GDP-Data.xls")
    imf_data_gdp_capita = pd.read_excel("IMF-GDP-Per-Capita-Data.xls")
    imf_data_pop = pd.read_excel("IMF-Population.xls")

    imf_data_gdp = imf_data_gdp.replace('no data', 0, regex=True)
    imf_data_gdp_capita = imf_data_gdp_capita.replace('no data', 0, regex=True)
    imf_data_pop = imf_data_pop.replace('no data', 0, regex=True)

    imf_data_gdp.columns = imf_data_gdp.columns.astype(str)
    imf_data_gdp_capita.columns = imf_data_gdp_capita.columns.astype(str)
    imf_data_pop.columns = imf_data_pop.columns.astype(str)

    imf_data_gdp.rename(columns={
                        'GDP, current prices (Billions of U.S. dollars)': 'country'}, inplace=True)
    imf_data_gdp_capita.rename(columns={
                               'GDP per capita, current prices\n (U.S. dollars per capita)': 'country'}, inplace=True)
    imf_data_pop.rename(
        columns={'Population (Millions of people)': 'country'}, inplace=True)

    melted_imf_data_gdp = imf_data_gdp.melt(
        id_vars=['country'], var_name='year', value_name='GDP')
    melted_imf_data_gdp_capita = imf_data_gdp_capita.melt(
        id_vars=['country'], var_name='year', value_name='GDP/Capita')
    melted_imf_data_pop = imf_data_pop.melt(
        id_vars=['country'], var_name='year', value_name='Population')

    return melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop

# Importing SP 500 Data


def sdf_i():
    sdf = pd.read_csv("SP500.csv")
    return sdf

# Importing Russell 3000 Data


def rdf_i():
    rdf = pd.read_csv("Russell3000.csv")
    return rdf

# Scatter Plot for SP500 and Russell 3000


def scatter_plot_options_sp_rus(df, type_index):
    options = ['median_worker_pay', 'pay_ratio', 'salary',
               'totalRevenue', 'grossProfits', 'freeCashflow', 'operatingCashflow']

    if type_index == "SP500":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox1")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox2")
        st.write(option_2)

    elif type_index == "Russ3000":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox3")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox4")
        st.write(option_2)

    elif type_index == "SP500_less_GDP_capita":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox5")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox6")
        st.write(option_2)

    elif type_index == "Russ3000_less_GDP_capita":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox7")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox8")
        st.write(option_2)

    elif type_index == "SP500_great_GDP_capita":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox9")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox10")
        st.write(option_2)

    elif type_index == "Russ3000_great_GDP_capita":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox11")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox12")
        st.write(option_2)

    elif type_index == "SP500_less_GDP":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox13")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox14")
        st.write(option_2)

    elif type_index == "Russ3000_less_GDP":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox15")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox16")
        st.write(option_2)

    elif type_index == "SP500_great_GDP":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox17")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox18")
        st.write(option_2)

    elif type_index == "Russ3000_great_GDP":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox19")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox20")
        st.write(option_2)

    elif type_index == "SP500_less_Population":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox21")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox22")
        st.write(option_2)

    elif type_index == "Russ3000_less_Population":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox23")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox24")
        st.write(option_2)

    elif type_index == "SP500_great_Population":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox25")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox26")
        st.write(option_2)

    elif type_index == "Russ3000_great_Population":

        option_1 = st.selectbox("Select X-axis Column",
                                options, key="selectbox27")
        st.write(option_1)

        # Remove the selected option from the list of options for the second select box
        options_2 = [option for option in options if option != option_1]

        option_2 = st.selectbox("Select Y-axis Column",
                                options_2, key="selectbox28")
        st.write(option_2)

    fig = px.scatter(df, x=option_1, y=option_2,
                     size="fullTimeEmployees", color="company_name",
                     hover_name=df["ticker"]+"\n I:"+df["industry"]+"\n S:"+df["sector"])
    fig.update_layout(height=800)
    st.plotly_chart(fig, theme=None, use_container_width=True)

# Map and Bar Polar Visualizations


def map_bar_polar(df, text_map, tab1, tab2, title_map, No, text_on_map):
    if text_map == "GDP":
        text_label = "GDP (Billions)"
    elif text_map == "GDP/Capita":
        text_label = "GDP/Capita (Thousands)"
    else:
        text_label = "Population (Millions)"
    if text_map == "GDP" or text_map == "GDP/Capita":
        tickprefix_ = "$"
    else:
        tickprefix_ = ""
    st.header(text_on_map)
    tab1_, tab2_ = st.tabs([tab1, tab2])
    with tab1_:
        fig = px.choropleth(df,
                            locations='country',
                            locationmode='country names',
                            color=text_map,
                            animation_frame='year',
                            title=title_map,
                            color_continuous_scale='OrRd')
        fig.update_layout(height=800)
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with tab2_:
        # Define the number of countries to show for each year
        N = No
        # Group the data by year and country, and sum the GDP values for each group
        df_agg = df.groupby(['year', 'country']).sum().reset_index()
        df_agg[str(text_map)+'_text'] = (df_agg[str(text_map)] /
                                         1000).round(1).astype(str) + 'B'
        # Create an empty DataFrame to store the filtered data
        df_agg_filtered = pd.DataFrame()
        # Loop through each year and select the top N countries based on their GDP
        for year in df_agg['year'].unique():
            df_agg_year = df_agg[df_agg['year'] == year]
            df_agg_top_countries = df_agg_year.sort_values(str(text_map), ascending=False)[
                'country'].head(N).tolist()
            df_agg_year_filtered = df_agg_year[df_agg_year['country'].isin(
                df_agg_top_countries)]
            df_agg_filtered = pd.concat(
                [df_agg_filtered, df_agg_year_filtered])
        # Plot the filtered data
        fig = px.bar_polar(df_agg_filtered,
                           r=str(text_map),
                           theta="country",
                           animation_frame="year",
                           animation_group="country",
                           color="country",
                           color_discrete_sequence=px.colors.qualitative.G10,
                           title="Comparison of " +
                           str(text_label) +
                           " over 1980-2028 for top {} countries".format(N),
                           labels={str(text_map): str(text_label)})
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=str(text_label),
                    angle=90,
                    tickprefix=tickprefix_,
                    ticks='outside',
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    tickfont=dict(color='black'),
                    title_font=dict(color='black')
                ),
                angularaxis=dict(
                    direction='clockwise',
                    period=1,
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    tickmode='array',
                    tickvals=list(
                        range(len(df_agg_filtered['country'].unique()))),
                    ticktext=list(df_agg_filtered['country'].unique())
                )
            ),
            showlegend=False,
            height=800
        )
        st.plotly_chart(fig, theme=None, use_container_width=True)

# Plotting Histogram for Numerical columns


def plot_histogram(df, col_name):
    series = df[col_name]
    # remove zero values items
    series = series[series != 0]
    # remove outliers for +- three standard deviations.
    series = series[~((series - series.mean()).abs() > 3 * series.std())]
    smin, smax = series.min(), series.max()
    percentiles = [np.percentile(series, n) for n in (2.5, 50, 97.5)]
    trace0 = go.Histogram(x=series,
                          histfunc='avg',
                          histnorm='probability density',
                          opacity=.75,
                          marker={'color': '#EB89B5'})
    data_ = go.Data([trace0])
    shapes = [{'line': {'color': '#0099FF', 'dash': 'solid', 'width': 2},
               'type': 'line',
               'x0': percentiles[0], 'x1':percentiles[0], 'xref':'x',
               'y0':-0.1, 'y1':1, 'yref':'paper'},
              {'line': {'color': '#00999F', 'dash': 'solid', 'width': 1},
               'type': 'line',
               'x0': percentiles[1], 'x1':percentiles[1], 'xref':'x',
               'y0':-0.1, 'y1':1, 'yref':'paper'},

              {'line': {'color': '#0099FF', 'dash': 'solid', 'width': 2},
               'type': 'line',
               'x0': percentiles[2], 'x1':percentiles[2], 'xref':'x',
               'y0':-0.1, 'y1':1, 'yref':'paper'}
              ]
    annotations = [{'x': percentiles[0], 'xref':'x', 'xanchor':'right',
                    'y': .3, 'yref':'paper',
                    'text':'2.5%', 'font':{'size': 16},
                    'showarrow': False},

                   {'x': percentiles[1], 'xref':'x', 'xanchor':'center',
                    'y': .2, 'yref':'paper',
                    'text':'95%<br>median = {0:,.2f}<br>mean = {1:,.2f}<br>min = {2:,}<br>max = {3:,}'
                    .format(percentiles[1], series.mean(), smin, smax),
                    'showarrow':False,
                    'font':{'size': 20}},
                   {'x': percentiles[2], 'xref':'x', 'xanchor':'left',
                    'y': .3, 'yref':'paper',
                    'text':'2.5%', 'font':{'size': 16},
                    'showarrow': False},
                   ]
    layout = go.Layout(title=col_name.replace('_', ' ').capitalize(),
                       titlefont={'size': 50},
                       yaxis={'title': 'Probability/Density'},
                       xaxis={'title': col_name, 'type': 'linear'},
                       shapes=shapes,
                       annotations=annotations
                       )
    figure = go.Figure(data=data_, layout=layout)
    figure.update_layout(height=800)
    st.plotly_chart(figure, theme=None, use_container_width=True)

# Plotting Histogram for Numerical Columns without outliers


def plot_histogram_without_outliers(df, col_name):
    series = df[col_name]
    # remove zero values items
    series = series[series != 0]
    smin, smax = series.min(), series.max()
    percentiles = [np.percentile(series, n) for n in (2.5, 50, 97.5)]
    trace0 = go.Histogram(x=series,
                          histfunc='avg',
                          histnorm='probability density',
                          opacity=.75,
                          marker={'color': '#EB89B5'})
    data_ = go.Data([trace0])
    shapes = [{'line': {'color': '#0099FF', 'dash': 'solid', 'width': 2},
               'type': 'line',
               'x0': percentiles[0], 'x1':percentiles[0], 'xref':'x',
               'y0':-0.1, 'y1':1, 'yref':'paper'},
              {'line': {'color': '#00999F', 'dash': 'solid', 'width': 1},
               'type': 'line',
               'x0': percentiles[1], 'x1':percentiles[1], 'xref':'x',
               'y0':-0.1, 'y1':1, 'yref':'paper'},

              {'line': {'color': '#0099FF', 'dash': 'solid', 'width': 2},
               'type': 'line',
               'x0': percentiles[2], 'x1':percentiles[2], 'xref':'x',
               'y0':-0.1, 'y1':1, 'yref':'paper'}
              ]
    annotations = [{'x': percentiles[0], 'xref':'x', 'xanchor':'right',
                    'y': .3, 'yref':'paper',
                    'text':'2.5%', 'font':{'size': 16},
                    'showarrow': False},

                   {'x': percentiles[1], 'xref':'x', 'xanchor':'center',
                    'y': .2, 'yref':'paper',
                    'text':'95%<br>median = {0:,.2f}<br>mean = {1:,.2f}<br>min = {2:,}<br>max = {3:,}'
                    .format(percentiles[1], series.mean(), smin, smax),
                    'showarrow':False,
                    'font':{'size': 20}},
                   {'x': percentiles[2], 'xref':'x', 'xanchor':'left',
                    'y': .3, 'yref':'paper',
                    'text':'2.5%', 'font':{'size': 16},
                    'showarrow': False},
                   ]

    layout = go.Layout(title=col_name.replace('_', ' ').capitalize(),
                       titlefont={'size': 50},
                       yaxis={'title': 'Probability/Density'},
                       xaxis={'title': col_name, 'type': 'linear'},
                       shapes=shapes,
                       annotations=annotations
                       )
    figure = go.Figure(data=data_, layout=layout)
    figure.update_layout(height=800)
    st.plotly_chart(figure, theme=None, use_container_width=True)

# Plotting Value Counts for Categorical Columns


def plot_value_counts(df, col_name, table=False, bar=False):
    N = 10
    df_grouped = df.groupby(col_name).apply(
        lambda x: [', '.join(list(x['company_name']))])
    df_grouped = df_grouped.to_frame('companies').reset_index()

    values_count = pd.DataFrame(df[col_name].value_counts())
    values_count.columns = ['count']
    # convert the index column into a regular column.
    values_count[col_name] = [str(i) for i in values_count.index]
    # add a column with the percentage of each data point to the sum of all data points.
    values_count['percent'] = values_count['count'].div(
        values_count['count'].sum()).multiply(100).round(2)
    # change the order of the columns.
    values_count = values_count.merge(
        df_grouped, left_on=col_name, right_on=col_name)

    values_count = values_count.reindex(
        [col_name, 'count', 'percent', 'companies'], axis=1)
    values_count.reset_index(drop=True, inplace=True)
    values_count = values_count.head(N)
    if bar:
        # add a font size for annotations0 which is relevant to the length of the data points.
        font_size = int(abs(20 - (.25 * len(values_count[col_name]))))

        trace0 = go.Bar(x=values_count[col_name], y=values_count['count'])
        data_ = go.Data([trace0])

        annotations0 = [dict(x=xi,
                             y=yi,
                             showarrow=False,
                             font={'size': font_size},
                             text="{:,}".format(yi),
                             xanchor='center',
                             yanchor='bottom')
                        for xi, yi, _, _ in values_count.values]

        annotations1 = [dict(x=xi,
                             y=yi/2,
                             showarrow=False,
                             text="{}%".format(pi),
                             xanchor='center',
                             yanchor='middle',
                             font={'color': 'yellow'})
                        for xi, yi, pi, _ in values_count.values if pi > 10]

        annotations = annotations0 + annotations1

        layout = go.Layout(title=col_name.replace('_', ' ').capitalize(),
                           titlefont={'size': 50},
                           yaxis={'title': 'count'},
                           xaxis={'type': 'category'},
                           annotations=annotations)
        figure = go.Figure(data=data_, layout=layout)
        figure.update_layout(height=800)
        st.plotly_chart(figure, theme=None, use_container_width=True)
    if table:
        table = go.Figure(data=[go.Table(
            header=dict(values=list(values_count.columns),
                        align='left'),
            cells=dict(values=[values_count[col_name], values_count['count'], values_count['percent'], values_count['companies']],
                       align='left'))
        ])
        table.update_layout(
            margin={'t': 50},
            template="plotly_dark"
        )
        table.update_traces(
            columnwidth=[1, 1, 1, 4],
            selector=dict(type='table')
        )
        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 12
        st.plotly_chart(table, theme=None, use_container_width=True)
    return values_count

# Visualizations for SP500 and Russell3000


def viz_sp500_russ3000(df, type_index):
    tab1_num, tab2_cat = st.tabs(["Descriptive Statistics on the numerical columns (Table)",
                                 "Descriptive Statistics on the categorical columns (Table)"])
    with tab1_num:
        num_cols = list(df.select_dtypes(include=np.number).columns)
        colorscale = [[0, '#1f77b4'], [.5, '#aec7e8'], [1, '#ff7f0e']]
        fig = ff.create_table(round(df[num_cols].describe().reset_index(), 2), font_colors=[
                              'white'], colorscale=colorscale)
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 12
        fig.update_layout(
            title_text='Descriptive Statistics on the numerical columns',
            margin={'t': 50},
            template="plotly_dark"
        )
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with tab2_cat:
        categorized_cols = list(df.select_dtypes(include='object').columns)
        colorscale = [[0, '#1f77b4'], [.5, '#aec7e8'], [1, '#ff7f0e']]
        fig = ff.create_table(df[categorized_cols].describe(
        ).reset_index(), font_colors=['white'], colorscale=colorscale)
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 12
        fig.update_layout(
            title_text='Statistics of categorical columns',
            margin={'t': 50},
            template="plotly_dark"
        )
        st.plotly_chart(fig, theme=None, use_container_width=True)

    st.header("Numerical Analysis")

    tab1_num_analysis, tab2_num_analysis_without_outliers = st.tabs(
        ["Numerical Analysis (Histogram)", "Numerical Analysis without outliers (Histogram)"])

    with tab1_num_analysis:
        for col in num_cols:
            plot_histogram(df, col)

    with tab2_num_analysis_without_outliers:
        for col in num_cols:
            plot_histogram_without_outliers(df, col)

    categorized_cols = [col for col in categorized_cols if col not in [
        'ticker', 'company_name', 'ceo_name', 'longBusinessSummary']]
    df_cat = []

    st.header("Categorical Analysis")

    for col in categorized_cols:
        values_count = plot_value_counts(df, col, 1, 1)
        df_cat.append(values_count)

    st.header("Bivariate Analysis")

    fig = px.scatter_matrix(
        df,
        dimensions=['median_worker_pay', 'pay_ratio', 'salary', 'totalRevenue',
                    'grossProfits', 'freeCashflow', 'operatingCashflow', "fullTimeEmployees"],
        color="sector",
        hover_name=df["ceo_name"] + " - " + df["company_name"],
        title=str(type_index)+" Variables Scatter Plot Matrix",
    )
    fig.update_layout(height=1800)
    st.plotly_chart(fig, theme=None, use_container_width=True)

    scatter_plot_options_sp_rus(df, type_index)


st.set_page_config(
    page_title="GDP, GDP/Capita, Population CEO Worker Pay", layout="wide")

melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop = gdp_gdp_capita_population()

st.header("Analysis on 1) GDP, GDP/Capita, Population and 2) SP500/Russell3000")

st.write("No of countries in the Dataset from IMF Data ",
         melted_imf_data_gdp["country"].nunique())

top_n = st.slider('Select top countries', 10, 25, 10)
st.write("Selected ", top_n, " countries")

tab_eda, tab_filter_analysis = st.tabs(
    ["Exploratory Data Analysis of GDP, GDP/Capita, Population and SP500/Russell3000", "Filter Analysis"])

with tab_eda:
    text_map = "GDP"
    text_on_map = "GDP"
    tab1 = "GDP over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of GDP over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "GDP (Billions of U.S. dollars) by Country (1980-2028)"
    map_bar_polar(melted_imf_data_gdp, text_map, tab1,
                  tab2, title_map, top_n, text_on_map)

    text_map = "GDP/Capita"
    text_on_map = "GDP/Capita"
    tab1 = "GDP/Capita over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of GDP/Capita over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "GDP/Capita (Thousands of U.S. dollars) by Country (1980-2028)"
    map_bar_polar(melted_imf_data_gdp_capita, text_map,
                  tab1, tab2, title_map, top_n, text_on_map)

    text_map = "Population"
    text_on_map = "Population"
    tab1 = "Population over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of Population over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "Population (Millions of people) by Country (1980-2028)"
    map_bar_polar(melted_imf_data_pop, text_map, tab1,
                  tab2, title_map, top_n, text_on_map)

    st.divider()

    # SP500
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    st.header("Data from SP500 and Russell3000")

    tab1_sp500, tab2_rus3000 = st.tabs(["SP500", "Russell 3000"])

    with tab1_sp500:
        st.header("SP500")
        sdf = sdf_i()

        viz_sp500_russ3000(sdf, "SP500")
    with tab2_rus3000:
        st.header("Russell3000")
        rdf = rdf_i()
        viz_sp500_russ3000(rdf, "Russ3000")

with tab_filter_analysis:
    # Median Worker Pay and GDP/Capita

    med_pay = st.slider('Median Pay Value', 7402, 146146, 7402)
    st.write("Median Pay Selected $", med_pay)

    melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop = gdp_gdp_capita_population()

    melted_imf_data_gdp_capita = melted_imf_data_gdp_capita[
        melted_imf_data_gdp_capita["GDP/Capita"] <= med_pay]
    text_map = "GDP/Capita"
    text_on_map = "GDP/Capita less than Median Pay $"+str(med_pay)
    tab1 = "GDP/Capita over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of GDP/Capita over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "GDP/Capita (Thousands of U.S. dollars) by Country (1980-2028)"
    map_bar_polar(melted_imf_data_gdp_capita, text_map,
                  tab1, tab2, title_map, top_n, text_on_map)

    tab1_sp500_gdp_capita, tab2_rus3000_gdp_capita = st.tabs(
        ["MedianPay less than $"+str(med_pay)+" for SP500", "MedianPay less than $"+str(med_pay)+" for Rus3000"])
    with tab1_sp500_gdp_capita:
        sdf = sdf_i()
        sdf = sdf[sdf["median_worker_pay"] <= med_pay]
        scatter_plot_options_sp_rus(sdf, "SP500_less_GDP_capita")
    with tab2_rus3000_gdp_capita:
        rdf = rdf_i()
        rdf = rdf[rdf["median_worker_pay"] <= med_pay]
        scatter_plot_options_sp_rus(rdf, "Russ3000_less_GDP_capita")

    melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop = gdp_gdp_capita_population()

    melted_imf_data_gdp_capita = melted_imf_data_gdp_capita[~(
        melted_imf_data_gdp_capita["GDP/Capita"] <= med_pay)]
    text_map = "GDP/Capita"
    text_on_map = "GDP/Capita greater than Median Pay $"+str(med_pay)
    tab1 = "GDP/Capita over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of GDP/Capita over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "GDP/Capita (Thousands of U.S. dollars) by Country (1980-2028)"
    map_bar_polar(melted_imf_data_gdp_capita, text_map,
                  tab1, tab2, title_map, top_n, text_on_map)

    tab1_sp500_gdp_capita, tab2_rus3000_gdp_capita = st.tabs(
        ["MedianPay greater than $"+str(med_pay)+" for SP500", "MedianPay greater than $"+str(med_pay)+" for Russ3000"])
    with tab1_sp500_gdp_capita:
        sdf = sdf_i()
        sdf = sdf[~(sdf["median_worker_pay"] <= med_pay)]
        scatter_plot_options_sp_rus(sdf, "SP500_great_GDP_capita")
    with tab2_rus3000_gdp_capita:
        rdf = rdf_i()
        rdf = rdf[~(rdf["median_worker_pay"] <= med_pay)]
        scatter_plot_options_sp_rus(rdf, "Russ3000_great_GDP_capita")

    # Gross Profits and GDP

    gross_profits = st.slider('Gross Profits Value $(Billions) ', 1, 225, 1)
    st.write("Gross Profit Selected $(Billions)", gross_profits)

    st.warning(
        'Gross Profits in Billions other columns in their inherent values', icon="⚠️")

    melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop = gdp_gdp_capita_population()

    melted_imf_data_gdp = melted_imf_data_gdp[melted_imf_data_gdp["GDP"]
                                              <= gross_profits]
    text_map = "GDP"
    text_on_map = "GDP less than Gross Profits $"+str(gross_profits)+"B"
    tab1 = "GDP over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of GDP over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "GDP (Billions of U.S. dollars) by Country (1980-2028)"
    map_bar_polar(melted_imf_data_gdp, text_map, tab1,
                  tab2, title_map, top_n, text_on_map)

    tab1_sp500_gdp, tab2_rus3000_gdp = st.tabs(["Gross Profits less than $"+str(
        gross_profits)+"B for SP500", "Gross Profits less than $"+str(gross_profits)+"B for Russ3000"])
    with tab1_sp500_gdp:
        sdf = sdf_i()
        sdf["grossProfits"] = sdf["grossProfits"]/1000000000
        sdf = sdf[sdf["grossProfits"] <= gross_profits]
        scatter_plot_options_sp_rus(sdf, "SP500_less_GDP")
    with tab2_rus3000_gdp:
        rdf = rdf_i()
        rdf["grossProfits"] = rdf["grossProfits"]/1000000000
        rdf = rdf[rdf["grossProfits"] <= gross_profits]
        scatter_plot_options_sp_rus(rdf, "Russ3000_less_GDP")

    melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop = gdp_gdp_capita_population()

    melted_imf_data_gdp = melted_imf_data_gdp[~(
        melted_imf_data_gdp["GDP"] <= gross_profits)]
    text_map = "GDP"
    text_on_map = "GDP greater than Gross Profits $"+str(gross_profits)+"B"
    tab1 = "GDP over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of GDP over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "GDP (Billions of U.S. dollars) by Country (1980-2028)"
    map_bar_polar(melted_imf_data_gdp, text_map, tab1,
                  tab2, title_map, top_n, text_on_map)

    tab1_sp500_gdp, tab2_rus3000_gdp = st.tabs(["Gross Profits greater than $"+str(
        gross_profits)+"B for SP500", "Gross Profits greater than $"+str(gross_profits)+"B for Russ3000"])
    with tab1_sp500_gdp:
        sdf = sdf_i()
        sdf["grossProfits"] = sdf["grossProfits"]/1000000000
        sdf = sdf[~(sdf["grossProfits"] <= gross_profits)]
        scatter_plot_options_sp_rus(sdf, "SP500_great_GDP")
    with tab2_rus3000_gdp:
        rdf = rdf_i()
        rdf["grossProfits"] = rdf["grossProfits"]/1000000000
        rdf = rdf[~(rdf["grossProfits"] <= gross_profits)]
        scatter_plot_options_sp_rus(rdf, "Russ3000_great_GDP")

    # Full Time Employess Profits and Population

    full_time_employees = st.slider(
        'Full time Employees (Millions) ', 0.0125, 2.1, 1.0)
    st.write("Full Time Employees Selected (Millions)", full_time_employees)

    st.warning(
        'Full Time Employees in Millions other columns in their inherent values', icon="⚠️")

    melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop = gdp_gdp_capita_population()

    melted_imf_data_pop = melted_imf_data_pop[melted_imf_data_pop["Population"]
                                              <= full_time_employees]
    text_map = "Population"
    text_on_map = "Population less than Full Time Employees: " + \
        str(full_time_employees)+"M people"
    tab1 = "Population over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of Population over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "Population of by Country (1980-2028)"
    map_bar_polar(melted_imf_data_pop, text_map, tab1,
                  tab2, title_map, top_n, text_on_map)

    tab1_sp500_pop, tab2_rus3000_pop = st.tabs(["Population less than "+str(full_time_employees) +
                                               "M people for SP500", "Population less than "+str(full_time_employees)+"M people for Russ3000"])
    with tab1_sp500_pop:
        sdf = sdf_i()
        sdf["fullTimeEmployees"] = sdf["fullTimeEmployees"]/1000000
        sdf = sdf[sdf["fullTimeEmployees"] <= full_time_employees]
        scatter_plot_options_sp_rus(sdf, "SP500_less_Population")
    with tab2_rus3000_pop:
        rdf = rdf_i()
        rdf["fullTimeEmployees"] = rdf["fullTimeEmployees"]/1000000
        rdf = rdf[rdf["fullTimeEmployees"] <= full_time_employees]
        scatter_plot_options_sp_rus(rdf, "Russ3000_less_Population")

    melted_imf_data_gdp, melted_imf_data_gdp_capita, melted_imf_data_pop = gdp_gdp_capita_population()

    melted_imf_data_pop = melted_imf_data_pop[~(
        melted_imf_data_pop["Population"] <= full_time_employees)]
    text_map = "Population"
    text_on_map = "Population greater than Full Time Employees: " + \
        str(full_time_employees)+" M people"
    tab1 = "Population over 1980-2028 (Choropleth Map)"
    tab2 = "Comparison of Population over 1980-2028 for top " + \
        str(top_n)+" countries (Bar Polar)"
    title_map = "Population of by Country (1980-2028)"
    map_bar_polar(melted_imf_data_pop, text_map, tab1,
                  tab2, title_map, top_n, text_on_map)

    tab1_sp500_pop, tab2_rus3000_pop = st.tabs(["Population greater than "+str(full_time_employees) +
                                               "M people for SP500", "Population greater than "+str(full_time_employees)+"M people for Russ3000"])
    with tab1_sp500_pop:
        sdf = sdf_i()
        sdf["fullTimeEmployees"] = sdf["fullTimeEmployees"]/1000000
        sdf = sdf[~(sdf["fullTimeEmployees"] <= full_time_employees)]
        scatter_plot_options_sp_rus(sdf, "SP500_great_Population")
    with tab2_rus3000_pop:
        rdf = rdf_i()
        rdf["fullTimeEmployees"] = rdf["fullTimeEmployees"]/1000000
        rdf = rdf[~(rdf["fullTimeEmployees"] <= full_time_employees)]
        scatter_plot_options_sp_rus(rdf, "Russ3000_great_Population")
