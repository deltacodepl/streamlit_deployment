import os
import pandas as pd
import streamlit as st
import altair as alt
from matplotlib import pyplot as plt
import numpy as np
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/src/service_account.json"

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Filter,
    FilterExpression,
    Dimension,
    Metric,
    RunReportRequest,
    OrderBy,
)


st.set_page_config(
    page_title="Google Analytics Data Dashboard",
    page_icon="🚀",
    layout="wide",  
    initial_sidebar_state="auto", 
)

# Set up Streamlit app title
st.title("📈 Data Dashboard")
st.divider()
st.sidebar.header("SEO report date range👇")

FILENAME = '/src/ua.csv'
property_id = "348254593"
with st.spinner('Loading Data from Google, Please Wait...'):
    time.sleep(7)
st.markdown("<h2><u>Form Submissions</u><h2>", unsafe_allow_html=True)

# Date range input for the first date frame
start_date_1 = st.sidebar.date_input("Start date of current month", pd.to_datetime("2023-01-01"), disabled=True)
end_date_1 = st.sidebar.date_input("End date of current month", pd.to_datetime("today"), disabled=True)

# Date range input for the second date frame
#start_date_2 = st.sidebar.date_input("Start date of month to compare", pd.to_datetime("2024-01-18"))
#end_date_2 = st.sidebar.date_input("End date of month to compare", pd.to_datetime("today"))
start_date_2 = pd.to_datetime("2023-01-01")
end_date_2 =  pd.to_datetime("2023-12-01")

placeholder = st.container()
placeholder.empty()

# ---
df_fs = None
df_fs = pd.read_csv(FILENAME)

df_fs['created'] = pd.to_datetime(df_fs['created'])
#define how to aggregate various fields
agg_functions = {'created': 'first'}

df_fs['count'] = df_fs.groupby([df_fs['created'].dt.year, df_fs['created'].dt.month])['created'].transform('count')

df1 = df_fs.groupby(df_fs['created'].dt.month).size().reset_index(name='Conversions')

#df['created'].dt.month.value_counts()
# df['created'].dt.month

res = df_fs.groupby(df_fs['created'].dt.month).size()
#print(res)
res = df_fs.groupby(df_fs['created'].dt.month)['created'].count()
#print(res)
res = df_fs.groupby(df_fs['created'].dt.month).value_counts()
#print(res)
#df.groupby(df['created'].dt.month).agg({'count'})


# df = pd.DataFrame({'employee_id': [1, 1, 2, 3, 3, 3],
#                     'employee_name': ['Carlos', 'Carlos', 'Dan', 'Samuel', 'Samuel', 'Samuel'],
#                     'sales': [4, 1, 3, 2, 5, 3],})

#create new DataFrame by combining rows with same id values as_index = True
df_new = pd.DataFrame()
df_new = df_fs.groupby([df_fs['created'].dt.year, df_fs['created'].dt.month]).aggregate("first")
df_new = df_new.rename(columns={'created': 'Date', 'count': 'Conversions'})
df_new.index.rename(['Year','Month'],inplace=True)

#st.write(df_new)
# count sum of state in each month
#df.groupby(df.created.dt.month)['state'].sum()

#ct = df.groupby('created').size().values

#df
# df = df.drop_duplicates(subset="created").assign(Count=ct)

# cnt = df.groupby(pd.Grouper(key='created', axis=0, freq='M')).size().rename('Count')

# result = df.drop_duplicates(subset='created').merge(cnt, left_on='created', right_index=True)
# result
# @title Conversions
time.sleep(1)
with placeholder:
   st.pyplot(df_new['Conversions'].plot(kind='line', figsize=(6, 4), title='Form submissions').figure, use_container_width=False)
placeholder.empty()
#st.pyplot(plt.gca().spines[['top', 'right']].set_visible(False))

# ---
# Run report request for the first date frame
client = BetaAnalyticsDataClient()
request_1 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="sessionDefaultChannelGroup")],
    metrics=[
        Metric(name="activeUsers"),
        Metric(name="newUsers"),
        Metric(name="engagedSessions"),
    ],
    date_ranges=[DateRange(start_date=start_date_1.strftime("%Y-%m-%d"), end_date=end_date_1.strftime("%Y-%m-%d"))],
    dimension_filter=FilterExpression(
        filter=Filter(
            field_name="sessionDefaultChannelGroup",
            string_filter={"value": "Organic Search"},
        )
    ),
)

response_1 = client.run_report(request_1)

# Run report request for the second date frame
request_2 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="sessionDefaultChannelGroup")],
    metrics=[
        Metric(name="activeUsers"),
        Metric(name="newUsers"),
        Metric(name="engagedSessions"),
    ],
    date_ranges=[DateRange(start_date=start_date_2.strftime("%Y-%m-%d"), end_date=end_date_2.strftime("%Y-%m-%d"))],
    dimension_filter=FilterExpression(
        filter=Filter(
            field_name="sessionDefaultChannelGroup",
            string_filter={"value": "Organic Search"},
        )
    ),
)

response_2 = client.run_report(request_2)

# Combine data into a single list
combined_data = []

for row in response_1.rows:
    combined_data.append({
        'Date Range': f"{start_date_1.strftime('%B %Y')}",
        'Channel': row.dimension_values[0].value,
        'Active Users': row.metric_values[0].value,
        'New Users': row.metric_values[1].value,
        'Engaged Sessions': row.metric_values[2].value
    })

for row in response_2.rows:
    combined_data.append({
        'Date Range': f"{start_date_2.strftime('%B %Y')}",
        'Channel': row.dimension_values[0].value,
        'Active Users': row.metric_values[0].value,
        'New Users': row.metric_values[1].value,
        'Engaged Sessions': row.metric_values[2].value
    })


# Create a single DataFrame
df_combined = pd.DataFrame(combined_data)



# st.title("Simulation[tm]")
# st.write("Here is our super important simulation")

# x = st.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
# y = st.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)

# st.write(f"x={x} y={y}")
# values = np.cumprod(1 + np.random.normal(x, y, (100, 10)), axis=0)

# for i in range(values.shape[1]):
#     plt.plot(values[:, i])

# st.pyplot()

# from matplotlib.dates import ConciseDateFormatter

# fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
# dates = np.arange(np.datetime64('2023-06-01'), np.datetime64('2024-06-24'),
#                   np.timedelta64(1, 'h'))
# # data = np.cumsum(np.random.randn(len(dates)))
# st.write(df_new['Conversions'])
# ax.plot([(2023,1), (2023,2), (2023,3), (2023,4)], [6,7,2,8])

# #ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))

# st.pyplot(fig)



# Display Combined DataFrame in Streamlit
#st.subheader("Month on Month Data")
#st.dataframe(df_combined)

# Bar Chart for Active Users, New Users, and Engaged Sessions
chart = alt.Chart(df_combined).mark_bar().encode(
    x=alt.X('Date Range:N', title='Date Range'),
    y=alt.Y('Active Users:Q', title='Count'),
    color=alt.Color('Channel:N', title='Channel'),
    column=alt.Column('Metric:N', title='Metrics')
).transform_fold(
    fold=['Active Users', 'New Users', 'Engaged Sessions'],
    as_=['Metric', 'Count']
)

# Display the chart in Streamlit
#st.subheader("Active users MoM")
#st.altair_chart(chart, use_container_width=True)

# Run report request for the top 10 landing pages
request_landing_pages = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[
        Dimension(name="landingPage"),
    ],
    metrics=[
        Metric(name="activeUsers"),
        Metric(name="newUsers"),
        Metric(name="engagedSessions"),
    ],
    date_ranges=[
        DateRange(start_date=start_date_1.strftime("%Y-%m-%d"), end_date=end_date_1.strftime("%Y-%m-%d"))
    ],
)

response_landing_pages = client.run_report(request_landing_pages)

# Extract top 10 landing pages
top_landing_pages_data = []
for row in response_landing_pages.rows[:10]:
    top_landing_pages_data.append({
        'Landing Page': row.dimension_values[0].value,
        'Active Users': row.metric_values[0].value,
        'New Users': row.metric_values[1].value,
        'Engaged Sessions': row.metric_values[2].value
    })

# Create DataFrame for top 10 landing pages
df_top_landing_pages = pd.DataFrame(top_landing_pages_data)

# Display DataFrame for top 10 landing pages in Streamlit
st.subheader("Top 10 Landing Pages")
st.dataframe(df_top_landing_pages)


st.markdown("<h2><u>Search Console Data</u><h2>", unsafe_allow_html=True)
#Run report request for the first GSC date frame
client = BetaAnalyticsDataClient()
request_1 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="yearMonth")],
    metrics=[
        Metric(name="organicGoogleSearchClicks"),
        Metric(name="organicGoogleSearchImpressions"),
        Metric(name="organicGoogleSearchClickThroughRate"),
    ],
    date_ranges=[DateRange(start_date=start_date_1.strftime("%Y-%m-%d"), end_date=end_date_1.strftime("%Y-%m-%d"))],
)

response_1 = client.run_report(request_1)

# Run report request for the second date frame
request_2 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="yearMonth")],
    metrics=[
        Metric(name="organicGoogleSearchClicks"),
        Metric(name="organicGoogleSearchImpressions"),
        Metric(name="organicGoogleSearchClickThroughRate"),
    ],
    date_ranges=[DateRange(start_date=start_date_2.strftime("%Y-%m-%d"), end_date=end_date_2.strftime("%Y-%m-%d"))],
)

response_2 = client.run_report(request_2)

# st.write(response_1)
# st.write(response_2)

# Combine GSC data into a single list
combined_GSC_data = []

for row in response_1.rows:
    combined_GSC_data.append({
        'Month': row.dimension_values[0].value,
        'Clicks': row.metric_values[0].value,
        'Impressions': row.metric_values[1].value,
        'CTR': row.metric_values[2].value
    })

# for row in response_2.rows:
#     combined_GSC_data.append({
#         'Month': row.dimension_values[0].value,
#         'Clicks': row.metric_values[0].value,
#         'Impressions': row.metric_values[1].value,
#         'CTR': row.metric_values[2].value
#     })

# Create a single DataFrame
df_GSC_combined = pd.DataFrame(combined_GSC_data)
#st.subheader("Month on Month Data")
st.dataframe(df_GSC_combined)

#GSC Charts
st.subheader("SEO Clicks Over Months")
gsc_chart = alt.Chart(df_GSC_combined).mark_bar().encode(
    x='Month:N',
    y='Clicks:Q',  
    color='Month:N'
).properties(
    width=600,
    height=400
)

st.altair_chart(gsc_chart, use_container_width=True)

# st.subheader("Impressions Over Months")
# gsc_chart = alt.Chart(df_GSC_combined).mark_bar().encode(
#     x='Month:N',
#     y='Impressions:Q',  
#     color='Month:N'
# ).properties(
#     width=600,
#     height=400
# )

# st.altair_chart(gsc_chart, use_container_width=True)


# Set credentials and define the right GA4 property

#credentials = service_account.Credentials.from_service_account_file('./drive/MyDrive/GA4-DATA/gareports-414112-289c0e433127.json')
#client = BetaAnalyticsDataClient(credentials=credentials)
default_metrics=[Metric(name="conversions:GA4_click_mail"), Metric(name='conversions:GA4_copy_mail'), Metric(name='conversions:GA4_click_tel'),
             Metric(name="conversions:GA4_copy_tel"), Metric(name="conversions:contact_form")]
estonia_metrics=[Metric(name="conversions:GA4_real_email"), Metric(name='conversions:GA4_real_copy_email'), Metric(name='conversions:GA4_real_call'),
             Metric(name="conversions:GA4_real_copy_tel")]
za_metrics=[Metric(name="conversions:GA4_real_email"), Metric(name='conversions:GA4_real_copy_email'), Metric(name='conversions:GA4_real_call'),
             Metric(name="conversions:GA4_real_copy_tel")]

country = 'PL (Poland)'
# Get the data from the API
from datetime import datetime
from dateutil.relativedelta import relativedelta

from dateutil import rrule, parser
date1 = '2023-01-01'
date2 = '2023-12-31'
date3 = '2024-01-01'
date4 = '2024-06-12' #datetime.now().date().strftime

#list(rrule.rrule(rrule.MONTHLY,count=10,dtstart=parser.parse(date1)))

datesx = list(rrule.rrule(rrule.MONTHLY, dtstart=parser.parse(date1), until=parser.parse(date2)))

datesy = list(rrule.rrule(rrule.MONTHLY, dtstart=parser.parse(date3), until=parser.parse(date4)))

drange = []
drange2 = []

for m in datesx:
  res = m + relativedelta(day=31)
  drange.append((m, res))

dranges = [DateRange(start_date=m[0].strftime('%Y-%m-%d'), end_date=m[1].strftime('%Y-%m-%d')) for m in drange ]

for m in datesy:
  res = m + relativedelta(day=31)
  drange2.append((m, res))

# Get the data from the API

dranges2 = [DateRange(start_date=m[0].strftime('%Y-%m-%d'), end_date=m[1].strftime('%Y-%m-%d')) for m in drange2 ]

ga_data = []
ga_data2 = []
month_start = 0

# ok 12 
# st.write(dranges)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[0]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )
response = client.run_report(request)
if response.row_count < 1:
   month_start = month_start + 1
else:
   ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[1]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
if response.row_count < 1:
   month_start = month_start + 1
else:
   ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[2]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
if response.row_count < 1:
   month_start = month_start + 1
else:
   ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[3]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
if response.row_count < 1:
   month_start = month_start + 1
else:
   ga_data.append(response)


# request = RunReportRequest(
#   property=f"properties/{property_id}",
#   #dimensions=[Dimension(name="yearMonth")],
#   dimensions=[], #Dimension(name='country')
#   metrics=default_metrics,
#   date_ranges=dranges[4:8],
#   #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
#   keep_empty_rows=False,
#   )
# response = client.run_report(request)
# # print(response)
# ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[4]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[5]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[6]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[7]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )
response = client.run_report(request)
ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[8]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
ga_data.append(response)

# request = RunReportRequest(
#   property=f"properties/{property_id}",
#   #dimensions=[Dimension(name="yearMonth")],
#   dimensions=[], #Dimension(name='country')
#   metrics=default_metrics,
#   date_ranges=dranges[8:],
#   #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
#   keep_empty_rows=False,
#   )
# response = client.run_report(request)
# # print(response)
# ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[9]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
ga_data.append(response)
request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[10]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
ga_data.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  #dimensions=[], #Dimension(name='country')
  metrics=default_metrics,
  date_ranges=[dranges[11]],
  #order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name='yearMonth'), desc=False)],
  keep_empty_rows=False,
  )

response = client.run_report(request)
ga_data.append(response)
# st.write(ga_data)





# 2024
month_start2 = 0
request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  metrics=default_metrics,
  date_ranges=[dranges2[0]],
  )
response = client.run_report(request)
if response.row_count < 1:
   month_start2 = month_start2 + 1
else:
   ga_data2.append(response)
# print(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  metrics=default_metrics,
  date_ranges=[dranges2[1]],
  )
response = client.run_report(request)
# print(response)
if response.row_count < 1:
   month_start2 = month_start2 + 1
else:
   ga_data2.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  metrics=default_metrics,
  date_ranges=[dranges2[2]],
  )
response = client.run_report(request)
# print(response)
if response.row_count < 1:
   month_start2 = month_start2 + 1
else:
   ga_data2.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  metrics=default_metrics,
  date_ranges=[dranges2[3]],
  )
response = client.run_report(request)
# print(response)
if response.row_count < 1:
   month_start2 = month_start2 + 1
else:
   ga_data2.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  metrics=default_metrics,
  date_ranges=[dranges2[4]],
  )
response = client.run_report(request)
# print(response)
if response.row_count < 1:
   month_start2 = month_start2 + 1
else:
   ga_data2.append(response)

request = RunReportRequest(
  property=f"properties/{property_id}",
  dimensions=[Dimension(name="yearMonth")],
  metrics=default_metrics,
  date_ranges=[dranges2[5]],
  )
response = client.run_report(request)
# print(response)
if response.row_count < 1:
   month_start2 = month_start2 + 1
else:
   ga_data2.append(response)

# st.write(ga_data2)
# Turn the raw data into a Table

def ga4_result_to_df(response):
    """Note"""
    result_dict = {}
    for dimensionHeader in response.dimension_headers:
        result_dict[dimensionHeader.name] = []
    for metricHeader in response.metric_headers:
        result_dict[metricHeader.name[12:]] = []
    for rowIdx, row in enumerate(response.rows):
        for i, dimension_value in enumerate(row.dimension_values):
            dimension_name = response.dimension_headers[i].name
            result_dict[dimension_name].append(dimension_value.value)
        for i, metric_value in enumerate(row.metric_values):
            metric_name = response.metric_headers[i].name[12:]
            result_dict[metric_name].append(metric_value.value)
    return pd.DataFrame(result_dict)

# df = ga4_result_to_df(response)

# df
df = pd.DataFrame({})

for i, response in enumerate(ga_data):
  df = pd.concat([df, ga4_result_to_df(response)], ignore_index=True)

df24 = pd.DataFrame({})
for i, response in enumerate(ga_data2):

  df24 = pd.concat([df24, ga4_result_to_df(response)], ignore_index=True)

import calendar
months_list = [calendar.month_name[i] for i in range(1, 13)]

# print(months_list.reverse())

#df.insert(0, 'Month', months_list[:10])

df.drop(df.columns[[0]], axis=1, inplace=True)
df.drop(df.columns[[0]], axis=1, inplace=True)

# df = df[df.values.sum(axis=1) != 0]

df24.drop(df24.columns[[0]], axis=1, inplace=True)

# reverse rows
# df_reversed = df.iloc[::-1,:]
# df24_rev = df24.iloc[::-1,:]

df_reversed = df
df24_rev = df24
# df_reversed.style.format({
#     "A": "{:.2f}",
#     "B": "{:,.5f}",
#     "C": "{:.1f}",
#     "D": "$ {:,.2f}"
# })

#df_reversed.style.clear()

df_reversed.reset_index(drop=True, inplace=True)
df24_rev.reset_index(drop=True, inplace=True)
# print(df.index.astype('int'))
# df['Sum']=df.iloc[:,1:5].sum(axis=1)

# st.write(df_reversed)
# st.write(df24_rev)


ct = pd.CategoricalIndex(months_list[month_start:13], ordered=True, name='Month')
df_reversed.set_index(ct,drop=True, inplace=True)
ct = pd.CategoricalIndex(months_list[0:6], ordered=True, name='Month')
df24_rev.set_index(ct,drop=True, inplace=True)
# del df_reversed['dateRange']
#ct
#df_reversed['GA4_click_mail'].to_numpy(dtype='int')
# df_reversed
#df_reversed.columns
df24_rev = df24_rev.astype(float)
df24_rev['Sum'] = df24_rev.sum(axis=1)


df_reversed = df_reversed.astype(float)
df_reversed['Sum'] = df_reversed.sum(axis=1)

# df_reversed = df_reversed.loc[(df!=float(0)).any(axis=1)]

#st.write(df_reversed)
#st.write(df24_rev)
#df_reversed
# df24_rev.groupby(df24_rev['GA4_click_mail'].dt.year).sum()

import cycler
from IPython import display
import matplotlib as mpl
from matplotlib import font_manager as fm
from pandas import plotting
import matplotlib_inline.backend_inline

# Register the fonts we downloaded above.
#for font_file in fm.findSystemFonts('.'):
#    fm.fontManager.addfont(font_file)

def set_default_mpl_styles(retina: bool = True) -> None:
  _FIGURE_DPI = 228
  _PLOT_DEFAULT_CMAP = 'Set2'
  _PLOT_TEXT_COLOR = '#444444'
  _PLOT_TICK_COLOR = '#666666'

  # Register Pandas formatters and converters with Matplotlib to help with
  # plotting timeseries data stored in a dataframe.
  plotting.register_matplotlib_converters()

  retina = True

  # Set display to retina.
  if retina:
    dpi = _FIGURE_DPI // 2
    #display.set_matplotlib_formats('retina')
    matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
  else:
    dpi = _FIGURE_DPI

  # Create our new styles dictionary for use in rcParams.
  mpl_styles = {
    'axes.axisbelow': True,
    'axes.edgecolor': _PLOT_TICK_COLOR,
    'axes.grid': True,
    'axes.labelcolor': _PLOT_TEXT_COLOR,
    # Set default color palette.
    'axes.prop_cycle': cycler.cycler(
        'color', mpl.colormaps.get_cmap(_PLOT_DEFAULT_CMAP).colors),
    'axes.spines.right': False,
    'axes.spines.top': False,
    #'font.family': ['Roboto', 'Open Sans'],
    'font.family': ['Liberation Mono'],
    # You can scale all other text by scaling this.
    'font.size': 6.0,
    'figure.figsize': (6, 4),
    # Up the default resolution for figures.
    'figure.dpi': dpi,
    'savefig.dpi': dpi,
    'lines.linewidth': 1.25,
    'grid.color': '#EEEEEE',
    'grid.linewidth': 0.4,
    'text.color': _PLOT_TEXT_COLOR,
    'xtick.color': _PLOT_TICK_COLOR,
    'ytick.color': _PLOT_TICK_COLOR,
    'legend.borderaxespad': 1.0,
    'legend.borderpad': 0.8,
    'legend.edgecolor': '1.0',
    'legend.facecolor': '#F3F3F3',
    'legend.fontsize': 'small',
    'legend.framealpha': 0.75,
    'legend.labelspacing': 0.6,
  }

  mpl.rcParams.update(mpl_styles)

set_default_mpl_styles()
#fm.fontManager.ttflist

# -- Conversions plot
#from matplotlib import pyplot as plt
# import seaborn as sns
#import pandas as pd
#import numpy as np

# plt.rcParams['figure.figsize'] = [14, 8]
fig, ax = plt.subplots()


#ax.bar(months_list[:10], df_reversed['GA4_click_mail'].to_numpy(dtype='int'), label='click_mail')
#ax.bar(df_reversed.index, df_reversed['GA4_copy_mail'], label='copy_email')
# ax.bar(df_reversed.index, df_reversed['GA4_click_tel'], label='click_tel')
# ax.bar(df_reversed.index, df_reversed['GA4_copy_tel'], label='cpoy_tel')
#ax.bar(months_list[:10], df_reversed['GA4_copy_mail'].to_numpy(dtype='int'), bottom=df_reversed['GA4_click_mail'].to_numpy(dtype='int'), label='copy_mail')
#ax.set_title('Conversions')
#_ = ax.legend()

#fig, ax = plt.subplots()

# # Initialize the bottom at zero for the first set of bars.
bottom = np.zeros(len(df_reversed))
#st.write(df_reversed.columns)

#---
# cols = df1.columns.union(df2.columns)

# df1 = df1.reindex(cols, axis=1, fill_value=0)
# df2 = df2.reindex(cols, axis=1, fill_value=0)
#---


# Plot each layer of the bar, adding each bar to the "bottom" so
# the next bar starts higher.
for i, col in enumerate(df_reversed.columns[:-1]):
  ax.bar(months_list[months_list.index(df_reversed.first_valid_index()):13], df_reversed[col].to_numpy(dtype='int'), bottom=bottom, label=col)
  bottom += df_reversed[col].to_numpy(dtype='int')

# totals = df_reversed.sum(axis=1)
# y_offset = 4
# for i, total in enumerate(totals):
#   ax.text(totals.index[i], total + y_offset, round(total), ha='center',
#           weight='bold')

ax.set_title('Conversions 2023')
_ = ax.legend()

st.pyplot(fig, use_container_width=True)

fig, ax = plt.subplots()
bottom = np.zeros(len(df24_rev))

#print(len(df24_rev))
#print(len(df_reversed))
#for i in range(12 - len(df24_rev)):
#  df24_rev.
# Plot each layer of the bar, adding each bar to the "bottom" so
# the next bar starts higher.
for i, col in enumerate(df24_rev.columns[:-1]):
  ax.bar(months_list[months_list.index(df24_rev.first_valid_index()):6], df24_rev[col].to_numpy(dtype='int'), bottom=bottom, label=col)
  bottom += df24_rev[col].to_numpy(dtype='int')

ax.set_title('Conversions 2024')
_ = ax.legend()

st.pyplot(fig, use_container_width=True)