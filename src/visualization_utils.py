import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates


def plot_wind_category_vs_temp(X_data, y_data, year=None, figsize=(10, 6)):
    """
    Vẽ biểu đồ so sánh Wind Category (categorical) và Temperature Next 5 Days
    """
    
    # Lọc dữ liệu theo năm nếu có chỉ định
    if year is not None:
        mask_year = y_data.index.year == year
        y_data_plot = y_data[mask_year]
        X_data_plot = X_data[mask_year]
        title_year = f" - Year {year}"
    else:
        y_data_plot = y_data
        X_data_plot = X_data
        title_year = ""
    
    # Kiểm tra xem có dữ liệu không
    if len(y_data_plot) == 0:
        print(f"⚠️ Không có dữ liệu cho năm {year}")
        return None
    
    # Kiểm tra xem có feature wind_category không
    if 'wind_category' not in X_data_plot.columns:
        print("⚠️ Không tìm thấy feature 'wind_category'")
        return None
    
    print(f"📊 Số lượng mẫu: {len(y_data_plot)}")
    print(f"💨 Các Wind Category: {sorted(X_data_plot['wind_category'].unique())}")
    print(f"🌡️ Temperature range: {y_data_plot['temp_next_5'].min():.2f} - {y_data_plot['temp_next_5'].max():.2f}")
    
    # Tạo figure
    plt.figure(figsize=figsize)
    
    # Dữ liệu cho plotting
    boxplot_data = pd.DataFrame({
        'wind_category': X_data_plot['wind_category'],
        'temp_next_5': y_data_plot['temp_next_5']
    })
    
    # Tính mean temperature và sắp xếp
    mean_temp_by_wind = boxplot_data.groupby('wind_category')['temp_next_5'].mean().sort_values()
    
    # Tạo màu gradient đỏ (nhiệt độ cao hơn = đỏ đậm hơn)
    min_temp = mean_temp_by_wind.min()
    max_temp = mean_temp_by_wind.max()
    
    # Tạo colors từ nhạt đến đậm theo nhiệt độ
    colors = []
    for temp in mean_temp_by_wind.values:
        # Normalize nhiệt độ từ 0 đến 1
        normalized = (temp - min_temp) / (max_temp - min_temp) if max_temp != min_temp else 0.5
        # Tạo màu đỏ với intensity theo normalized value
        red_intensity = 0.6 + 0.4 * normalized  # từ 0.6 đến 1.0
        colors.append((red_intensity, 0.2, 0.2, 0.8))  # RGBA
    
    # Bar plot (mean temperature)
    bars = plt.bar(mean_temp_by_wind.index, mean_temp_by_wind.values, 
                   color=colors, edgecolor='darkred', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Wind Category', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Temperature Next 5 Days (°C)', fontsize=12, fontweight='bold')
    plt.title(f'Mean Temperature by Wind Category{title_year}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị trên mỗi bar
    for bar, value in zip(bars, mean_temp_by_wind.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                f'{value:.2f}°C', ha='center', va='bottom', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Thêm số lượng samples dưới mỗi bar
    wind_counts = boxplot_data['wind_category'].value_counts()
    for i, category in enumerate(mean_temp_by_wind.index):
        count = wind_counts[category]
        plt.text(i, mean_temp_by_wind.values[i] * 0.05, f'n={count}', 
                ha='center', va='bottom', fontsize=9, color='darkblue', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_cloudcover_vs_temp_simple(X_data, y_data, year=None, figsize=(10, 6)):
    """
    Vẽ biểu đồ đơn giản giữa Cloudcover Rolling Mean và Temperature Next 5 Days
    """
    
    # Lọc dữ liệu theo năm nếu có chỉ định
    if year is not None:
        mask_year = y_data.index.year == year
        y_data_plot = y_data[mask_year]
        X_data_plot = X_data[mask_year]
        title_year = f" - Year {year}"
    else:
        y_data_plot = y_data
        X_data_plot = X_data
        title_year = ""
    
    # Tính correlation
    correlation = np.corrcoef(X_data_plot['cloudcover_roll_mean_21'], y_data_plot['temp_next_5'])[0, 1]
    
    # Vẽ biểu đồ
    plt.figure(figsize=figsize)
    
    plt.scatter(X_data_plot['cloudcover_roll_mean_21'], y_data_plot['temp_next_5'], 
               alpha=0.6, s=20, c=y_data_plot['temp_next_5'], cmap='coolwarm')
    
    # Vẽ regression line
    z = np.polyfit(X_data_plot['cloudcover_roll_mean_21'], y_data_plot['temp_next_5'], 1)
    p = np.poly1d(z)
    plt.plot(X_data_plot['cloudcover_roll_mean_21'], p(X_data_plot['cloudcover_roll_mean_21']), 
            "r--", alpha=0.8, linewidth=2, 
            label=f'Regression line (r={correlation:.3f})')
    
    plt.xlabel('Cloudcover Rolling Mean (21 Days)')
    plt.ylabel('Temperature Next 5 Days (°C)')
    plt.title(f'Cloudcover vs Temperature Next 5 Days{title_year}')
    plt.colorbar(label='Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Correlation: {correlation:.3f}")
    print(f"Cloudcover stats: mean={X_data_plot['cloudcover_roll_mean_21'].mean():.3f}, "
          f"std={X_data_plot['cloudcover_roll_mean_21'].std():.3f}")
    
def plot_thermal_vs_temp_simple(X_data, y_data, year=None, figsize=(10, 6)):
    """
    Vẽ biểu đồ đơn giản giữa Thermal Index và Temperature Next 5h
    """
    
    # Lọc dữ liệu theo năm nếu có chỉ định
    if year is not None:
        mask_year = y_data.index.year == year
        y_data_plot = y_data[mask_year]
        X_data_plot = X_data[mask_year]
        title_year = f" - Year {year}"
    else:
        y_data_plot = y_data
        X_data_plot = X_data
        title_year = ""
    
    # Tính correlation
    correlation = np.corrcoef(X_data_plot['thermal_index'], y_data_plot['temp_next_5'])[0, 1]
    
    # Vẽ biểu đồ
    plt.figure(figsize=figsize)
    
    plt.scatter(X_data_plot['thermal_index'], y_data_plot['temp_next_5'], 
               alpha=0.6, s=20, c=y_data_plot['temp_next_5'], cmap='coolwarm')
    
    # Vẽ regression line
    z = np.polyfit(X_data_plot['thermal_index'], y_data_plot['temp_next_5'], 1)
    p = np.poly1d(z)
    plt.plot(X_data_plot['thermal_index'], p(X_data_plot['thermal_index']), 
            "r--", alpha=0.8, linewidth=2, 
            label=f'Regression line (r={correlation:.3f})')
    
    plt.xlabel('Thermal Index')
    plt.ylabel('Temperature Next 5 Days (°C)')
    plt.title(f'Thermal Index vs Temperature Next 5 Days {title_year}')
    plt.colorbar(label='Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Correlation: {correlation:.3f}")
    print(f"Thermal Index stats: mean={X_data_plot['thermal_index'].mean():.2f}, "
          f"std={X_data_plot['thermal_index'].std():.2f}")
    
def plot_sunrise_vs_temp(X_data, y_data, year=2023, figsize=(16, 8)):
    """
    Vẽ biểu đồ so sánh Sunrise Time và Temperature Next 5h theo năm
    
    Parameters:
    -----------
    X_data : DataFrame
        DataFrame chứa feature 'sunrise'
    y_data : DataFrame
        DataFrame chứa target 'temp_next_5'  
    year : int
        Năm muốn vẽ biểu đồ (default: 2023)
    figsize : tuple
        Kích thước figure (default: (16, 8))
    """
    # Lọc dữ liệu chỉ trong năm chỉ định
    mask_year = y_data.index.year == year
    y_data_year = y_data[mask_year]
    X_data_year = X_data[mask_year]
    
    # Kiểm tra xem có dữ liệu không
    if len(y_data_year) == 0:
        print(f"Không có dữ liệu cho năm {year}")
        return

    # Tạo figure và axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Màu sắc cho các đường
    color1 = 'tab:blue'
    color2 = 'tab:red'
    
    # Vẽ đường sunrise trên trục Y bên trái
    ax1.set_ylabel('Sunrise Time (Hour)', color=color1, fontsize=12)
    line1 = ax1.plot(y_data_year.index, X_data_year['sunrise'], 
                    color=color1, linewidth=1.5, alpha=0.8, label='Sunrise Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(5, 7)  # Giới hạn trục Y cho sunrise từ 5-7
    ax1.grid(True, alpha=0.3)
    
    # Định dạng trục X cho đẹp
    plt.xticks(rotation=45)
    
    # Tạo trục Y thứ hai bên phải cho temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature Next 5 Days (°C)', color=color2, fontsize=12)
    line2 = ax2.plot(y_data_year.index, y_data_year['temp_next_5'], 
                    color=color2, linewidth=1.5, alpha=0.8, label='Temperature Next 5h')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(10, 36)  # Giới hạn trục Y cho temperature từ 10-36
    
    # Title và legend
    plt.title(f'Sunrise and Temperature in Next 5 horizon - Year {year}', 
              fontsize=14, fontweight='bold')
    
    # Kết hợp legends từ cả hai trục
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    

# icon and temp
def icon(df0):
   summary = df0.groupby('icon')['temp'].agg(['mean', 'std', 'count']).round(2)

   plt.figure(figsize=(10, 8))
   sns.heatmap(summary, annot=True, cmap='YlOrRd', fmt='.2f')
   plt.title('Temperature statistics by icon')
   plt.show()


def plot_temp_icon_2024(df0):
    # --- Step 0: Filter for 2024 ---
    df_2024 = df0[df0['datetime'].dt.year == 2024].copy()

    # --- Step 1: Colors for each icon ---
    icon_types = df_2024['icon'].unique()
    colors = {
        icon_types[i]: plt.cm.tab10(i % 10)
        for i in range(len(icon_types))
    }

    # --- Step 2: Create the plot ---
    fig, ax = plt.subplots(figsize=(17, 7))

    ax.plot(
        df_2024['datetime'], df_2024['temp'],
        color='gray', alpha=0.4, linewidth=1,
        label='Temperature Trend'
    )

    # --- Step 3: Scatter by icon ---
    for icon in icon_types:
        df_icon = df_2024[df_2024['icon'] == icon]
        ax.scatter(
            df_icon['datetime'], df_icon['temp'],
            color=colors[icon], s=30, label=icon
        )

    # --- Step 4: Formatting ---
    ax.set_title("Temperature and Weather Icons — 2024", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(right=0.75)

    # --- Step 5: Insight text ---
    insight_text = """
Insight:

This chart visualizes daily temperatures
alongside weather condition icons,
revealing several patterns:

- "clear-day" / "partly-cloudy-day":
  These icons tend to cluster around the
  higher temperature peaks.

- "rain" / "cloudy":
  These conditions often align with dips
  in temperature due to cooling effects.

- Seasonal Trends:
  The gray line shows the seasonal temperature
  curve, with icons explaining daily variations.
"""
    fig.text(0.77, 0.5, insight_text, ha='left', va='center', fontsize=10, wrap=True)

    # Unique legend
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = dict(zip(labels, handles))
    ax.legend(unique_handles.values(), unique_handles.keys(), fontsize=10, loc='upper right')

    plt.show()

def solar(df0):
   plt.figure(figsize=(15, 10))
   plt.title('Solarradiation vs Temperature are highly correlated', fontsize=11)
   sns.regplot(data=df0, x='solarradiation', y='temp',
               scatter_kws={'color':'#ffb366', 'alpha':0.6},
               line_kws={'color':'red', 'linewidth':1.5})
   plt.xlabel('Solarradiation')
   plt.ylabel('Temperature (°C)')

   plt.tight_layout()
   plt.show()


def plot_wind_distribution(df0):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    # --- Step 1: Prepare data for the year 2024 ---
    df0['datetime'] = pd.to_datetime(df0['datetime'])
    df_2024 = df0[df0['datetime'].dt.year == 2024].copy()

    # Create a 'month' column for easy grouping
    df_2024['month'] = df_2024['datetime'].dt.month


    # --- Step 2: Categorize wind direction into 4 groups ---
    def get_wind_category(deg):
        # Note: 0 degrees (North) is grouped with Northwest for a complete 360-degree cycle.
        if 0 < deg <= 90:
            return 'Northeast (NE)'
        elif 90 < deg <= 180:
            return 'Southeast (SE)'
        elif 180 < deg <= 270:
            return 'Southwest (SW)'
        elif 270 < deg <= 360 or deg == 0:
            return 'Northwest (NW)'
        else:
            return 'Unknown'

    # Apply the function to create a new category column
    df_2024['wind_category'] = df_2024['winddir'].apply(get_wind_category)


    # --- Step 3: Count occurrences and calculate proportions ---
    wind_proportions = pd.crosstab(
        index=df_2024['month'],
        columns=df_2024['wind_category'],
        normalize='index'
    ) * 100

    desired_order = ['Northeast (NE)', 'Southeast (SE)', 'Southwest (SW)', 'Northwest (NW)']
    wind_proportions = wind_proportions.reindex(columns=desired_order, fill_value=0)

    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    wind_proportions = wind_proportions.rename(index=month_names)


    # --- Step 4: Create the plot object explicitly ---
    fig, ax = plt.subplots(figsize=(16, 8))

    wind_proportions.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='viridis',
        edgecolor='white',
        width=0.8
    )

    # --- Step 5: Add Percentage Labels ---
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.0f%%',
            label_type='center',
            color='white',
            weight='bold',
            labels=[f'{h:.0f}%' if h > 4 else '' for h in container.datavalues]
        )

    # --- Step 6: Format ---
    ax.set_title('Monthly Wind Direction Proportions in 2024', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.yaxis.set_visible(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(title='Wind Direction', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.subplots_adjust(right=0.75)

    insight_text = """
    The chart clearly shows a seasonal
    wind pattern for Hanoi in 2024.

    - Summer Dominance (May-Aug):
      Southeast (SE) winds are the most
      prevalent, corresponding to the
      humid summer monsoon season.

    - Winter Dominance (Oct-Feb):
      Northwest (NW) and Northeast (NE)
      winds dominate, bringing cooler
      and drier air from the mainland.
    """

    fig.text(0.77, 0.5, insight_text, ha='left', va='center', fontsize=11, wrap=True)

    plt.show()
    

def plot_temp_dewpoint_2024(df, datetime_col='datetime',
                             temp_col='temp', dew_col='dew',
                             title='Dew Point and Temperature are Closely Linked — Year 2024 (Hanoi)'):
    """
    Plot temperature and dew point for year 2024 with humid/dry season highlights.
    """

    # --- Preprocess ---
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df['year'] = df[datetime_col].dt.year

    # Filter and sort data for 2024
    dfi = df[df['year'] == 2024].sort_values(datetime_col)

    # --- Create plot ---
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot lines
    ax.plot(dfi[datetime_col], dfi[temp_col], label='Temperature (°C)', color='orange', linewidth=1)
    ax.plot(dfi[datetime_col], dfi[dew_col], label='Dew Point (°C)', color='blue', linewidth=1)

    # Highlight specific periods
    ax.axvspan(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-04-30'),
               color='blue', alpha=0.15, lw=0)
    ax.axvspan(pd.to_datetime('2024-10-01'), pd.to_datetime('2024-12-31'),
               color='goldenrod', alpha=0.15, lw=0)

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Title + legend
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize='small')

    # Improve layout
    plt.subplots_adjust(right=0.75)

    # Insight text
    insight_text = """
- Humid "Nồm" Season (Blue):
  Jan–Apr: Dew point nearly equals
  temperature → High humidity.

- Dry Season (Goldenrod):
  Oct–Dec: Larger gap between
  temperature and dew point →
  Lower humidity.
"""
    fig.text(0.77, 0.5, insight_text, ha='left', va='center', fontsize=10, wrap=True)

    plt.show()


def plot_temp_feelslike_2024(df1):
    df0 = df1.copy()
    df0['datetime'] = pd.to_datetime(df0['datetime'])
    # --- Step 0: Filter for year 2024 only ---
    df_2024 = df0[(df0['datetime'].dt.year == 2024)].copy()

    # --- Step 1: Create the plot ---
    fig, ax = plt.subplots(figsize=(17, 7))

    # Plot daily data
    ax.plot(df_2024['datetime'], df_2024['temp'], 
            label='Temperature (°C)', color='orangered', linewidth=1.5)

    ax.plot(df_2024['datetime'], df_2024['feelslike'], 
            label='Feels Like (°C)', color='dodgerblue', linewidth=1.5)

    # --- Step 2: Highlight seasons ---
    # Summer: May → Sep
    ax.axvspan(pd.to_datetime('2024-05-01'), pd.to_datetime('2024-09-30'),
               color='orange', alpha=0.2, lw=0, label='Summer')

    # Winter early 2024: Jan → Mar
    ax.axvspan(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-03-31'),
               color='lightblue', alpha=0.3, lw=0, label='Winter')

    # Winter late 2024: Nov → Dec
    ax.axvspan(pd.to_datetime('2024-11-01'), pd.to_datetime('2024-12-31'),
               color='lightblue', alpha=0.3, lw=0)

    # --- Step 3: Formatting ---
    ax.set_title('"Feels Like" vs Actual Temperature — Daily Values (2024)', fontsize=16)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(rotation=0)

    # Legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize='medium')

    # --- Step 4: Insight panel ---
    plt.subplots_adjust(right=0.75)

    insight_text = """ Insight: 
The chart illustrates how "Feels Like" 
temperature diverges from the actual 
temperature based on the season. 

- Summer (Orange Period): The "Feels Like" value 
is consistently higher than the actual temperature 
due to high humidity, which reduces the body's 
ability to cool down. 

- Winter (Blue Period): The "Feels Like" value is 
often lower due to wind chill, which accelerates 
heat loss from the body. """

    fig.text(0.77, 0.5, insight_text, ha='left', va='center', fontsize=10, wrap=True)

    plt.show()

def plot_corr_heatmap(df0, figsize=(10,8), title="Feature Correlation Matrix",
                      cols=None, cmap="coolwarm"):
    """
    Vẽ heatmap cho ma trận tương quan của các cột số trong DataFrame.

    Parameters:
        df (DataFrame): dữ liệu đầu vào
        figsize (tuple): kích thước hình
        title (str): tiêu đề
        cols (list or None): danh sách cột muốn vẽ (mặc định lấy cột số)
        cmap (str): colormap
    """
    df = df0.copy()
    df = df.drop(['snow','snowdepth'], axis = 1)
    # Chọn cột
    if cols is None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"])
    else:
        numeric_cols = df[cols]

    corr = numeric_cols.corr()

    # Vẽ heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap=cmap, annot=False)
    plt.title(title)
    plt.show()


def generate_enhanced_stats_table(df, caption='Basic Statistics of Numerical Features'):
    """
    Generate an enhanced statistics table for numerical features with styling
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    caption (str): Table caption
    
    Returns:
    styled_table: Styled pandas DataFrame
    """
    # Select numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    stats_table = numerical_df.describe().T
    
    # Calculate null statistics
    null_count = numerical_df.isnull().sum()
    null_pct = (numerical_df.isnull().mean() * 100).round(2)
    
    def highlight_zero_std(s):
        """Highlight rows with zero standard deviation"""
        is_zero = s == 0
        return ['background-color: yellow' if v else '' for v in is_zero]
    
    # Create enhanced statistics dataframe
    enhanced_stats = pd.DataFrame({
        'null_count': null_count,
        'null_pct': null_pct,
        'mean': stats_table['mean'].round(2),
        'std': stats_table['std'].round(2),
        'min': stats_table['min'].round(2),
        '25%': stats_table['25%'].round(2),
        '50%': stats_table['50%'].round(2),
        '75%': stats_table['75%'].round(2),
        'max': stats_table['max'].round(2)
    })
    
    # Apply styling
    styled_table = enhanced_stats.style\
        .background_gradient(subset=['null_pct'], cmap='Reds')\
        .apply(highlight_zero_std, subset=['std'])\
        .format({
            'null_pct': '{:.2f}%',
            'mean': '{:.2f}',
            'std': '{:.2f}',
            'min': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'max': '{:.2f}'
        })\
        .set_properties(**{
            'border': '1px solid black',
            'text-align': 'center'
        })\
        .set_caption(caption)\
        .set_table_styles([{
            'selector': 'th',
            'props': [('background-color', '#4CAF50'), 
                     ('color', 'white'),
                     ('font-weight', 'bold')]
        }])
    
    return styled_table

def create_variable_groups_table(df, datetime_cols=None, category_cols=None, low_unique_threshold=5):
    """
    Create a comprehensive table grouping variables by pre-defined types with statistics.
    Also detect low-unique categorical columns and list their value counts.
    """

    # Defaults
    if datetime_cols is None:
        datetime_cols = ['datetime', 'sunrise', 'sunset']
    if category_cols is None:
        category_cols = ['name', 'preciptype', 'conditions', 'description', 'icon', 'stations']

    defined_cols = datetime_cols + category_cols
    numerical_cols = [col for col in df.columns if col not in defined_cols]

    variable_groups = []

    # ==== DATETIME COLUMNS ====
    for col in datetime_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df) * 100).round(2)
            unique_count = df[col].nunique()

            variable_groups.append({
                'Variable': col,
                'Type': 'datetime',
                'Count': len(df),
                'Null Count': null_count,
                'Null %': null_pct,
                'Additional Info': f"Unique: {unique_count}"
            })

    # ==== CATEGORICAL COLUMNS ====
    for col in category_cols:
        if col in df.columns:

            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df) * 100).round(2)
            unique_count = df[col].nunique()

            # Top value
            if df[col].dropna().empty:
                top_value = "N/A"
            else:
                top_value = df[col].mode().iloc[0]

            # Low unique detection
            if unique_count <= low_unique_threshold:
                vc = df[col].value_counts().to_dict()
                vc_str = ", ".join([f"{k}({v})" for k, v in vc.items()])
                additional = f"Unique: {unique_count} | Values: {vc_str}"
            else:
                additional = f"Unique: {unique_count}, Top: {str(top_value)[:20]}..."

            # Append to table
            variable_groups.append({
                'Variable': col,
                'Type': 'categorical',
                'Count': len(df),
                'Null Count': null_count,
                'Null %': null_pct,
                'Additional Info': additional
            })

    # ==== CONVERT TO DF ====
    groups_df = pd.DataFrame(variable_groups)

    # ==== STYLING ====
    def color_variable_type(val):
        if val == 'datetime':
            return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
        elif val == 'numerical':
            return 'background-color: #e3f2fd; color: #1565c0; font-weight: bold'
        elif val == 'categorical':
            return 'background-color: #f3e5f5; color: #7b1fa2; font-weight: bold'
        else:
            return ''

    styled_table = (
        groups_df.style
        .applymap(color_variable_type, subset=['Type'])
        .background_gradient(subset=['Null %'], cmap='Reds')
        .format({'Null %': '{:.2f}%'})
        .set_properties(**{
            'border': '1px solid #dee2e6',
            'text-align': 'center',
            'padding': '8px',
            'font-size': '12px'
        })
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#495057'),
                                        ('color', 'white'),
                                        ('font-weight', 'bold'),
                                        ('padding', '10px')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#f8f9fa')]}
        ])
        .set_caption('<h4> Variable Groups Overview</h4>')
        .hide(axis='index')
    )

    return styled_table


def hourly_daily():
   # Data preparation - chỉ so sánh 2 models
   models = ['Daily LGBM\nFinal', 'Hourly Enriched\nLGBM (tuned)']
   rmse_means = [2.2992, 2.2162]
   mae_means = [1.8211, 1.7556]
   r2_means = [0.7813, 0.7935]

   rmse_stds = [0.1346, 0.4519]
   mae_stds = [0.1236, 0.3747]
   r2_stds = [0.0290, 0.0717]

   # Create subplots
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

   # Colors
   colors = ['#2E86AB', '#D32F2F']

   # RMSE plot
   bars1 = ax1.bar(models, rmse_means, yerr=rmse_stds, capsize=5, color=colors, alpha=0.8)
   ax1.set_title('RMSE Comparison', fontweight='bold', fontsize=12)
   ax1.tick_params(axis='x', labelsize=10)

   # MAE plot
   bars2 = ax2.bar(models, mae_means, yerr=mae_stds, capsize=5, color=colors, alpha=0.8)
   ax2.set_title('MAE Comparison', fontweight='bold', fontsize=12)
   ax2.tick_params(axis='x', labelsize=10)

   # R² plot
   bars3 = ax3.bar(models, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.8)
   ax3.set_title('R² Comparison', fontweight='bold', fontsize=12)
   ax3.tick_params(axis='x', labelsize=10)

   # Highlight the best performer (Hourly Enriched LGBM tuned)
   bars1[1].set_edgecolor('red')
   bars1[1].set_linewidth(2)
   bars2[1].set_edgecolor('red')
   bars2[1].set_linewidth(2)
   bars3[1].set_edgecolor('red')
   bars3[1].set_linewidth(2)

   # Remove y-axis labels
   ax1.set_ylabel('')
   ax2.set_ylabel('')
   ax3.set_ylabel('')

   plt.tight_layout()
   plt.savefig('figures/daily_vs_hourly_performance.png', dpi=300, bbox_inches='tight')
   plt.show()

def plot_horizon_comparison():
    # Data preparation
    days = [1, 2, 3, 4, 5]

    # RMSE values
    rmse_daily = [1.468, 2.110, 2.364, 2.476, 2.514]
    rmse_hourly = [1.306, 2.048, 2.346, 2.486, 2.500]

    # MAE values
    mae_daily = [1.125, 1.651, 1.877, 1.982, 1.988]
    mae_hourly = [1.003, 1.611, 1.864, 1.989, 1.988]

    # Colors
    daily_color = '#2E86AB'   # Blue
    hourly_color = '#d63031'  # Red
    highlight_bg = "#e2b1b1"  # Light gray for highlight

    # Create figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ===== PLOT 1: RMSE comparison =====
    ax1.plot(days, rmse_daily, color=daily_color, linewidth=2.5, marker='o', markersize=6, label='Daily Features')
    ax1.plot(days, rmse_hourly, color=hourly_color, linewidth=2.5, marker='s', markersize=6, label='Hourly Features')

    # Highlight horizon 3-5
    ax1.axvspan(3, 5, color=highlight_bg, alpha=0.3)

    ax1.set_xlabel('Forecasting Horizon', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(days)

    # ===== PLOT 2: MAE comparison =====
    ax2.plot(days, mae_daily, color=daily_color, linewidth=2.5, marker='o', markersize=6, label='Daily Features')
    ax2.plot(days, mae_hourly, color=hourly_color, linewidth=2.5, marker='s', markersize=6, label='Hourly Features')

    # Highlight horizon 3-5
    ax2.axvspan(3, 5, color=highlight_bg, alpha=0.3)

    ax2.set_xlabel('Forecasting Horizon', fontsize=12, fontweight='bold')
    ax2.set_title('MAE', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(days)

    # ===== LEGEND CHUNG =====
    legend_elements = [
        Line2D([0], [0], color=daily_color, marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Daily Features'),
        Line2D([0], [0], color=hourly_color, marker='s', linestyle='-', linewidth=2.5, markersize=8, label='Hourly Features'),
        Patch(facecolor=highlight_bg, edgecolor="#361236", alpha=0.3, label='Longer Horizon 3-5'),
    ]

    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
            fontsize=12, frameon=True, fancybox=True, shadow=True, ncol=1)

    # ===== KEY INSIGHTS TEXT BOX =====
    insight_text = """KEY INSIGHTS:

    • Hourly features significantly improve
    short-term forecasting (Day 1-2)
    
    • Day 1: RMSE reduced by 11%
    • Day 1: MAE reduced by 11%

    • Performance gap narrows for
    longer horizons (Day 3-5)

    → Hourly features are most beneficial
    for 1-2 day forecasts"""

    fig.text(0.83, 0.75, insight_text, fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='#2E86AB', alpha=0.9),
            verticalalignment='top', linespacing=1.5)

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Make space for text box
    plt.savefig('figures/daily_vs_hourly_horizon.png', dpi=300, bbox_inches='tight')
    plt.show()

def cv_metric_horizon():
    # Data for Ridge
    ridge_rmse_means = [1.5325, 2.2207, 2.5017, 2.5879, 2.6164]
    ridge_mae_means = [1.1858, 1.7490, 1.9916, 2.0732, 2.0849]

    # Data for LGBM
    lgbm_rmse_means = [1.5587, 2.2607, 2.5045, 2.5682, 2.5987]
    lgbm_mae_means = [1.2018, 1.7782, 1.9940, 2.0520, 2.0750]

    # Horizons
    horizons = [1, 2, 3, 4, 5]

    # Colors
    ridge_color = '#d63031'  # Đỏ
    lgbm_color = '#2E86AB'   # Xanh dương
    ridge_bg = '#ffe6e6'     # Background nhạt cho Ridge
    lgbm_bg = '#e6f0ff'      # Background nhạt cho LGBM

    # Create figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ===== PLOT 1: RMSE comparison =====
    ax1.plot(horizons, ridge_rmse_means, color=ridge_color, linewidth=2.5, marker='o', markersize=6, label='Ridge Tuned')
    ax1.plot(horizons, lgbm_rmse_means, color=lgbm_color, linewidth=2.5, marker='s', markersize=6, label='LGBM Tuned')

    # Highlight horizon 3-5
    ax1.axvspan(3, 5, color=ridge_bg, alpha=0.3)
    ax1.axvspan(3, 5, color=lgbm_bg, alpha=0.3)

    ax1.set_xlabel('Forecasting Horizon (Days)', fontsize=12, fontweight='bold')
    ax1.set_title('CV RMSE', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(horizons)

    # ===== PLOT 2: MAE comparison =====
    ax2.plot(horizons, ridge_mae_means, color=ridge_color, linewidth=2.5, marker='o', markersize=6, label='Ridge Tuned')
    ax2.plot(horizons, lgbm_mae_means, color=lgbm_color, linewidth=2.5, marker='s', markersize=6, label='LGBM Tuned')

    # Highlight horizon 3-5
    ax2.axvspan(3, 5, color=ridge_bg, alpha=0.3)
    ax2.axvspan(3, 5, color=lgbm_bg, alpha=0.3)

    ax2.set_xlabel('Forecasting Horizon (Days)', fontsize=12, fontweight='bold')
    ax2.set_title('CV MAE', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(horizons)

    # ===== LEGEND CHUNG =====
    legend_elements = [
        Line2D([0], [0], color=ridge_color, marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Ridge Tuned'),
        Line2D([0], [0], color=lgbm_color, marker='s', linestyle='-', linewidth=2.5, markersize=8, label='LGBM Tuned'),
        Patch(facecolor=ridge_bg, edgecolor="#361236", alpha=0.3, label='Longer Horizon 3-5'),
    ]

    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
            fontsize=12, frameon=True, fancybox=True, shadow=True, ncol=1)

    # ===== KEY INSIGHTS TEXT BOX (English) =====
    insight_text = """KEY INSIGHTS:

    • Ridge performs better at short-term
    forecasting (Day 1-2)

    • LGBM performs better from horizon 3-5
    - Lower RMSE than Ridge
    - Lower MAE than Ridge  
    - Higher stability

    → LGBM is more suitable for
    longer term forecasting (3-5 days)"""

    fig.text(0.83, 0.75, insight_text, fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='#2E86AB', alpha=0.9),
            verticalalignment='top', linespacing=1.5)

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Để chỗ cho text box
    plt.savefig('figures/per_horizon_cv_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def ridge_lgbm():
    # Data for tuned models
    models = ['Ridge Tuned', 'LGBM Tuned']
    rmse_values = [2.2919, 2.2992]
    mae_values = [1.8169, 1.8211]
    r2_values = [0.7814, 0.7813]

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Colors
    ridge_color = '#d63031'
    lgbm_color = '#2E86AB'

    # Plot 1: RMSE comparison
    x = np.arange(len(models))
    width = 0.6

    bars1 = ax1.bar(x, rmse_values, width, color=[ridge_color, lgbm_color], alpha=0.8)
    ax1.set_title('RMSE', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)  # Bỏ rotation
    ax1.set_ylim(0, 3)  # Tăng y-limit đến 3
    ax1.grid(True, alpha=0.3)

    # Add values on bars
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 2: MAE comparison
    bars2 = ax2.bar(x, mae_values, width, color=[ridge_color, lgbm_color], alpha=0.8)
    ax2.set_title('MAE', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)  # Bỏ rotation
    ax2.set_ylim(0, 3)  # Tăng y-limit đến 3
    ax2.grid(True, alpha=0.3)

    # Add values on bars
    for bar, value in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 3: R² comparison
    bars3 = ax3.bar(x, r2_values, width, color=[ridge_color, lgbm_color], alpha=0.8)
    ax3.set_title('R²', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)  # Bỏ rotation
    ax3.set_ylim(0, 1)  # R² vẫn giữ 0-1
    ax3.grid(True, alpha=0.3)

    # Add values on bars
    for bar, value in zip(bars3, r2_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Bỏ tất cả xlabel và ylabel
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax3.set_xlabel('')
    ax3.set_ylabel('')

    plt.tight_layout()
    plt.savefig('figures/tuned_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_test_gap():
    # --- Chart 2: Horizon-wise Gap ---
    horizons = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
    baseline_gap = [0.6915, 1.0250, 1.1818, 1.3531, 1.4625]
    optimized_gap = [0.2626, 0.3331, 0.5779, 0.3747, 0.7057]

    plt.figure(figsize=(10, 5))
    x = range(len(horizons))

    plt.plot(x, baseline_gap, marker='o', label="Baseline Gap", linewidth=2, markersize=8, color='red')
    plt.plot(x, optimized_gap, marker='s', label="Optimized Gap", linewidth=2, markersize=8, color='green')

    plt.xticks(x, horizons, fontsize=11)
    plt.xlabel("Forecasting Horizon", fontsize=12, fontweight='bold')
    plt.title("Train-Test Generalization Gap: Baseline vs Optimized Model", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.6)  # Trục y bắt đầu từ 0

    # Thêm giá trị trên các điểm
    for i, (base, opt) in enumerate(zip(baseline_gap, optimized_gap)):
        plt.text(i, base + 0.05, f'{base:.3f}', ha='center', va='bottom', fontsize=9, color='red')
        plt.text(i, opt - 0.08, f'{opt:.3f}', ha='center', va='top', fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig('figures/horizon_gap_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def cv_range():
   # Data
   metrics = ['RMSE', 'MAE', 'R²']
   cv_means = [2.2992, 1.8211, 0.7813]
   cv_stds = [0.1346, 0.1236, 0.0290]
   test_values = [2.1866, 1.7246, 0.8149]

   # Create figure
   fig, ax = plt.subplots(figsize=(10, 6))

   x_pos = np.arange(len(metrics))
   width = 0.6

   # Plot CV means with error bars (representing ±1 std)
   bars = ax.bar(x_pos, cv_means, width, yerr=cv_stds, 
               capsize=8, label='CV Mean ± Std', 
               color='lightblue', alpha=0.7, edgecolor='navy', linewidth=1.5)

   # Plot test values as points
   # ax.scatter(x_pos, test_values, color='red', s=100, zorder=3, 
   #            label='Test Value', marker='o', edgecolor='darkred', linewidth=1)

   # Customize plot
   ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
   ax.set_ylabel('Score', fontsize=12, fontweight='bold')
   ax.set_title('Generalization Check: CV Performance vs Test Set', 
               fontsize=14, fontweight='bold', pad=20)
   ax.set_xticks(x_pos)
   ax.set_xticklabels(metrics, fontsize=11)
   ax.legend(fontsize=10)

   plt.tight_layout()
   plt.savefig('figures/generalization_check.png', dpi=300, bbox_inches='tight')
   plt.show()

def metric_overall():
    # 1. Biểu đồ so sánh average metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Biểu đồ 1: Average metrics comparison
    metrics = ['RMSE', 'MAE', 'R²']
    baseline_avg = [2.4121, 1.8246, 0.7642]
    optimized_avg = [2.1866, 1.7246, 0.8149]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_avg, width, label='Baseline', color='gray', alpha=0.7)
    bars2 = ax1.bar(x + width/2, optimized_avg, width, label='Optimized', color='#2E86AB', alpha=0.8)

    ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax1.set_title('Average Performance: Optimized vs Baseline', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Thêm giá trị trên cột
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Biểu đồ 2: Horizon-wise RMSE comparison
    horizons = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
    baseline_rmse = [1.4713, 2.2180, 2.5523, 2.8270, 2.9917]
    optimized_rmse = [1.4682, 2.1102, 2.3645, 2.4761, 2.5139]

    x2 = np.arange(len(horizons))
    bars3 = ax2.bar(x2 - width/2, baseline_rmse, width, label='Baseline', color='gray', alpha=0.7)
    bars4 = ax2.bar(x2 + width/2, optimized_rmse, width, label='Optimized', color='#2E86AB', alpha=0.8)

    ax2.set_xlabel('Forecasting Horizon', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE by Horizon: Optimized vs Baseline', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(horizons, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Thêm giá trị trên cột
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/optimized_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def cv_overview():
   # Dữ liệu từ output của bạn
   folds_data = {
      'fold1': {
         'train_start': '2015-10-20', 'train_end': '2017-01-19',
         'val_start': '2017-02-19', 'val_end': '2018-05-21',
         'train_samples': 458, 'val_samples': 457
      },
      'fold2': {
         'train_start': '2015-10-20', 'train_end': '2018-05-21',
         'val_start': '2018-06-21', 'val_end': '2019-09-20',
         'train_samples': 945, 'val_samples': 457
      },
      'fold3': {
         'train_start': '2015-10-20', 'train_end': '2019-09-20',
         'val_start': '2019-10-21', 'val_end': '2021-01-19',
         'train_samples': 1432, 'val_samples': 457
      },
      'fold4': {
         'train_start': '2015-10-20', 'train_end': '2021-01-19',
         'val_start': '2021-02-19', 'val_end': '2022-05-21',
         'train_samples': 1919, 'val_samples': 457
      },
      'fold5': {
         'train_start': '2015-10-20', 'train_end': '2022-05-21',
         'val_start': '2022-06-21', 'val_end': '2023-09-20',
         'train_samples': 2406, 'val_samples': 457
      }
   }

   # Chuyển đổi ngày tháng
   for fold in folds_data.values():
      for key in ['train_start', 'train_end', 'val_start', 'val_end']:
         fold[key] = datetime.strptime(fold[key], '%Y-%m-%d')

   # Màu sắc
   train_color = '#1f77b4'  # blue
   val_color = '#ff7f0e'    # orange
   gap_color = 'lightgray'

   # Vẽ biểu đồ tổng quan
   plt.figure(figsize=(14, 6))

   # Tạo timeline tổng quan
   all_dates = []
   for fold_data in folds_data.values():
      all_dates.extend([fold_data['train_start'], fold_data['train_end'], 
                        fold_data['val_start'], fold_data['val_end']])
   overall_start = min(all_dates)
   overall_end = max(all_dates)

   # Tạo danh sách các điểm mốc thời gian quan trọng cho trục X
   important_dates = [
      datetime(2016, 1, 1), datetime(2017, 1, 1), datetime(2018, 1, 1),
      datetime(2019, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1),
      datetime(2022, 1, 1), datetime(2023, 1, 1)
   ]

   # Vẽ từng fold trên cùng một timeline
   for i, (fold_name, fold_data) in enumerate(folds_data.items()):
      y_pos = i  # Fold 1 ở trên cùng, fold 5 ở dưới cùng
      
      # Tính positions dựa trên datetime
      train_start_pos = fold_data['train_start']
      train_end_pos = fold_data['train_end']
      val_start_pos = fold_data['val_start']
      val_end_pos = fold_data['val_end']
      
      # Tính gap period
      gap_start_pos = fold_data['train_end']
      gap_end_pos = fold_data['val_start']
      
      # Vẽ train period
      plt.barh(y_pos, (train_end_pos - train_start_pos).days, 
               left=(train_start_pos - overall_start).days, 
               height=0.4, color=train_color, alpha=0.7, label='Train' if i==0 else "")
      
      # Vẽ gap period
      plt.barh(y_pos, (gap_end_pos - gap_start_pos).days, 
               left=(gap_start_pos - overall_start).days, 
               height=0.4, color=gap_color, alpha=0.5, label='Cutoff 30 days (lag/rolling)' if i==0 else "")
      
      # Vẽ validation period
      plt.barh(y_pos, (val_end_pos - val_start_pos).days, 
               left=(val_start_pos - overall_start).days, 
               height=0.4, color=val_color, alpha=0.7, label='Validation' if i==0 else "")
      
      # Thêm thông tin samples - FONT SIZE LỚN HƠN
      plt.text((train_start_pos - overall_start).days, y_pos, 
               f" {fold_data['train_samples']}", va='center', ha='left', 
               fontsize=12, fontweight='bold')  # Tăng từ 8 lên 12
      plt.text((val_start_pos - overall_start).days, y_pos, 
               f" {fold_data['val_samples']}", va='center', ha='left', 
               fontsize=12, fontweight='bold')  # Tăng từ 8 lên 12

   # Customize overview plot - FONT SIZE LỚN HƠN
   plt.yticks(range(len(folds_data)), [f'Fold {i+1}' for i in range(len(folds_data))], 
            fontsize=12)  # Tăng font size cho trục Y
   plt.ylabel('Folds', fontweight='bold', fontsize=14)  # Tăng font size

   # Thiết lập trục X với timeline - NẰM NGANG
   x_ticks = [(date - overall_start).days for date in important_dates]
   x_tick_labels = [date.strftime('%Y-%m') for date in important_dates]
   plt.xticks(x_ticks, x_tick_labels, rotation=0, fontsize=11)  # rotation=0 để nằm ngang, tăng font size
   plt.xlabel('Timeline', fontweight='bold', fontsize=14)

   plt.title('Time Series Cross-Validation - Overview of All Folds', fontsize=16, fontweight='bold')

   # LEGEND TO HƠN
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Tăng font size legend

   plt.grid(True, alpha=0.3, axis='x')
   plt.tight_layout()
   plt.show()

def plot_month_average(df):
    df0 = df.copy()
    # Tính trung bình nhiệt độ theo tháng
    df0['datetime'] = pd.to_datetime(df0['datetime'])
    monthly_avg = df0.groupby(df0['datetime'].dt.month)['temp'].mean()

    # Vẽ biểu đồ đường
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_avg.index, monthly_avg.values, color='#E74C3C', marker='o', 
            markerfacecolor='#C0392B', markersize=8, linestyle='-', linewidth=3, 
            label='Average Temperature')

    # Làm nổi bật điểm peak (tháng 6) và điểm thấp nhất (tháng 1)
    max_temp_idx = monthly_avg.idxmax()
    min_temp_idx = monthly_avg.idxmin()
    
    # Highlight peak point
    plt.scatter(max_temp_idx, monthly_avg[max_temp_idx], color='red', s=150, 
                zorder=5, label=f'Hottest Month (Jun: {monthly_avg[max_temp_idx]:.1f}°C)',
                edgecolors='darkred', linewidth=2)
    
    # Highlight lowest point  
    plt.scatter(min_temp_idx, monthly_avg[min_temp_idx], color='blue', s=150,
                zorder=5, label=f'Coldest Month (Jan: {monthly_avg[min_temp_idx]:.1f}°C)',
                edgecolors='darkblue', linewidth=2)

    # Ghi giá trị nhiệt độ bên trên mỗi điểm
    for i, val in enumerate(monthly_avg.values):
        plt.text(monthly_avg.index[i], val + 0.3, f"{val:.1f}°C", 
                ha='center', fontsize=10, fontweight='bold', color='#2C3E50')

    # Trang trí biểu đồ
    plt.title("Hanoi Average Monthly Temperature (2015–2025)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Month", fontsize=12, fontweight='bold')
    #plt.ylabel("Temperature (°C)", fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.xticks(range(1, 13), 
              ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 

    # Đặt legend và key insights gần nhau hơn
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # Thêm Key Insights - đặt sát dưới legend
    insights_text = """KEY INSIGHTS:

• SUMMER: Jun-Aug hottest
  (June: 30.3°C peak)

• WINTER: Dec-Feb coldest  
  (Jan: 17.9°C lowest)

• RAPID TRANSITIONS: 
  - Spring (Mar-May): Fast warming
  - Autumn (Sep-Nov): Fast cooling

• STABLE: Strong 10-year pattern"""

    plt.figtext(0.83, 0.76, insights_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.9),
                verticalalignment='top')

    plt.tight_layout()
    plt.show()


def plot_volatility_customized(df1):
    df = df1.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").set_index("datetime")
    df_2024 = df[df.index.year == 2024].copy()
    df_2024['temp_rolling_std_5'] = df_2024['temp'].rolling(window=5).std()
    
    # Tạo figure với layout tốt hơn
    plt.figure(figsize=(15, 8))
    
    # Tạo vùng màu theo mùa với độ nổi bật khác nhau
    winter_mask = (df_2024.index.month >= 1) & (df_2024.index.month <= 3)
    spring_mask = (df_2024.index.month >= 4) & (df_2024.index.month <= 5)
    summer_mask = (df_2024.index.month >= 6) & (df_2024.index.month <= 8)
    autumn_mask = (df_2024.index.month >= 9) & (df_2024.index.month <= 12)
    
    plt.fill_between(df_2024[winter_mask].index, 
                    df_2024[winter_mask]['temp'] - df_2024[winter_mask]['temp_rolling_std_5'], 
                    df_2024[winter_mask]['temp'] + df_2024[winter_mask]['temp_rolling_std_5'],
                    alpha=0.7, color='#FF8A65', label='High Volatility (Winter)')  # Cam đậm
    
    # Spring - Medium-High volatility
    plt.fill_between(df_2024[spring_mask].index, 
                    df_2024[spring_mask]['temp'] - df_2024[spring_mask]['temp_rolling_std_5'], 
                    df_2024[spring_mask]['temp'] + df_2024[spring_mask]['temp_rolling_std_5'],
                    alpha=0.6, color='#FF9800', label='Medium-High Volatility (Spring)')  # Cam vừa
    
    # Autumn - Medium volatility  
    plt.fill_between(df_2024[autumn_mask].index, 
                    df_2024[autumn_mask]['temp'] - df_2024[autumn_mask]['temp_rolling_std_5'], 
                    df_2024[autumn_mask]['temp'] + df_2024[autumn_mask]['temp_rolling_std_5'],
                    alpha=0.5, color='#FFB74D', label='Medium Volatility (Autumn)')  # Cam nhạt
    
    # Summer - Lowest volatility - màu nhạt nhất
    plt.fill_between(df_2024[summer_mask].index, 
                    df_2024[summer_mask]['temp'] - df_2024[summer_mask]['temp_rolling_std_5'], 
                    df_2024[summer_mask]['temp'] + df_2024[summer_mask]['temp_rolling_std_5'],
                    alpha=0.4, color='#FFE0B2', label='Low Volatility (Summer)')  # Cam rất nhạt
    
    
    # Vẽ đường nhiệt độ
    plt.plot(df_2024.index, df_2024['temp'], color='darkred', linewidth=2, 
             label='Daily Temperature', marker='o', markersize=2, alpha=0.8)
    
    # Đánh dấu các điểm volatility cao nhất
    high_vol_threshold = df_2024['temp_rolling_std_5'].quantile(0.95)
    high_vol_points = df_2024[df_2024['temp_rolling_std_5'] > high_vol_threshold]
    plt.scatter(high_vol_points.index, high_vol_points['temp'], 
                color='darkblue', s=30, zorder=5, label='Peak Volatility Days')
    
    # Customize biểu đồ
    # plt.ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.title('Winter Shows Highest Volatility (2024)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Đặt legend bên ngoài
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Thêm text box với key insights - đặt ngay dưới legend
    insights_text = """KEY INSIGHTS:
• WINTER (Jan-Mar): Highest volatility
  - Frequent cold fronts & temperature swings
  - Hardest to forecast accurately
  
• SUMMER (Jun-Aug): Most stable period
  - Consistent high temperatures
  - Easiest for 5-day forecasting
  
• SPRING/AUTUMN: Moderate volatility
  - Transition seasons with mixed patterns
  
• ⚠️ PEAK VOLATILITY: 
  - Sudden weather events like 'Gio Mua Dong Bac'"""
    
    
    plt.figtext(1.05, 0.7, insights_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.65", facecolor="lightyellow", alpha=0.9),
                verticalalignment='top',
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()


def plot_yearly_temperature(df):
    """
    Plot yearly temperature trend for Hanoi with detailed annotations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'datetime' and 'temp' columns
    """
    # Prepare data
    df["datetime"] = pd.to_datetime(df["datetime"])
    yearly_temp = df.groupby(df['datetime'].dt.year)['temp'].mean()

    plt.figure(figsize=(14, 7))

    # Plot main line
    plt.plot(yearly_temp.index, yearly_temp.values, 
             linewidth=3, color='#2E86AB', marker='o', 
             markersize=8, markerfacecolor='#A23B72', 
             markeredgecolor='white', markeredgewidth=2,
             label='Yearly Average Temperature')

    # Find hottest year and significant dip year (2022)
    hottest_year = yearly_temp.idxmax()
    hottest_temp = yearly_temp.max()
    dip_year = 2022
    dip_temp = yearly_temp.loc[dip_year]

    # Highlight hottest year
    plt.annotate(f'HOTTEST: {hottest_temp:.1f}°C', 
                 xy=(hottest_year, hottest_temp), 
                 xytext=(hottest_year-0.5, hottest_temp+0.3),
                 fontsize=11, fontweight='bold', color='#F24236',
                 arrowprops=dict(arrowstyle='->', color='#F24236', lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE5E5', edgecolor='#F24236'))

    # Highlight significant dip year (2022)
    plt.annotate(f'SHARP DROP\n{dip_temp:.1f}°C', 
                 xy=(dip_year, dip_temp), 
                 xytext=(dip_year-0.8, dip_temp-0.5),
                 fontsize=10, fontweight='bold', color='#5D6D7E',
                 arrowprops=dict(arrowstyle='->', color='#5D6D7E', lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#F8F9F9', edgecolor='#5D6D7E'))

    # Fill area under the line
    plt.fill_between(yearly_temp.index, yearly_temp.values, alpha=0.2, color='#2E86AB')

    # Customize title and labels
    plt.title("HANOI YEARLY AVERAGE TEMPERATURE TREND (2015-2025)", 
              fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

    # X-axis (không rotate)
    plt.xlabel("YEAR", fontsize=12, fontweight='bold', color='#2C3E50', labelpad=10)
    plt.xticks(range(2015, 2026), fontsize=11)
    plt.xlim(2014.5, 2025.5)

    # Y-axis
    plt.ylim(23.2, 25.8)
    plt.yticks(np.arange(23.5, 26.0, 0.5), fontsize=11)

    # Background
    plt.gca().set_facecolor('#FDFEFE')

    # Add trend line to show clear trend
    z = np.polyfit(yearly_temp.index, yearly_temp.values, 1)
    p = np.poly1d(z)
    plt.plot(yearly_temp.index, p(yearly_temp.index), "--", 
             color="#E74C3C", alpha=0.7, linewidth=2, 
             label=f'Trend Line')

    # Add analysis text box (KEY INSIGHTS)
    analysis_text = """KEY INSIGHTS:
• 2019: Hottest year (25.3°C)
• 2022: Significant temperature drop
• Trend: +{:.1f}°C per decade
• 2023-2025: Stabilized at high level""".format(z[0]*10)

    plt.text(0.02, 0.98, analysis_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8, edgecolor='#3498DB'))

    # Add legend ngay dưới KEY INSIGHTS
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.8), fontsize=11, framealpha=0.9)

    # Display values on each point
    for year, temp in yearly_temp.items():
        plt.text(year, temp-0.15, f'{temp:.1f}°C', 
                 ha='center', va='top', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.show()

def get_temperature_statistics(df):
    """
    Get detailed temperature statistics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'datetime' and 'temp' columns
        
    Returns:
    --------
    dict : Dictionary containing temperature statistics
    """
    df["datetime"] = pd.to_datetime(df["datetime"])
    yearly_temp = df.groupby(df['datetime'].dt.year)['temp'].mean()
    
    z = np.polyfit(yearly_temp.index, yearly_temp.values, 1)
    trend_per_decade = z[0] * 10
    
    stats = {
        'yearly_temperatures': yearly_temp.to_dict(),
        'hottest_year': yearly_temp.idxmax(),
        'hottest_temp': yearly_temp.max(),
        'coolest_year': yearly_temp.idxmin(),
        'coolest_temp': yearly_temp.min(),
        'trend_per_decade': trend_per_decade,
        'average_temperature': yearly_temp.mean()
    }
    
    return stats


def create_data_types_table(df):
    """
    Create a styled table for data types overview
    """
    # Analyze data types
    dtype_counts = df.dtypes.value_counts()
    dtype_details = []
    
    for dtype, count in dtype_counts.items():
        if 'datetime' in str(dtype):
            dtype_details.append({'Data Type': 'datetime', 'Count': count, 'Examples': 'DateTime index'})
        elif 'float' in str(dtype) or 'int' in str(dtype):
            examples = df.select_dtypes(include=[dtype]).columns[:3].tolist()
            dtype_details.append({'Data Type': 'numeric', 'Count': count, 'Examples': ', '.join(examples)})
        elif 'object' in str(dtype):
            examples = df.select_dtypes(include=[dtype]).columns[:3].tolist()
            dtype_details.append({'Data Type': 'object', 'Count': count, 'Examples': ', '.join(examples)})
        else:
            examples = df.select_dtypes(include=[dtype]).columns[:3].tolist()
            dtype_details.append({'Data Type': str(dtype), 'Count': count, 'Examples': ', '.join(examples)})
    
    # Create DataFrame
    dtype_df = pd.DataFrame(dtype_details)
    
    # Apply styling
    styled_dtype_table = dtype_df.style\
        .set_properties(**{
            'border': '1px solid #dee2e6',
            'text-align': 'left',
            'padding': '8px'
        })\
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#6c757d'), 
                                       ('color', 'white'),
                                       ('font-weight', 'bold'),
                                       ('padding', '12px')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#f8f9fa')]}
        ])\
        .set_caption('<h3>📋 Data Types Overview</h3>')\
        .hide(axis='index')
    
    return styled_dtype_table

def create_detailed_data_types_table(df):
    """
    Create a more detailed table showing each column's data type
    """
    dtype_info = []
    
    for col in df.columns:
        dtype = df[col].dtype
        if 'datetime' in str(dtype):
            category = 'datetime'
        elif 'float' in str(dtype) or 'int' in str(dtype):
            category = 'numeric'
        elif 'object' in str(dtype):
            category = 'object'
        else:
            category = str(dtype)
            
        dtype_info.append({
            'Column': col,
            'Data Type': category,
            'Specific Type': str(dtype),
            'Null Count': df[col].isnull().sum(),
            'Unique Values': df[col].nunique()
        })
    
    detailed_dtype_df = pd.DataFrame(dtype_info)
    
    # Color coding for data types
    def color_data_type(val):
        if val == 'datetime':
            return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
        elif val == 'numeric':
            return 'background-color: #e3f2fd; color: #1565c0; font-weight: bold'
        elif val == 'object':
            return 'background-color: #f3e5f5; color: #7b1fa2; font-weight: bold'
        else:
            return 'background-color: #fff3e0; color: #ef6c00; font-weight: bold'
    
    styled_detailed_table = detailed_dtype_df.style\
        .applymap(color_data_type, subset=['Data Type'])\
        .background_gradient(subset=['Null Count'], cmap='Reds')\
        .background_gradient(subset=['Unique Values'], cmap='Blues')\
        .set_properties(**{
            'border': '1px solid #dee2e6',
            'text-align': 'left',
            'padding': '8px'
        })\
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#495057'), 
                                       ('color', 'white'),
                                       ('font-weight', 'bold'),
                                       ('padding', '12px')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#f8f9fa')]}
        ])\
        .set_caption('<h3>📋 Detailed Data Types Information</h3>')
    
    return styled_detailed_table