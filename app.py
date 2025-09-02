import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
from collections import defaultdict
import os

# Initialize the Dash app
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"
]
BASE_PATH = os.getenv('APP_BASE_PATH', '/')
if not BASE_PATH.endswith('/'):
    BASE_PATH = BASE_PATH + '/'
path_kwargs = {}
if BASE_PATH not in (/,):
    path_kwargs = dict(requests_pathname_prefix=BASE_PATH, routes_pathname_prefix=BASE_PATH)
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    **path_kwargs
)
server = app.server

# Color palette for requirements
COLORS = {
    'Design Core - Foundations': '#FF6B6B',
    'Design Core - Reflection': '#4ECDC4',
    'Methods - Emerging Technologies': '#45B7D1',
    'Methods - Human Behavior': '#96CEB4',
    'Methods - Physical': '#FFEAA7',
    'Domain Focus Area': '#DDA0DD',
    'Capstone': '#FFB347',
    'Other': '#95A5A6'
}

def _hex_to_rgb(hex_color: str):
    s = hex_color.lstrip('#')
    if len(s) == 3:
        s = ''.join([c*2 for c in s])
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return r, g, b
    except Exception:
        return 0, 0, 0

def pick_text_color(bg_hex: str) -> str:
    """Return '#FFFFFF' or '#000000' depending on background brightness (YIQ)."""
    r, g, b = _hex_to_rgb(bg_hex)
    yiq = (r * 299 + g * 587 + b * 114) / 1000
    return '#000000' if yiq >= 150 else '#FFFFFF'

DAY_ABBREV = {
    'Mon': 'Mon', 'Tue': 'Tue', 'Wed': 'Wed', 'Thu': 'Thu', 'Fri': 'Fri', 'Sat': 'Sat', 'Sun': 'Sun',
    'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun',
    'Thurs': 'Thu', 'Thur': 'Thu', 'Thu.': 'Thu'
}
DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def _zero_pad_time_component(val):
    try:
        return f"{int(val):02d}"
    except Exception:
        return "00"

def normalize_time_string(time_str: str) -> str:
    """Normalize various time formats to HH:MM 24h (e.g., '9:30' -> '09:30', '1:20 PM' -> '13:20')."""
    if pd.isna(time_str):
        return '00:00'
    s = str(time_str).strip()
    if s == '':
        return '00:00'
    # Handle AM/PM
    lower = s.lower().replace(' ', '')
    ampm = None
    if lower.endswith('am'):
        ampm = 'am'
        lower = lower[:-2]
    elif lower.endswith('pm'):
        ampm = 'pm'
        lower = lower[:-2]
    # Split hour/minute
    if ':' in lower:
        h, m = lower.split(':', 1)
    else:
        # e.g., '900' -> 9:00
        if len(lower) in (3, 4) and lower.isdigit():
            h = lower[:-2]
            m = lower[-2:]
        else:
            # Fallback
            return '00:00'
    hour = int(h) if h.isdigit() else 0
    # Remove any trailing non-digits in minutes
    m_digits = ''.join([c for c in m if c.isdigit()])
    minute = int(m_digits) if m_digits.isdigit() else 0
    if ampm == 'am':
        if hour == 12:
            hour = 0
    elif ampm == 'pm':
        if hour != 12:
            hour += 12
    return f"{hour:02d}:{minute:02d}"

def to_minutes(time_str: str) -> int:
    try:
        parts = str(time_str).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0

def load_and_normalize_data(upload_contents: str = None) -> pd.DataFrame:
    """Load from upload, course_conflicts_with_dates.csv, or sample.csv and normalize to canonical columns.
    Canonical columns: course, title, quarter, day, start_time, end_time, requirement, notes, course_id
    """
    df = None
    if upload_contents is not None:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception:
            df = None
    if df is None:
        # Prefer the curated conflicts file if present
        try:
            df = pd.read_csv('course_conflicts_with_dates.csv')
        except Exception:
            try:
                df = pd.read_csv('sample.csv')
            except Exception:
                df = pd.DataFrame({
                    'course': ['SAMPLE 101'],
                    'title': ['Sample Course'],
                    'quarter': ['Sample Quarter'],
                    'day': ['Mon'],
                    'start_time': ['09:00'],
                    'end_time': ['10:30'],
                    'requirement': ['Other'],
                    'notes': ['Demo row']
                })

    columns = set(df.columns)
    # Detect schema variant and normalize
    if {'Quarter', 'Day', 'Start Time', 'End Time'}.issubset(columns):
        # course_conflicts_with_dates.csv style
        def first_requirement(val: str) -> str:
            if pd.isna(val) or str(val).strip() == '':
                return 'Other'
            s = str(val)
            # Some rows have multiple categories joined with '/'
            return s.split('/')[0].strip()

        norm = pd.DataFrame({
            'course': df.get('Course Code', pd.Series([''] * len(df))).fillna('').astype(str).str.strip(),
            'title': df.get('Title', pd.Series([''] * len(df))).fillna('').astype(str).str.strip(),
            'quarter': df['Quarter'].fillna('').astype(str).str.strip(),
            'day': df['Day'].fillna('').astype(str).map(lambda x: DAY_ABBREV.get(x.strip(), x.strip()[:3])).map(lambda x: 'Thu' if x in ['Th', 'Thur', 'Thurs'] else x),
            'start_time': df['Start Time'].map(normalize_time_string),
            'end_time': df['End Time'].map(normalize_time_string),
            'requirement': df.get('Requirement', pd.Series(['Other'] * len(df))).map(first_requirement),
            'notes': df.get('Meeting Days/Times', pd.Series([''] * len(df))).fillna('').astype(str)
        })
        # If course is empty, fall back to Title
        norm['course'] = norm.apply(lambda r: r['course'] if r['course'] else (r['title'] or 'Unknown Course'), axis=1)
    else:
        # Expect canonical style columns already
        # Ensure required columns exist
        for col in ['course', 'title', 'quarter', 'day', 'start_time', 'end_time', 'requirement', 'notes']:
            if col not in columns:
                df[col] = ''
        norm = df[['course', 'title', 'quarter', 'day', 'start_time', 'end_time', 'requirement', 'notes']].copy()
        norm['day'] = norm['day'].map(lambda x: DAY_ABBREV.get(str(x), str(x)[:3]))
        norm['start_time'] = norm['start_time'].map(normalize_time_string)
        norm['end_time'] = norm['end_time'].map(normalize_time_string)
        norm['requirement'] = norm['requirement'].fillna('Other').replace('', 'Other')

    # Derive minutes and ids
    norm['start_minutes'] = norm['start_time'].map(to_minutes)
    norm['end_minutes'] = norm['end_time'].map(to_minutes)
    norm['course_id'] = (
        norm['quarter'].astype(str) + '|' +
        norm['day'].astype(str) + '|' +
        norm['course'].astype(str) + '|' +
        norm['start_time'].astype(str) + '-' + norm['end_time'].astype(str)
    )
    # Drop bogus rows
    norm = norm[(norm['day'].isin(DAY_ORDER)) & (norm['end_minutes'] > norm['start_minutes'])]
    return norm.reset_index(drop=True)

def assign_overlap_lanes(day_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a lane (y position) to each class to stack overlapping ones."""
    if day_df.empty:
        day_df['lane'] = []
        return day_df
    records = []
    lanes_end = []  # last end time per lane
    for _, row in day_df.sort_values('start_minutes').iterrows():
        placed = False
        for lane_index, last_end in enumerate(lanes_end):
            if row['start_minutes'] >= last_end:
                lanes_end[lane_index] = row['end_minutes']
                records.append({**row.to_dict(), 'lane': lane_index})
                placed = True
                break
        if not placed:
            lanes_end.append(row['end_minutes'])
            records.append({**row.to_dict(), 'lane': len(lanes_end) - 1})
    return pd.DataFrame(records)

def build_weekly_figure(df: pd.DataFrame, selected_quarter: str, requirement_filter: list, search_text: str, hour_range: list, color_by_requirement: bool, selected_ids: set) -> go.Figure:
    quarter_df = df[df['quarter'] == selected_quarter].copy() if selected_quarter else df.copy()
    if requirement_filter:
        quarter_df = quarter_df[quarter_df['requirement'].isin(requirement_filter)]
    if search_text:
        mask = (
            quarter_df['course'].str.contains(search_text, case=False, na=False) |
            quarter_df['title'].str.contains(search_text, case=False, na=False)
        )
        quarter_df = quarter_df[mask]
    if quarter_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No courses match the current filters", x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        return fig

    # Hour range filtering
    start_bound = (hour_range[0] if hour_range else 8) * 60
    end_bound = (hour_range[1] if hour_range else 18) * 60
    display_df = quarter_df[(quarter_df['end_minutes'] > start_bound) & (quarter_df['start_minutes'] < end_bound)].copy()
    if display_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No courses within selected hours", x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        return fig

    # Clamp to hour bounds for drawing
    display_df['draw_start'] = display_df['start_minutes'].clip(lower=start_bound)
    display_df['draw_end'] = display_df['end_minutes'].clip(upper=end_bound)

    # Days as Y categories using numeric indices for precise stacking
    days_present = [d for d in DAY_ORDER if d in display_df['day'].unique()]
    if not days_present:
        days_present = DAY_ORDER
    day_to_index = {d: i for i, d in enumerate(days_present)}

    fig = go.Figure()
    lane_spacing = 0.28
    jitter_offset = 0.12
    max_y_pos = -0.5

    # Build horizontal blocks per day with overlap-aware lanes
    for day in days_present:
        day_courses = display_df[display_df['day'] == day].copy()
        if day_courses.empty:
            continue
        day_courses = assign_overlap_lanes(day_courses)
        base = day_to_index[day]
        for _, c in day_courses.iterrows():
            y_pos = base + jitter_offset + (c['lane'] * lane_spacing)
            max_y_pos = max(max_y_pos, y_pos)
            color = COLORS.get(c['requirement'], COLORS['Other']) if color_by_requirement else '#3498DB'
            is_selected = c['course_id'] in selected_ids
            width = 36 if is_selected else 28
            fig.add_trace(
                go.Scatter(
                    x=[max(start_bound, c['draw_start']), min(end_bound, c['draw_end'])],
                    y=[y_pos, y_pos],
                    mode='lines',
                    line=dict(color=color, width=width),
                    name=c['course'],
                    customdata=[[c['course_id'], c['course'], c['title'], c['requirement'], c['start_time'], c['end_time'], c['day']]],
                    hovertemplate=(
                        f"<b>{c['course']}</b><br>" +
                        (f"{c['title']}<br>" if c['title'] else '') +
                        f"{day} {c['start_time']} – {c['end_time']}<br>" +
                        f"Requirement: {c['requirement']}<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=False
                )
            )
            # Label
            mid_x = (max(start_bound, c['draw_start']) + min(end_bound, c['draw_end'])) / 2
            text_color = pick_text_color(color)
            anno_bg = 'rgba(0,0,0,0.32)' if text_color == '#FFFFFF' else 'rgba(255,255,255,0.28)'
            fig.add_annotation(
                x=mid_x,
                y=y_pos,
                text=c['course'],
                showarrow=False,
                font=dict(size=12 if is_selected else 11, color=text_color),
                bgcolor=anno_bg,
                borderpad=2
            )

    # Dashed separators between day rows + minor vertical 30-min gridlines
    shapes = []
    for i in range(len(days_present) - 1):
        y_sep = i + 0.5
        shapes.append(dict(
            type='line',
            x0=start_bound, x1=end_bound,
            y0=y_sep, y1=y_sep,
            line=dict(color='#3a3f45', width=1, dash='dash')
        ))
    # Minor vertical grid at 30-minute intervals (no labels)
    y_top = max(max_y_pos + 0.6, len(days_present) - 0.4)
    for t in range(start_bound + 30 - (start_bound % 30 or 30), end_bound, 30):
        if t % 60 == 0:
            continue
        shapes.append(dict(
            type='line',
            x0=t, x1=t,
            y0=-0.5, y1=y_top,
            line=dict(color='#252a31', width=1, dash='dot'),
            layer='below'
        ))

    # Axes configuration: X=time, Y=days
    x_ticks = list(range(start_bound - (start_bound % 60), end_bound + 1, 60))
    x_labels = [f"{m // 60:02d}:00" for m in x_ticks]
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_ticks,
        ticktext=x_labels,
        range=[start_bound - 10, end_bound + 10],
        title_text='Time',
        side='top',
        showgrid=True,
        gridcolor='#333',
        tickfont=dict(color='#f0f0f0'),
        title_font=dict(color='#f0f0f0')
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=[day_to_index[d] for d in days_present],
        ticktext=days_present,
        range=[max(max_y_pos + 0.6, len(days_present) - 0.4), -0.5],
        title_text='Day',
        showgrid=False,
        tickfont=dict(color='#f0f0f0'),
        title_font=dict(color='#f0f0f0')
    )

    fig.update_layout(
        title=dict(text=f"{selected_quarter or ''}", x=0.5, font=dict(color='#f0f0f0')),
        height=720,
        plot_bgcolor='#0b0f14',
        paper_bgcolor='#0b0f14',
        margin=dict(l=40, r=40, t=130, b=40),
        font=dict(color='#f0f0f0'),
        hoverlabel=dict(bgcolor='#1c1f24', font_color='#f0f0f0', font_size=12),
        shapes=shapes
    )
    return fig

def compute_plan_summary(df: pd.DataFrame, selected_ids: set) -> dict:
    sel = df[df['course_id'].isin(selected_ids)].copy()
    if sel.empty:
        return {'requirements': {}, 'conflicts': [], 'by_quarter': {}}
    # Requirement coverage
    req_counts = sel['requirement'].value_counts().to_dict()
    # Conflicts within same quarter and day
    conflicts = []
    for (q, d), group in sel.groupby(['quarter', 'day']):
        if len(group) <= 1:
            continue
        g = group.sort_values('start_minutes')
        active = []
        for _, r in g.iterrows():
            # Remove ended
            active = [a for a in active if a['end_minutes'] > r['start_minutes']]
            for a in active:
                if r['start_minutes'] < a['end_minutes'] and r['end_minutes'] > a['start_minutes']:
                    conflicts.append({
                        'quarter': q,
                        'day': d,
                        'a': a['course'],
                        'b': r['course'],
                        'time': f"{max(a['start_time'], r['start_time'])}–{min(a['end_time'], r['end_time'])}"
                    })
            active.append(r)
    by_quarter = defaultdict(list)
    for _, r in sel.iterrows():
        by_quarter[r['quarter']].append(f"{r['course']} ({r['day']} {r['start_time']}–{r['end_time']})")
    return {'requirements': req_counts, 'conflicts': conflicts, 'by_quarter': by_quarter}

# Layout
app.layout = html.Div([
    html.H2("Pick Classes — See Conflicts & Requirements", style={'textAlign': 'center', 'marginBottom': 10}),

    # Controls row
    html.Div([
        html.Div([
            html.Label("Quarter"),
            dcc.Dropdown(id='quarter-select', placeholder='Select quarter...', style={'minHeight': '40px'})
        ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),
        html.Div([
            html.Label("Requirements"),
            dcc.Dropdown(id='requirement-filter', multi=True, placeholder='Filter by requirements...', style={'minHeight': '40px'})
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),
        html.Div([
            html.Label("Search"),
            dcc.Input(id='search-text', type='text', placeholder='Course code/title...', style={'width': '100%'})
        ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),
        html.Div([
            html.Label("Hours"),
            dcc.RangeSlider(id='hour-range', min=7, max=21, value=[9, 18], marks={i: f"{i:02d}" for i in range(7, 22, 2)}, tooltip={"placement": "bottom", "always_visible": True})
        ], style={'width': '22%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'margin': '10px 20px'}),

    # Legend & color toggle
    html.Div([
        dcc.Checklist(id='color-toggle', options=[{'label': ' Color by requirement', 'value': 'color'}], value=['color'], style={'display': 'inline-block', 'marginRight': '10px'}),
        html.Span("Legend:", style={'fontWeight': 'bold', 'marginRight': '8px'}),
        html.Div(children=[
            html.Span([
                html.Span(style={'display': 'inline-block', 'width': 10, 'height': 10, 'backgroundColor': color, 'marginRight': 6, 'borderRadius': 6, 'boxShadow': f'0 0 0 2px {color}33'}),
                html.Span(req)
            ], style={'display': 'inline-block', 'padding': '4px 8px', 'borderRadius': '12px', 'backgroundColor': '#141a21', 'border': '1px solid #2a2f36'})
            for req, color in COLORS.items()
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'alignItems': 'center'})
    ], style={'margin': '0 20px 10px 20px'}),

    # Calendar full width
    html.Div([
        dcc.Graph(id='weekly-calendar', style={'height': '720px', 'width': '100%'})
    ], style={'margin': '0 20px'}),

    # Data store (preloaded with default data)
    dcc.Store(id='data-store', data=load_and_normalize_data().to_dict('records')),
], style={'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif', 'backgroundColor': '#0b0f14', 'color': '#f0f0f0'})

# Callbacks
@callback(
    Output('quarter-select', 'options'),
    Output('quarter-select', 'value'),
    Output('requirement-filter', 'options'),
    Output('requirement-filter', 'value'),
    Input('data-store', 'data')
)
def init_data_store(data):
    df = pd.DataFrame(data or [])
    quarter_options = [{'label': q, 'value': q} for q in sorted(df['quarter'].unique())]
    default_quarter = quarter_options[0]['value'] if quarter_options else None
    req_values_sorted = sorted(df['requirement'].dropna().unique())
    req_options = [{'label': r, 'value': r} for r in req_values_sorted]
    return quarter_options, default_quarter, req_options, req_values_sorted

@callback(
    Output('weekly-calendar', 'figure'),
    Input('data-store', 'data'),
    Input('quarter-select', 'value'),
    Input('requirement-filter', 'value'),
    Input('search-text', 'value'),
    Input('hour-range', 'value'),
    Input('color-toggle', 'value')
)
def update_calendar(data, quarter_value, req_filter, search_text, hour_range, color_toggle):
    df = pd.DataFrame(data or [])
    if df.empty:
        return go.Figure()
    color_by_req = 'color' in (color_toggle or [])
    return build_weekly_figure(df, quarter_value, req_filter or [], search_text or '', hour_range or [9, 18], color_by_req, set())

@callback(
    Output('data-store', 'data'),
    Input('weekly-calendar', 'clickData'),
    State('data-store', 'data')
)
def noop(click_data, data):
    # No selection logic needed now; just return the existing data
    return data

# Removed plan-related callbacks; no-op placeholder not needed

# Health check endpoint
@app.server.route('/healthz')
def health_check():
    return 'OK', 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', '8080'))
    app.run(debug=True, host='0.0.0.0', port=port)