import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go

# Wczytanie pliku
df = pd.read_csv('PREPARE DATA/gotowe_csv/k.csv', nrows=1)
columns = df.columns.tolist()
values = df.iloc[0].tolist()

# Punkty
points = []
for i in range(21):
    x = float(values[3 + i * 3])
    y = float(values[3 + i * 3 + 1])
    z = float(values[3 + i * 3 + 2])
    points.append([x, y, z])
points = np.array(points)

# Wektory
vectors = []
vector_origins = []
vector_pattern = re.compile(r'vec_(\d+)-(\d+)_vx')

for i in range(66, len(columns), 3):
    vec_name = columns[i]
    match = vector_pattern.match(vec_name)
    if match:
        start_idx = int(match.group(1))
        vec = [float(values[i]), float(values[i + 1]), float(values[i + 2])]
        vector_origins.append(start_idx)
        vectors.append(vec)

# Plotly wykres
fig = go.Figure()

# Punkty
fig.add_trace(go.Scatter3d(
    x=points[:, 0], y=points[:, 1], z=points[:, 2],
    mode='markers+text',
    marker=dict(size=4, color='blue'),
    text=[f'lm_{i}' for i in range(21)],
    name='Punkty'
))

# Wektory
for origin_idx, vec in zip(vector_origins, vectors):
    origin = points[origin_idx]
    end = origin + vec
    fig.add_trace(go.Scatter3d(
        x=[origin[0], end[0]],
        y=[origin[1], end[1]],
        z=[origin[2], end[2]],
        mode='lines',
        line=dict(color='red', width=4),
        name=f'wektor z lm_{origin_idx}'
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
    ),
    title='Punkty i wektory 3D (interaktywne)',
    showlegend=False
)

fig.show()