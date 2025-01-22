import numpy as np
from flask import Flask, render_template
from openai import OpenAI
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import INLINE
from flask_socketio import SocketIO, emit
from flask import request
import time

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize the OpenAI client
client = OpenAI(api_key="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")

# Store last update timestamp for rate limiting
last_update = {}

@app.route("/")
def index():
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    return render_template("index.html",
                         js_resources=js_resources,
                         css_resources=css_resources)

@socketio.on('stream_update')
def handle_stream_update(data):
    global last_update
    
    # Rate limiting (max 2 requests per second per client)
    client_id = request.sid
    current_time = time.time()
    if client_id in last_update and current_time - last_update[client_id] < 0.5:
        return
    last_update[client_id] = current_time

    text1 = data.get('text1', '').strip()
    text2 = data.get('text2', '').strip()

    if not text1 or not text2:
        emit('plot_update', {
            'status': 'error',
            'message': 'Please enter text in both boxes.'
        })
        return

    try:
        # Get embeddings
        response1 = client.embeddings.create(input=text1, model="text-embedding-3-large")
        response2 = client.embeddings.create(input=text2, model="text-embedding-3-large")

        embedding_1 = np.array(response1.data[0].embedding)
        embedding_2 = np.array(response2.data[0].embedding)
        embedding_diff = embedding_1 - embedding_2

        # Create visualization
        p = figure(title="Real-time Embedding Differences",
                  x_axis_label="Dimension",
                  y_axis_label="Difference",
                  height=400,
                  width=800,
                  sizing_mode="stretch_width")

        # Add bars
        p.vbar(x=np.arange(len(embedding_diff)),
               top=embedding_diff,
               width=1,
               color="#3498db",
               alpha=0.7)

        # Style improvements
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_alpha = 0.1
        p.background_fill_color = "#f8f9fa"

        # Get components
        script, div = components(p)

        # Emit update
        emit('plot_update', {
            'status': 'success',
            'script': script,
            'div': div,
            'stats': {
                'max_diff': float(np.max(embedding_diff)),
                'min_diff': float(np.min(embedding_diff)),
                'mean_diff': float(np.mean(embedding_diff)),
                'std_diff': float(np.std(embedding_diff))
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        emit('plot_update', {
            'status': 'error',
            'message': str(e)
        })

if __name__ == "__main__":
    socketio.run(app, debug=True)