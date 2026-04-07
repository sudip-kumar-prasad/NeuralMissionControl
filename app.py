import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from engine import NeuralNetwork, Layer, generate_data
import time

# --- Setup & Branding ---
st.set_page_config(page_title="Neural Mission Control", layout="wide", initial_sidebar_state="expanded")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Custom Metric Component
def ui_metric(label, value, color='cyan'):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{label}</div>
            <div class="metric-value" style="color: var(--{color}-glow)">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar Configuration (Mission Briefing) ---
st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 1.5rem; margin-bottom: 0;">NEURAL ENGINE</h1>
        <div style="font-size: 0.7rem; color: #94a3b8; letter-spacing: 0.2rem;">VERSION 2.0 // PRO</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("ENV_CONFIG")
dataset_type = st.sidebar.selectbox("Target Pattern", ["moons", "circles", "blobs"])
noise = st.sidebar.slider("Signal Noise", 0.05, 0.5, 0.1)

st.sidebar.subheader("NODE_CONFIG")
n_hidden_layers = st.sidebar.number_input("Core Depth", 1, 5, 2)
neurons_per_layer = st.sidebar.slider("Synapse Width", 2, 16, 8)
activation = st.sidebar.selectbox("Transfer Function", ["relu", "sigmoid", "tanh"])
learning_rate = st.sidebar.select_slider("Learning Loop Rate", options=[0.001, 0.01, 0.1, 0.5], value=0.1)

# Initialize session state for the model and data
if 'model' not in st.session_state or st.sidebar.button("INIT_SYSTEM"):
    X, y = generate_data(dataset_type, noise=noise)
    st.session_state.X = X
    st.session_state.y = y
    
    config = [(2, activation)] # Input
    for _ in range(n_hidden_layers):
        config.append((neurons_per_layer, activation))
    config.append((1, 'sigmoid')) # Output
    
    st.session_state.model = NeuralNetwork(config, learning_rate=learning_rate)
    st.session_state.history = []
    st.session_state.epoch = 0

# --- Top Banner Area ---
t1, t2, t3, t4 = st.columns(4)
with t1: ui_metric("STATUS", "OPERATIONAL" if st.session_state.epoch == 0 else "OPTIMIZING", 'cyan')
with t2: ui_metric("EPOCH", f"{st.session_state.epoch:04d}", 'magenta')
with t3: ui_metric("ERROR_RATE", f"{st.session_state.history[-1]:.4f}" if st.session_state.history else "N/A", 'cyan')
with t4: ui_metric("NODES", f"{sum([l.weights.size for l in st.session_state.model.layers])}", 'magenta')

tabs = st.tabs([
    "01_SYNAPSE", 
    "02_TOPOLOGY", 
    "03_PROPAGATION", 
    "04_GRADIENTS", 
    "05_OPTIMIZER"
])

# Helpers for Plotly
CYAN = "#22d3ee"
MAGENTA = "#f472b6"

# --- Tab 1: Neuron Computation ---
with tabs[0]:
    st.markdown("### SYMBOLIC COMPUTATION ENGINE")
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        test_z = st.slider("Input Signal (z)", -10.0, 10.0, 2.0)
        z_vals = np.linspace(-10, 10, 100)
        if activation == 'relu':
            a_vals = np.maximum(0, z_vals)
            current_a = max(0, test_z)
        elif activation == 'sigmoid':
            a_vals = 1 / (1 + np.exp(-z_vals))
            current_a = 1 / (1 + np.exp(-test_z))
        else: # tanh
            a_vals = np.tanh(z_vals)
            current_a = np.tanh(test_z)
            
        fig_act = go.Figure()
        fig_act.add_trace(go.Scatter(x=z_vals, y=a_vals, name='f(z)', line=dict(color=CYAN, width=3)))
        fig_act.add_trace(go.Scatter(x=[test_z], y=[current_a], mode='markers', marker=dict(size=15, color='#fff', line=dict(color=CYAN, width=2))))
        fig_act.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              xaxis=dict(gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig_act, use_container_width=True)

    with col2:
        st.markdown(f"""
        #### MATHEMATICAL MODELING
        ```python
        z = np.dot(x, weights) + bias
        activation = {activation}(z)
        ```
        The non-linear transfer allows the network to approximate any continuous function given enough width.
        """)

# --- Tab 2: Architecture ---
with tabs[1]:
    st.markdown("### SYSTEM TOPOLOGY MAP")
    
    def plot_network(model):
        layers = model.layers
        nodes = [[{'y': i, 'x': 0} for i in range(2)]]
        for i, layer in enumerate(layers):
            nodes.append([{'y': j - (layer.weights.shape[1]-1)/2, 'x': i+1} for j in range(layer.weights.shape[1])])
            
        fig_net = go.Figure()
        for idx in range(len(nodes)-1):
            for i, n1 in enumerate(nodes[idx]):
                for j, n2 in enumerate(nodes[idx+1]):
                    w = model.layers[idx].weights[i, j]
                    color = CYAN if w > 0 else MAGENTA
                    fig_net.add_trace(go.Scatter(x=[n1['x'], n2['x']], y=[n1['y'], n2['y']], 
                                                 mode='lines', line=dict(width=abs(w)*3 + 0.2, color=color), 
                                                 hoverinfo='none', opacity=0.3, showlegend=False))
        
        for layer_nodes in nodes:
            fig_net.add_trace(go.Scatter(x=[n['x'] for n in layer_nodes], y=[n['y'] for n in layer_nodes], 
                                         mode='markers', marker=dict(size=18, color='#020617', line=dict(color='#94a3b8', width=2))))
            
        fig_net.update_layout(template="plotly_dark", showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              margin=dict(l=0,r=0,t=0,b=0), height=500)
        return fig_net

    st.plotly_chart(plot_network(st.session_state.model), use_container_width=True)

# --- Tab 3: Forward Pass ---
with tabs[2]:
    st.markdown("### SPATIAL REPRESENTATION FIELD")
    cols = st.columns([1.5, 1])
    
    with cols[0]:
        x_min, x_max = st.session_state.X[:, 0].min() - 0.5, st.session_state.X[:, 0].max() + 0.5
        y_min, y_max = st.session_state.X[:, 1].min() - 0.5, st.session_state.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        Z = st.session_state.model.forward(grid)
        Z = Z.reshape(xx.shape)
        
        fig_db = go.Figure(data=[
            go.Contour(z=Z, x=np.linspace(x_min, x_max, 60), y=np.linspace(y_min, y_max, 60), 
                       colorscale=[[0, MAGENTA], [0.5, '#020617'], [1, CYAN]], opacity=0.6, showscale=False),
            go.Scatter(x=st.session_state.X[:,0], y=st.session_state.X[:,1], mode='markers', 
                       marker=dict(color=st.session_state.y.flatten(), colorscale=[[0, MAGENTA], [1, CYAN]], line=dict(width=1, color='#fff')))
        ])
        fig_db.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig_db, use_container_width=True)
    
    with cols[1]:
        ui_metric("SYSTEM_ENTROPY", f"{np.std(Z):.4f}", 'cyan')
        st.markdown("""
        #### FIELD INTERPRETATION
        The system projects 2D space into a probability density. Areas in **Cyan** represent Class A, **Magenta** Class B. 
        Dark regions indicate uncertainty.
        """)

# --- Tab 4: Loss & Backprop ---
with tabs[3]:
    st.markdown("### GRADIENT DESCENT MANIFOLD")
    st.latex(r"W_{new} = W_{old} - \eta \cdot \nabla_{W} \mathcal{L}")
    
    st.markdown("""
    Backpropagation calculates the "Partial Derivatives" layer by layer. 
    Think of it as propagating the 'Blame' for the total error backwards to each individual neuron.
    """)
    
    if st.button("EXECUTE SINGLE UPDATE LOOP"):
        with st.spinner("CALCULATING GRADIENTS..."):
            loss = st.session_state.model.train_step(st.session_state.X, st.session_state.y)
            st.session_state.history.append(loss)
            st.session_state.epoch += 1
            st.success(f"DIVERGENCE REDUCED | LOSS: {loss:.6f}")

# --- Tab 5: Optimizer ---
with tabs[4]:
    st.markdown("### LIVE TRAINING PIPELINE")
    
    col_ctrl, col_plot = st.columns([1, 2.5])
    
    with col_ctrl:
        n_epochs = st.number_input("ITERATION_BATCH", 10, 1000, 100)
        train_btn = st.button("RUN GLOBAL OPTIMIZER", use_container_width=True)
        
        if train_btn:
            status_placeholder = st.empty()
            loss_placeholder = st.empty()
            
            # Simple Loop for Live Monitoring
            for epoch in range(n_epochs):
                loss = st.session_state.model.train_step(st.session_state.X, st.session_state.y)
                st.session_state.history.append(loss)
                st.session_state.epoch += 1
                
                if epoch % 5 == 0:
                    # Update History Plot
                    fig_h = px.line(y=st.session_state.history[-200:], 
                                    title="SIGNAL CONVERGENCE (RECENT 200)", 
                                    color_discrete_sequence=[CYAN])
                    fig_h.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=40,b=0), 
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
                    loss_placeholder.plotly_chart(fig_h, use_container_width=True)
                    status_placeholder.markdown(f"**CONVERGING...** EPOCH: {st.session_state.epoch} | LOSS: {loss:.5f}")

    with col_plot:
        Z = st.session_state.model.forward(grid).reshape(xx.shape)
        fig_res = go.Figure(data=[
            go.Contour(z=Z, x=np.linspace(x_min, x_max, 60), y=np.linspace(y_min, y_max, 60), 
                       colorscale=[[0, MAGENTA], [0.5, '#020617'], [1, CYAN]], opacity=0.7, showscale=False),
            go.Scatter(x=st.session_state.X[:,0], y=st.session_state.X[:,1], mode='markers', 
                       marker=dict(color=st.session_state.y.flatten(), colorscale=[[0, MAGENTA], [1, CYAN]], line=dict(width=1, color='#fff')))
        ])
        fig_res.update_layout(title="OPTIMIZED PATTERN RECOGNITION", template="plotly_dark", 
                              margin=dict(l=0,r=0,t=40,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=550)
        st.plotly_chart(fig_res, use_container_width=True)
