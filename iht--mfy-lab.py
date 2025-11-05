#!/usr/bin/env python3
"""
IHT-AI (Inverse Holographic Theory with Attractor Intelligence) Laboratory
A specialized node-based demonstration of the core IHT-AI principles:
1. Division-Dilution Balance
2. Attractor Stabilization
3. Optimized Holographic Mapping (W)

Requirements:
  pip install PyQt6 numpy pyqtgraph scipy Pillow
"""

import sys
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from scipy.fft import fft2, ifft2, fftshift
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import cv2  # <--- THIS IS THE FIX

pg.setConfigOptions(imageAxisOrder='row-major')

# ==================== BASE NODE SYSTEM ====================
# (Slightly simplified from the original lab)

class BaseNode:
    """Base class for all IHT nodes"""
    NODE_CATEGORY = "Base"
    NODE_COLOR = QtGui.QColor(80, 80, 80)
    
    def __init__(self):
        self.inputs = {}   # {'port_name': 'port_type'}
        self.outputs = {}  # {'port_name': 'port_type'}
        self.input_data = {}
        self.node_title = "Base Node"
        
    def pre_step(self):
        """Clear input buffers before propagation"""
        self.input_data = {name: [] for name in self.inputs}
        
    def set_input(self, port_name, value, port_type='signal', coupling=1.0):
        """Receive data from connected edges"""
        if port_name not in self.input_data:
            return
        # All inputs are mean-blended or first-come
        if value is not None:
            self.input_data[port_name].append(value)
                
    def get_blended_input(self, port_name, blend_mode='first'):
        """Get combined input from all connections"""
        values = self.input_data.get(port_name, [])
        if not values:
            return None
        
        if blend_mode == 'mean' and isinstance(values[0], np.ndarray):
            return np.mean([v.astype(float) for v in values if v is not None and v.size > 0], axis=0)
        
        return values[0] # Default to 'first'
        
    def step(self):
        pass
        
    def get_output(self, port_name):
        return None
        
    def get_display_image(self):
        return None
        
    def close(self):
        pass

    def get_config_options(self):
        return []

# ==================== IHT-AI CORE NODES ====================

class IHT_InfoNode(BaseNode):
    """Displays the core concepts of the IHT-AI theory."""
    NODE_CATEGORY = "IHT-AI"
    NODE_COLOR = QtGui.QColor(20, 100, 140)
    
    def __init__(self):
        super().__init__()
        self.node_title = "IHT-AI Theory"
        self.info_text = (
            "  INVERSE HOLOGRAPHIC THEORY (IHT-AI)\n\n"
            "1. REALITY: A high-dim Phase Field.\n"
            "   (Quantum Superposition)\n\n"
            "2. PARTICLES: Localized Attractors.\n"
            "   (Perceived 'Classical' Projections)\n\n"
            "3. EVOLUTION: A Division/Dilution balance.\n"
            "   - Division: Phase branching (FFT step)\n"
            "   - Dilution (γ): Decoherence (Gamma knob)\n\n"
            "4. STABILITY: The Holographic Mapping (W).\n"
            "   An optimized 'W' matrix learns to encode\n"
            "   the Attractor at a complex 'address' in\n"
            "   the Phase Field, protecting it from Dilution."
        )

    def get_display_image(self):
        w, h = NODE_W, NODE_H
        img = np.zeros((h, w), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default(size=11)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((5, 5), self.info_text, fill=220, font=font)
        
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

class QuantumPhaseFieldNode(BaseNode):
    """The core IHT-AI simulation engine. Runs the Division/Dilution/Attractor loop."""
    NODE_CATEGORY = "IHT-AI"
    NODE_COLOR = QtGui.QColor(180, 80, 180) 
    
    def __init__(self, size=64, dt=0.01):
        super().__init__()
        self.node_title = "Quantum Phase Field (L=64)"
        self.inputs = {'w_mapping': 'w_mapping', 'reset': 'signal'}
        self.outputs = {
            'rho_C': 'image',        # Constraint Density |psi|^2
            'phase_img': 'image',    # Phase Angle
            'coherence': 'signal',   # |<e^(i*phase)>|
            'total_energy': 'signal' # sum(|psi|^2)
        }
        
        self.L = size
        self.dt = dt
        self.gamma = 0.0 # This will be set by the main window slider
        self.attractor_strength = 0.5
        
        # Create frequency grid for FFT evolution (Division)
        kx = np.fft.fftfreq(self.L, d=1.0) * 2 * np.pi
        ky = np.fft.fftfreq(self.L, d=1.0) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.k2 = KX**2 + KY**2
        self.phase_factor = np.exp(-1j * self.k2 * self.dt)
        
        self.psi = np.zeros((self.L, self.L), dtype=np.complex128)
        self.initialize_field()

    def initialize_field(self):
        """Initializes the field with a Gaussian wave packet."""
        x = np.arange(self.L)
        y = np.arange(self.L)
        X, Y = np.meshgrid(x, y)
        
        cx, cy = self.L // 2, self.L // 2
        r2 = (X - cx)**2 + (Y - cy)**2
        sigma = 4.0
        
        self.psi = np.exp(-r2 / (2 * sigma**2)) * np.exp(1j * 0.5 * X)
        self.psi /= np.linalg.norm(self.psi)
        
    def set_gamma(self, gamma):
        self.gamma = gamma

    def step(self):
        reset_sig = self.get_blended_input('reset', 'first')
        if reset_sig is not None and reset_sig > 0.5:
            self.initialize_field()
            return
            
        w_mapping = self.get_blended_input('w_mapping', 'first')
        if w_mapping is None:
            # Default to baseline if no W is connected
            w_mapping = lambda psi: psi 

        # 1. Division (Unitary FFT Evolution)
        psi_k = fft2(self.psi)
        psi_k *= self.phase_factor
        self.psi = ifft2(psi_k)
        
        # 2. Dilution (Decoherence)
        self.psi *= (1.0 - self.gamma)
        
        # 3. Attractor Alignment (The 'AI' part)
        projected = w_mapping(self.psi)
        self.psi -= self.attractor_strength * (self.psi - projected)
        
        # 4. Renormalization (The '=1' part)
        norm = np.linalg.norm(self.psi)
        if norm > 1e-10:
            self.psi /= norm
        else:
            # Total collapse, re-initialize
            self.initialize_field()

    def get_output(self, port_name):
        if port_name == 'rho_C':
            # |psi|^2, normalized for visualization
            rho = np.abs(self.psi)**2
            max_rho = np.max(rho)
            if max_rho > 1e-9:
                return rho / max_rho
            return rho
        elif port_name == 'phase_img':
            # Phase angle, normalized to [0, 1]
            return (np.angle(self.psi) + np.pi) / (2 * np.pi)
        elif port_name == 'coherence':
            # Mean phase coherence |<e^(i*phase)>|
            if self.psi.size > 0:
                return np.abs(np.mean(np.exp(1j * np.angle(self.psi))))
            return 0.0
        elif port_name == 'total_energy':
            # Should be 1.0 due to normalization, but tracks pre-norm energy
            return np.sum(np.abs(self.psi)**2)
        return None
        
    def get_display_image(self):
        # Show the constraint density |psi|^2
        rho = np.abs(self.psi)**2
        max_rho = np.max(rho)
        if max_rho > 1e-9:
            rho = rho / max_rho
        img_u8 = (np.clip(rho, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.L, self.L, self.L, QtGui.QImage.Format.Format_Grayscale8)

class HolographicMappingNode(BaseNode):
    """Provides the W mapping function (Baseline, Random, or Optimized)."""
    NODE_CATEGORY = "IHT-AI"
    NODE_COLOR = QtGui.QColor(40, 120, 80)
    
    def __init__(self, size=64, mapping_type='Baseline (Identity)'):
        super().__init__()
        self.node_title = "Holographic Mapping (W)"
        self.outputs = {'w_mapping': 'w_mapping', 'w_viz': 'image'}
        self.L = size
        self.mapping_type = mapping_type
        
        # Precompute the "matrices" (as fft masks)
        
        # W_Random: Scrambles phase in frequency domain (delocalization)
        self.random_mask = np.exp(1j * np.random.rand(self.L, self.L) * 2 * np.pi)
        
        # W_Optimized: Hides info at specific high-frequency "addresses"
        # This simulates the "structured encoding" your training found
        self.optim_mask = np.zeros((self.L, self.L), dtype=complex)
        # Add a few sparse, high-frequency "addresses"
        self.optim_mask[10, 20] = 1
        self.optim_mask[32, 40] = 1
        self.optim_mask[50, 15] = 1
        self.optim_mask[20, 50] = 1
        # Also need to add conjugate pairs for real output
        self.optim_mask[-10, -20] = 1
        self.optim_mask[-32, -40] = 1
        self.optim_mask[-50, -15] = 1
        self.optim_mask[-20, -50] = 1
        
        # Also keep the low-frequency "gist" (the particle's shape)
        cx, cy = self.L // 2, self.L // 2
        Y, X = np.ogrid[:self.L, :self.L]
        r2 = (X - cx)**2 + (Y - cy)**2
        self.low_pass_mask = np.exp(-r2 / (2 * 4**2))
        
        self.optim_mask = fftshift(self.optim_mask) + fftshift(self.low_pass_mask)
        self.optim_mask[self.optim_mask != 0] = 1.0 # Binarize
        
        self.w_viz = np.zeros((self.L, self.L), dtype=np.float32)
        self.update_mapping_function()

    def W_Baseline(self, psi):
        return psi # Identity
        
    def W_Random(self, psi):
        psi_k = fft2(psi)
        psi_k_scrambled = psi_k * self.random_mask # Scramble in freq domain
        return ifft2(psi_k_scrambled)
        
    def W_Optimized(self, psi):
        psi_k = fft2(psi)
        # Project onto the sparse, structured basis
        psi_k_projected = psi_k * self.optim_mask
        return ifft2(psi_k_projected)
        
    def update_mapping_function(self):
        if self.mapping_type == 'Baseline (Identity)':
            self.mapping_func = self.W_Baseline
            self.w_viz = np.ones((self.L, self.L), dtype=np.float32) * 0.5
        elif self.mapping_type == 'Random (Delocalized)':
            self.mapping_func = self.W_Random
            self.w_viz = (np.angle(self.random_mask) + np.pi) / (2 * np.pi)
        elif self.mapping_type == 'Optimized (Structured)':
            self.mapping_func = self.W_Optimized
            self.w_viz = np.abs(fftshift(self.optim_mask)).astype(float)
            
        self.node_title = f"W: {self.mapping_type}"

    def get_output(self, port_name):
        if port_name == 'w_mapping':
            return self.mapping_func
        elif port_name == 'w_viz':
            return self.w_viz
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.w_viz, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.L, self.L, self.L, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Mapping Type", "mapping_type", self.mapping_type, [
                ("Baseline (Identity)", "Baseline (Identity)"), 
                ("Random (Delocalized)", "Random (Delocalized)"),
                ("Optimized (Structured)", "Optimized (Structured)"),
            ])
        ]

# ==================== UTILITY NODES ====================

class ProjectionDisplayNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120)
    
    def __init__(self, width=128, height=128, title="Display"):
        super().__init__()
        self.node_title = title
        self.inputs = {'image': 'image'}
        self.w, self.h = width, height
        self.img = np.zeros((self.h, self.w), dtype=np.float32)
        
    def step(self):
        img = self.get_blended_input('image', 'first')
        if img is not None:
            if img.shape != (self.h, self.w):
                # Use cv2.resize (which is why we added the import)
                img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            self.img = img
        else:
            self.img *= 0.95
            
    def get_display_image(self):
        img_u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

class SignalMonitorNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120)
    
    def __init__(self, history_len=500):
        super().__init__()
        self.node_title = "Signal Monitor"
        self.inputs = {'signal': 'signal'}
        self.history = deque(maxlen=history_len)
        self.history_len = history_len
        self.plot_widget = None 
        self.plot_curve = None 
        
    def step(self):
        val = self.get_blended_input('signal', 'first')
        if val is None:
            val = 0.0
        # Handle potential arrays from mean blending
        if isinstance(val, np.ndarray):
            val = val.mean()
            
        self.history.append(float(val))
            
    def get_display_image(self):
        w, h = 64, 32
        img = np.zeros((h, w), dtype=np.uint8)
        if len(self.history) > 1:
            history_array = np.array(list(self.history))
            
            # Use fixed 0-1 range for coherence
            min_val, max_val = 0.0, 1.0 
            range_val = max_val - min_val
            
            if range_val > 1e-6:
                vis_history = (history_array - min_val) / range_val
            else:
                vis_history = np.full_like(history_array, 0.5) 
            
            for i in range(min(len(vis_history) - 1, w - 1)):
                val1 = vis_history[-(i+1)]
                y1 = int((1 - val1) * (h-1)) 
                x1 = w - 1 - i
                y1 = np.clip(y1, 0, h-1)
                img[y1, x1] = 255

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def close(self):
        if self.plot_widget:
            self.plot_widget.close()
        self.plot_curve = None
        super().close()

# ==================== NODE REGISTRY ====================

NODE_TYPES = {
    'IHT Info': IHT_InfoNode,
    'Quantum Phase Field': QuantumPhaseFieldNode,
    'Holographic Mapping': HolographicMappingNode,
    'Projection Display': ProjectionDisplayNode,
    'Signal Monitor': SignalMonitorNode,
}

PORT_COLORS = {
    'signal': QtGui.QColor(200, 200, 200),
    'image': QtGui.QColor(100, 150, 255),
    'w_mapping': QtGui.QColor(100, 255, 150),
}

# ==================== GRAPHICS ITEMS & DIALOGS ====================
# (Mostly unchanged, with audio-related code removed)

PORT_RADIUS = 7
NODE_W, NODE_H = 200, 200 # Made nodes taller for info text

class PortItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent, name, port_type, is_output=False):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, PORT_RADIUS*2, PORT_RADIUS*2, parent)
        self.name = name
        self.port_type = port_type
        self.is_output = is_output
        self.base_color = PORT_COLORS.get(port_type, QtGui.QColor(255, 0, 0))
        self.setBrush(QtGui.QBrush(self.base_color))
        self.setZValue(3)
        self.setAcceptHoverEvents(True)
        
    def hoverEnterEvent(self, ev):
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 60)))
    def hoverLeaveEvent(self, ev):
        self.setBrush(QtGui.QBrush(self.base_color))

class EdgeItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, src_port, tgt_port=None):
        super().__init__()
        self.src = src_port
        self.tgt = tgt_port
        self.port_type = src_port.port_type
        self.setZValue(1)
        self.effect_val = 0.0
        pen = QtGui.QPen(PORT_COLORS.get(self.port_type, QtGui.QColor(200,200,200)))
        pen.setWidthF(2.0)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        
    def update_path(self):
        sp = self.src.scenePos()
        tp = self.tgt.scenePos() if self.tgt else sp
        path = QtGui.QPainterPath()
        path.moveTo(sp)
        dx = (tp.x() - sp.x()) * 0.5
        c1 = QtCore.QPointF(sp.x() + dx, sp.y())
        c2 = QtCore.QPointF(tp.x() - dx, tp.y())
        path.cubicTo(c1, c2, tp)
        self.setPath(path)
        self.update_style()
        
    def update_style(self):
        val = np.clip(self.effect_val, 0.0, 1.0)
        alpha = int(80 + val * 175)
        w = 2.0 + val * 4.0
        color = PORT_COLORS.get(self.port_type, QtGui.QColor(200,200,200)).lighter(130)
        color.setAlpha(alpha)
        pen = QtGui.QPen(color)
        pen.setWidthF(w)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

class NodeItem(QtWidgets.QGraphicsItem):
    def __init__(self, sim_node):
        super().__init__()
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.sim = sim_node
        self.in_ports = {}
        self.out_ports = {}
        
        y_in = 40
        for name, ptype in self.sim.inputs.items():
            port = PortItem(self, name, ptype, is_output=False)
            port.setPos(0, y_in)
            self.in_ports[name] = port
            y_in += 25
            
        y_out = 40
        for name, ptype in self.sim.outputs.items():
            port = PortItem(self, name, ptype, is_output=True)
            port.setPos(NODE_W, y_out)
            self.out_ports[name] = port
            y_out += 25
            
        self.rect = QtCore.QRectF(0, 0, NODE_W, NODE_H)
        self.setZValue(2)
        self.display_pix = None
        
    def boundingRect(self):
        return self.rect.adjusted(-8, -8, 8, 8)
        
    def paint(self, painter, option, widget):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        base = self.sim.NODE_COLOR
        if self.isSelected():
            base = base.lighter(150)
        painter.setBrush(QtGui.QBrush(base))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 2))
        painter.drawRoundedRect(self.rect, 10, 10)
        
        painter.setPen(QtGui.QColor(240, 240, 240))
        font = QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QtCore.QRectF(8, 4, NODE_W-24, 20), self.sim.node_title)
        
        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.setFont(QtGui.QFont("Arial", 7))
        painter.drawText(QtCore.QRectF(8, 18, NODE_W-16, 12), self.sim.NODE_CATEGORY)
        
        painter.setFont(QtGui.QFont("Arial", 7))
        for name, port in self.in_ports.items():
            painter.drawText(port.pos() + QtCore.QPointF(12, 4), name)
        for name, port in self.out_ports.items():
            w = painter.fontMetrics().boundingRect(name).width()
            painter.drawText(port.pos() + QtCore.QPointF(-w - 12, 4), name)
            
        if self.display_pix:
            # Adjusted display area for taller node
            img_h = NODE_H - 50 
            img_w = NODE_W - 16
            target_rect = QtCore.QRectF(8, 38, img_w, img_h)
            
            # For info node, don't keep aspect ratio
            if isinstance(self.sim, IHT_InfoNode):
                scaled = self.display_pix.scaled(
                    int(img_w), int(img_h),
                    QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                    QtCore.Qt.TransformationMode.FastTransformation
                )
                x, y = 8, 38
            else:
                scaled = self.display_pix.scaled(
                    int(img_w), int(img_h),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.FastTransformation
                )
                x = 8 + (img_w - scaled.width()) / 2
                y = 38 + (img_h - scaled.height()) / 2

            painter.drawPixmap(QtCore.QRectF(x, y, scaled.width(), scaled.height()),
                               scaled, QtCore.QRectF(scaled.rect()))
                                
    def update_display(self):
        qimg = self.sim.get_display_image()
        if qimg:
            self.display_pix = QtGui.QPixmap.fromImage(qimg)
        self.update()

class NodeConfigDialog(QtWidgets.QDialog):
    def __init__(self, node_item, parent=None):
        super().__init__(parent)
        self.node = node_item.sim
        self.node_item = node_item
        self.setWindowTitle(f"Configure: {self.node.node_title}")
        self.setFixedWidth(300)
        
        layout = QtWidgets.QVBoxLayout(self)
        self.inputs = {}

        for display_name, key, current_value, options in self.node.get_config_options():
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(QtWidgets.QLabel(display_name + ":"))

            if options:
                combo = QtWidgets.QComboBox()
                for name, value in options:
                    combo.addItem(name, userData=value)
                    if value == current_value:
                        combo.setCurrentIndex(combo.count() - 1)
                h_layout.addWidget(combo, 1)
                self.inputs[key] = combo
            else:
                line_edit = QtWidgets.QLineEdit(str(current_value))
                h_layout.addWidget(line_edit, 1)
                self.inputs[key] = line_edit
                
            layout.addLayout(h_layout)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.apply_config)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def apply_config(self):
        """Apply the new configuration to the simulation node."""
        for key, widget in self.inputs.items():
            if isinstance(widget, QtWidgets.QComboBox):
                new_value = widget.currentData()
            elif isinstance(widget, QtWidgets.QLineEdit):
                text = widget.text()
                try: new_value = int(text)
                except ValueError:
                    try: new_value = float(text)
                    except ValueError: new_value = text
            
            setattr(self.node, key, new_value)
        
        # Special action for mapping node
        if hasattr(self.node, 'update_mapping_function'):
            self.node.update_mapping_function()
            self.node_item.update_display()
            self.node_item.update()
            
        self.accept()

# ==================== MAIN SCENE ====================

class PerceptionScene(QtWidgets.QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(25, 25, 25)))
        self.nodes = []
        self.edges = []
        self.temp_edge = None
        self.connecting_src = None
        
    def add_node(self, node_class, x=0, y=0):
        sim = node_class()
        node = NodeItem(sim)
        self.addItem(node)
        node.setPos(x, y)
        self.nodes.append(node)
        node.update_display()
        return node
        
    def remove_node(self, node_item):
        if node_item in self.nodes:
            edges_to_remove = [
                e for e in self.edges 
                if e.src.parentItem() == node_item or e.tgt.parentItem() == node_item
            ]
            for edge in edges_to_remove:
                self.remove_edge(edge)
                
            node_item.sim.close()
            self.removeItem(node_item)
            self.nodes.remove(node_item)
            
    def remove_edge(self, edge):
        if edge in self.edges:
            self.removeItem(edge)
            self.edges.remove(edge)
            
    def start_connection(self, src_port):
        self.connecting_src = src_port
        self.temp_edge = EdgeItem(src_port)
        self.addItem(self.temp_edge)
        self.temp_edge.update_path()
        
    def finish_connection(self, tgt_port):
        if not self.connecting_src:
            return
        # Type check
        if (self.connecting_src.is_output and not tgt_port.is_output and
            self.connecting_src.port_type == tgt_port.port_type):
            
            # No self-connection
            if self.connecting_src.parentItem() == tgt_port.parentItem():
                self.cancel_connection()
                return

            # No duplicate edges
            edge_exists = any(
                e.src == self.connecting_src and e.tgt == tgt_port for e in self.edges
            )
            if edge_exists:
                self.cancel_connection()
                return
            
            # Allow multiple inputs, but only one source for W_mapping
            if tgt_port.port_type == 'w_mapping':
                existing = [e for e in self.edges if e.tgt == tgt_port]
                for e in existing: self.remove_edge(e)
            
            edge = EdgeItem(self.connecting_src, tgt_port)
            self.addItem(edge)
            edge.update_path()
            self.edges.append(edge)
        self.cancel_connection()
        
    def cancel_connection(self):
        if self.temp_edge:
            self.removeItem(self.temp_edge)
        self.temp_edge = None
        self.connecting_src = None
        
    def mousePressEvent(self, ev):
        item = self.itemAt(ev.scenePos(), QtGui.QTransform())
        if isinstance(item, PortItem):
            if item.is_output:
                self.start_connection(item)
                return
            elif self.connecting_src:
                self.finish_connection(item)
                return
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if self.temp_edge and self.connecting_src:
            class FakePort:
                def __init__(self, pos): self._p = pos
                def scenePos(self): return self._p
            self.temp_edge.tgt = FakePort(ev.scenePos())
            self.temp_edge.update_path()
            
    def mouseReleaseEvent(self, ev):
        item = self.itemAt(ev.scenePos(), QtGui.QTransform())
        if isinstance(item, PortItem) and not item.is_output and self.connecting_src:
            self.finish_connection(item)
            return
        if self.connecting_src:
            self.cancel_connection()
        super().mouseReleaseEvent(ev)
        
    def contextMenuEvent(self, ev):
        selected_nodes = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        
        menu = QtWidgets.QMenu()

        if selected_nodes:
            delete_action = menu.addAction(f"Delete Selected Node{'s' if len(selected_nodes) > 1 else ''} ({len(selected_nodes)})")
            delete_action.triggered.connect(lambda: self.delete_selected_nodes())
            
            if len(selected_nodes) == 1 and selected_nodes[0].sim.get_config_options():
                menu.addSeparator()
                config_action = menu.addAction("⚙ Configure Node...")
                config_action.triggered.connect(lambda: self.parent().configure_node(selected_nodes[0]))
        else:
            for name, cls in NODE_TYPES.items():
                action = menu.addAction(f"Add {name}")
                action.triggered.connect(lambda checked, c=cls, p=ev.scenePos(): 
                                         self.add_node(c, x=p.x(), y=p.y()))

        menu.exec(ev.screenPos())

    def delete_selected_nodes(self):
        selected_nodes = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        for node in selected_nodes:
            self.remove_node(node)
            
    def close_all(self):
        for node in self.nodes:
            node.sim.close()

# ==================== MAIN WINDOW ====================

class PerceptionLab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IHT-AI Laboratory") # <-- RENAMED
        self.resize(1400, 900)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        toolbar = self._create_toolbar()
        layout.addLayout(toolbar)
        
        self.scene = PerceptionScene()
        self.scene.parent = lambda: self 
        
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | 
                                QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        layout.addWidget(self.view, 1)
        
        self.status = QtWidgets.QLabel("Welcome! Press Start to run the IHT-AI simulation.")
        self.status.setStyleSheet("color: #aaa; padding: 4px;")
        layout.addWidget(self.status)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step)
        self.is_running = False
        
        self._create_starter_graph()
        
    def _create_toolbar(self):
        tb = QtWidgets.QHBoxLayout()
        
        add_btn = QtWidgets.QPushButton("➕ Add Node")
        add_menu = QtWidgets.QMenu()
        categories = {}
        for name, cls in NODE_TYPES.items():
            cat = cls.NODE_CATEGORY
            if cat not in categories: categories[cat] = []
            categories[cat].append((name, cls))
        for cat, items in sorted(categories.items()):
            cat_menu = add_menu.addMenu(cat)
            for name, cls in items:
                action = cat_menu.addAction(name)
                action.triggered.connect(lambda checked, c=cls: self.add_node(c))
        add_btn.setMenu(add_menu)
        tb.addWidget(add_btn)
        
        self.run_btn = QtWidgets.QPushButton("▶ Start")
        self.run_btn.clicked.connect(self.toggle_run)
        self.run_btn.setStyleSheet("background: #16a34a; color: white; padding: 6px 12px; font-weight: bold;")
        tb.addWidget(self.run_btn)
        
        clear_btn = QtWidgets.QPushButton("🗑 Clear Edges")
        clear_btn.clicked.connect(self.clear_edges)
        tb.addWidget(clear_btn)
        
        # --- Gamma Slider (Replaced Coupling) ---
        tb.addWidget(QtWidgets.QLabel(" |  Gamma (Dilution):"))
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(0, 200) # 0 to 0.200
        self.gamma_slider.setValue(10)     # Default 0.01
        self.gamma_slider.setMaximumWidth(200)
        tb.addWidget(self.gamma_slider)
        self.gamma_label = QtWidgets.QLabel("0.010")
        self.gamma_slider.valueChanged.connect(
            lambda v: self.gamma_label.setText(f"{v/1000.0:.3f}"))
        tb.addWidget(self.gamma_label)
        
        tb.addStretch()
        
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #666; font-family: monospace;")
        tb.addWidget(self.fps_label)
        
        return tb
        
    def _create_starter_graph(self):
        """Creates the default IHT-AI demonstration graph"""
        self.scene.nodes = []
        self.scene.edges = []
        self.scene.clear()
        
        # Create Nodes
        info = self.scene.add_node(IHT_InfoNode, x=50, y=50)
        w_select = self.scene.add_node(HolographicMappingNode, x=50, y=300)
        
        qpf = self.scene.add_node(QuantumPhaseFieldNode, x=350, y=200)
        
        disp_rho = self.scene.add_node(ProjectionDisplayNode, x=650, y=50)
        disp_rho.sim.node_title = "Display: ρ_C (Attractor)"
        
        disp_phase = self.scene.add_node(ProjectionDisplayNode, x=650, y=300)
        disp_phase.sim.node_title = "Display: Phase (Substrate)"

        mon_coherence = self.scene.add_node(SignalMonitorNode, x=650, y=550)
        mon_coherence.sim.node_title = "Monitor: Coherence"
        
        # Connect Graph
        self.connect_nodes(w_select, 'w_mapping', qpf, 'w_mapping')
        self.connect_nodes(qpf, 'rho_C', disp_rho, 'image')
        self.connect_nodes(qpf, 'phase_img', disp_phase, 'image')
        self.connect_nodes(qpf, 'coherence', mon_coherence, 'signal')
        
        for e in self.scene.edges:
            e.update_path()

        self.status.setText("IHT-AI Graph loaded. Press Start. Change Gamma or W-Mapping (Right-Click > Configure).")
        
    def connect_nodes(self, src_node_item, src_port_name, tgt_node_item, tgt_port_name):
        src_port = src_node_item.out_ports[src_port_name]
        tgt_port = tgt_node_item.in_ports[tgt_port_name]
        edge = EdgeItem(src_port, tgt_port)
        self.scene.addItem(edge)
        self.scene.edges.append(edge)

    def add_node(self, node_class):
        view_center = self.view.mapToScene(self.view.viewport().rect().center())
        node = self.scene.add_node(node_class, x=view_center.x()-100, y=view_center.y()-80)
        self.status.setText(f"Added {node.sim.node_title}")
        
    def configure_node(self, node_item):
        dialog = NodeConfigDialog(node_item, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.status.setText(f"Configured {node_item.sim.node_title}")
        
    def toggle_run(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.run_btn.setText("⏸ Stop")
            self.run_btn.setStyleSheet("background: #dc2626; color: white; padding: 6px 12px; font-weight: bold;")
            self.timer.start(33) # ~30 FPS
            self.status.setText("Running...")
            self.last_time = QtCore.QTime.currentTime()
            self.frame_count = 0
        else:
            self.run_btn.setText("▶ Start")
            self.run_btn.setStyleSheet("background: #16a34a; color: white; padding: 6px 12px; font-weight: bold;")
            self.timer.stop()
            self.status.setText("Stopped")
            
    def clear_edges(self):
        for edge in list(self.scene.edges):
            self.scene.remove_edge(edge)
        self.status.setText("Cleared all edges")
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Delete or event.key() == QtCore.Qt.Key.Key_Backspace:
            self.scene.delete_selected_nodes()
            self.status.setText("Deleted selected nodes.")
            return
        super().keyPressEvent(event)
        
    def step(self):
        
        # Get global Gamma value from slider
        gamma_value = self.gamma_slider.value() / 1000.0
        
        for node in self.scene.nodes:
            # Inject global gamma into the QPF node(s)
            if isinstance(node.sim, QuantumPhaseFieldNode):
                node.sim.set_gamma(gamma_value)
            node.sim.pre_step()
            
        node_map = {n: n for n in self.scene.nodes}
        
        for edge in self.scene.edges:
            src_node = edge.src.parentItem()
            tgt_node = edge.tgt.parentItem()
            
            if src_node not in node_map or tgt_node not in node_map:
                continue
                
            output = src_node.sim.get_output(edge.src.name)
            if output is None:
                continue
                
            # Use coupling=1.0 for all IHT connections
            tgt_node.sim.set_input(edge.tgt.name, output, edge.src.port_type, 1.0) 
            
            # Simple edge effect
            if edge.src.port_type == 'signal':
                if isinstance(output, (float, int)):
                    edge.effect_val = abs(float(output))
                else: edge.effect_val = 0.5
            else:
                edge.effect_val = 0.8
            edge.update_path()
            
        for node in self.scene.nodes:
            node.sim.step()
            node.update_display()
            
        # FPS Counter
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = QtCore.QTime.currentTime()
            elapsed = self.last_time.msecsTo(current_time) / 1000.0
            if elapsed > 0:
                fps = 30.0 / elapsed
                self.fps_label.setText(f"FPS: {fps:.1f}")
            self.last_time = current_time
            
    def closeEvent(self, event):
        self.timer.stop()
        self.scene.close_all()
        super().closeEvent(event)

# ==================== APPLICATION ENTRY ====================

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 80, 80))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(100, 150, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(100, 150, 255))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(20, 20, 20))
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QWidget { font-family: 'Segoe UI', Arial, sans-serif; }
        QPushButton {
            border: none; border-radius: 4px; padding: 6px 12px;
            background: #3a3a3a; color: #ddd;
        }
        QPushButton:hover { background: #4a4a4a; }
        QPushButton:pressed { background: #2a2a2a; }
        QPushButton::menu-indicator { width: 0px; }
        QMenu { background: #2a2a2a; border: 1px solid #444; }
        QMenu::item { padding: 6px 20px; }
        QMenu::item:selected { background: #3a5a8a; }
        QSlider::groove:horizontal {
            height: 4px; background: #3a3a3a; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #6495ed; width: 14px; margin: -5px 0; border-radius: 7px;
        }
        QSlider::handle:horizontal:hover { background: #7ab5ff; }
        QLineEdit, QComboBox {
            background: #3a3a3a; border: 1px solid #555; padding: 2px;
            color: #ddd; border-radius: 4px; height: 24px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding; subcontrol-position: top right;
            width: 15px; border-left-width: 1px; border-left-color: #555;
            border-left-style: solid;
        }
    """)
    
    window = PerceptionLab()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()