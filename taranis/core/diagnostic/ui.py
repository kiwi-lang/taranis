# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: BSD-3-Clause


import math
import sys
import traceback

from PySide6.QtCore import (QEasingCurve, QLineF,
                            QParallelAnimationGroup, QPointF,
                            QPropertyAnimation, QRectF, Qt)
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF, QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QComboBox, QGraphicsItem,
                               QGraphicsObject, QGraphicsScene, QGraphicsView,
                               QStyleOptionGraphicsItem, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QSpinBox, QGridLayout)

import networkx as nx
from taranis.core.image import to_np_image

class Node(QGraphicsObject):

    """A QGraphicsItem representing node in a graph"""

    def __init__(self, name: str, diagnostic, parent=None):
        """Node constructor

        Args:
            name (str): Node label
        """
        super().__init__(parent)
        self._name = name
        self._mem = diagnostic.find_node(name)
        self._edges = []
        self._color = "#5AD469"
        self._selected = False
        self._radius = 30
        self._rect = QRectF(0, 0, self._radius * 2, self._radius * 2)

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.selection_callback = None

    def get_data(self):
        return self._mem

    def select(self):
        self._color = "#2fb33f"
        self._selected = True
        self.update()

    def unselect(self):
        self._color = "#5AD469"
        self._selected = False
        self.update()

    def boundingRect(self) -> QRectF:
        """Override from QGraphicsItem

        Returns:
            QRect: Return node bounding rect
        """
        return self._rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        """Override from QGraphicsItem

        Draw node

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(
            QPen(
                QColor(self._color).darker(),
                2,
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.setBrush(QBrush(QColor(self._color)))
        painter.drawEllipse(self.boundingRect())
        painter.setPen(QPen(QColor("white")))
        painter.drawText(self.boundingRect(), Qt.AlignCenter, self._name)

    def add_edge(self, edge):
        """Add an edge to this node

        Args:
            edge (Edge)
        """
        self._edges.append(edge)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        """Override from QGraphicsItem

        Args:
            change (QGraphicsItem.GraphicsItemChange)
            value (Any)

        Returns:
            Any
        """
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self._edges:
                edge.adjust()

        return super().itemChange(change, value)
    
    def mouseDoubleClickEvent(self, *args):
        if self.selection_callback is not None:
            self.selection_callback(self, *args)


class Edge(QGraphicsItem):
    def __init__(self, source: Node, dest: Node, diagnostic, parent: QGraphicsItem = None):
        """Edge constructor

        Args:
            source (Node): source node
            dest (Node): destination node
        """
        super().__init__(parent)
        self._source = source
        self._dest = dest
        self._size = None

        mem = diagnostic.find_node(dest._name)
        self._size = tuple(mem.x_in[0].shape[1:])


        self._tickness = 2
        self._color = "#2BB53C"
        self._arrow_size = 20

        self._source.add_edge(self)
        self._dest.add_edge(self)

        self._line = QLineF()
        self.setZValue(-1)
        self.adjust()

    def boundingRect(self) -> QRectF:
        """Override from QGraphicsItem

        Returns:
            QRect: Return node bounding rect
        """
        return (
            QRectF(self._line.p1(), self._line.p2())
            .normalized()
            .adjusted(
                -self._tickness - self._arrow_size,
                -self._tickness - self._arrow_size,
                self._tickness + self._arrow_size,
                self._tickness + self._arrow_size,
            )
        )

    def adjust(self):
        """
        Update edge position from source and destination node.
        This method is called from Node::itemChange
        """
        self.prepareGeometryChange()
        self._line = QLineF(
            self._source.pos() + self._source.boundingRect().center(),
            self._dest.pos() + self._dest.boundingRect().center(),
        )

    def _draw_arrow(self, painter: QPainter, start: QPointF, end: QPointF):
        """Draw arrow from start point to end point.

        Args:
            painter (QPainter)
            start (QPointF): start position
            end (QPointF): end position
        """
        painter.setBrush(QBrush(self._color))

        line = QLineF(end, start)

        angle = math.atan2(-line.dy(), line.dx())
        arrow_p1 = line.p1() + QPointF(
            math.sin(angle + math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi / 3) * self._arrow_size,
        )
        arrow_p2 = line.p1() + QPointF(
            math.sin(angle + math.pi - math.pi / 3) * self._arrow_size,
            math.cos(angle + math.pi - math.pi / 3) * self._arrow_size,
        )

        arrow_head = QPolygonF()
        arrow_head.clear()
        arrow_head.append(line.p1())
        arrow_head.append(arrow_p1)
        arrow_head.append(arrow_p2)
        painter.drawLine(line)
        painter.drawPolygon(arrow_head)

        painter.setPen(
                QPen(
                    QColor("#000000"),
                    self._tickness,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
        painter.drawText(self.boundingRect(), Qt.AlignCenter, str(self._size))

    def _arrow_target(self) -> QPointF:
        """Calculate the position of the arrow taking into account the size of the destination node

        Returns:
            QPointF
        """
        target = self._line.p1()
        center = self._line.p2()
        radius = self._dest._radius
        vector = target - center
        length = math.sqrt(vector.x() ** 2 + vector.y() ** 2)
        if length == 0:
            return target
        normal = vector / length
        target = QPointF(center.x() + (normal.x() * radius), center.y() + (normal.y() * radius))

        return target

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        """Override from QGraphicsItem

        Draw Edge. This method is called from Edge.adjust()

        Args:
            painter (QPainter)
            option (QStyleOptionGraphicsItem)
        """

        if self._source and self._dest:
            painter.setRenderHints(QPainter.Antialiasing)

            painter.setPen(
                QPen(
                    QColor(self._color),
                    self._tickness,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            painter.drawLine(self._line)
            self._draw_arrow(painter, self._line.p1(), self._arrow_target())
            self._arrow_target()




def snake_layout_factory(colcount=4, x_spacing=1, y_spacing=1):
    def snake_layout(G: nx.Graph, scale=1, center=None, dim=2):
        
        n = len(G)
        pos = []

        for i in range(n):
            row = i // colcount
            col = i % colcount

            posx = col * x_spacing
            if row % 2 == 1:
                posx = (colcount - 1) * x_spacing - posx

            pos.append([posx, row * y_spacing])

        return dict(zip(G, pos))

    return snake_layout



class GraphView(QGraphicsView):
    def __init__(self, diagnostic, parent=None):
        """GraphView constructor

        This widget can display a directed graph

        Args:
            graph (nx.DiGraph): a networkx directed graph
        """
        super().__init__()
        self.diagnostic = diagnostic
        self.selection_callback = None

        self._graph = diagnostic.graph
        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        # Used to add space between nodes
        self._graph_scale = 150
        
        # Map node name to Node object {str=>Node}
        self._nodes_map = {}

        # List of networkx layout function
        self._nx_layout = {
            "snake": snake_layout_factory(4, 1, 0.75),
            "circular": nx.circular_layout,
            "planar": nx.planar_layout,
            "random": nx.random_layout,
            "shell_layout": nx.shell_layout,
            "kamada_kawai_layout": nx.kamada_kawai_layout,
            "spring_layout": nx.spring_layout,
            "spiral_layout": nx.spiral_layout,
        }

        self._selection = None
        self._load_graph()
        self.set_nx_layout("snake")

    def get_nx_layouts(self) -> list:
        """Return all layout names

        Returns:
            list: layout name (str)
        """
        return self._nx_layout.keys()

    def set_nx_layout(self, name: str):
        """Set networkx layout and start animation

        Args:
            name (str): Layout name
        """
        if name in self._nx_layout:
            self._nx_layout_function = self._nx_layout[name]

            # Compute node position from layout function
            positions = self._nx_layout_function(self._graph)

            # Change position of all nodes using an animation
            self.animations = QParallelAnimationGroup()
            for node, pos in positions.items():
                x, y = pos
                x *= self._graph_scale
                y *= self._graph_scale
                item = self._nodes_map[node]

                animation = QPropertyAnimation(item, b"pos")
                animation.setDuration(1000)
                animation.setEndValue(QPointF(x, y))
                animation.setEasingCurve(QEasingCurve.OutExpo)
                self.animations.addAnimation(animation)

            self.animations.start()

    def select_node(self, item, *args):
        if self._selection is not None:
            self._selection.unselect()

        self._selection = item
        item.select()

        if self.selection_callback is not None:
            self.selection_callback(item, *args)

    def _load_graph(self):
        """Load graph into QGraphicsScene using Node class and Edge class"""

        self.scene().clear()
        self._nodes_map.clear()
        manager = self.diagnostic

        # Add nodes
        for node in self._graph:
            item = Node(node, manager)
            item.selection_callback = self.select_node
            self.scene().addItem(item)
            self._nodes_map[node] = item

        # Add edges
        for a, b in self._graph.edges:
            source = self._nodes_map[a]
            dest = self._nodes_map[b]
            self.scene().addItem(Edge(source, dest, manager))


class ImageView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent) 

        v_layout = QVBoxLayout(self)
        h_layout = QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.batch_index = self.new_input(h_layout, "batch")
        # self.channel_index = self.new_input(h_layout, "channel")

        self.x_in, self.x_in_l = self.new_image(v_layout, "Input")
        self.x_out, self.x_out_l = self.new_image(v_layout, "Output")
        v_layout.addStretch()
        self.setLayout(v_layout)

        self.col = 4
        self.item = None

    def new_input(self, layout, name, default=0):
        h_layout = QHBoxLayout()
        label = QLabel(name)
        input = QSpinBox()
        input.setValue(default)
        input.valueChanged.connect(self.update_image)
        h_layout.addWidget(label)
        h_layout.addWidget(input)
        layout.addLayout(h_layout)
        return input

    def new_image(self, layout, name):
        images_holder = QGridLayout()
        l = QLabel(name)
        
        v_layout = QVBoxLayout()
        v_layout.addWidget(l)
        v_layout.addLayout(images_holder)

        layout.addLayout(v_layout)
        return images_holder, l

    def to_pixmap(self, tensor):
        data =  to_np_image(tensor)
        w, h = data.shape
        img = QImage(data.data, w, h, 1 * w, QImage.Format_Indexed8)
        pix = QPixmap.fromImage(img).scaledToHeight(100, Qt.TransformationMode.FastTransformation)
        return pix

    def preview(self, item, *args):
        self.item = item
        self.update_image()

    def update_image(self):
        mem = self.item.get_data()

        b = self.batch_index.value()
        # c = self.channel_index.value()

        def clear(layout: QHBoxLayout):
            while layout.count() > 0:
                item = layout.takeAt(0)
                if widget := item.widget():
                    # Parent is none it will get GC by Qt
                    widget.setParent(None)

        def show_channels(layout, x, channels=None, max=None):
            if channels is None:
                channels = range(x.shape[1])
            
            if max is not None:
                channels = list(channels)[:max]
            
            for c in channels:
                img = QLabel()
                img.setPixmap(self.to_pixmap(x[b, c]))
                layout.addWidget(img, c // self.col, c % self.col)

        clear(self.x_out)
        clear(self.x_in)

        print(mem.grad_in[0].shape)
        print(mem.grad_out[0].shape)
    
        try:
            x_in = mem.x_in[0]
            self.x_in_l.setText(f"Input {str(tuple(x_in[b].shape))}")
            show_channels(self.x_in, x_in)
        except Exception:
            traceback.print_exc()

        try:
            x_out = mem.x_out
            self.x_out_l.setText(f"Output {str(tuple(x_out[b].shape))}")
            show_channels(self.x_out, x_out)
                
        except Exception:
            traceback.print_exc()


class MainWindow(QWidget):
    def __init__(self, graph, parent=None):
        super().__init__()

        # self.graph = nx.DiGraph()
        # self.graph.add_edges_from(
        #     [
        #         ("1", "2"),
        #         ("2", "3"),
        #         ("3", "4"),
        #         ("1", "5"),
        #         ("1", "6"),
        #         ("1", "7"),
        #     ]
        # )

        self.view = GraphView(graph)
        

        self.choice_combo = QComboBox()
        self.choice_combo.addItems(self.view.get_nx_layouts())
        self.image_view = ImageView()

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.choice_combo)
        v_layout.addWidget(self.view)

        self.choice_combo.currentTextChanged.connect(self.view.set_nx_layout)

        h_layout = QHBoxLayout()
        h_layout.addLayout(v_layout)
        h_layout.addWidget(self.image_view)
        self.setLayout(h_layout)


        self.view.selection_callback = self.image_view.preview


def main():
    from taranis.core.diagnostic.example import Net2
    from taranis.core.diagnostic.diagnostic import DiagnosticView

    import torch
    import torch.nn as nn

    view = DiagnosticView()

    net = Net2()
    view.register(net)
    print()

    bs = 3
    input = torch.randn(bs, 1, 32, 32)
    target = torch.randn((bs, 10))    # a dummy target, for example

    out = net(input)

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    loss.backward()

    # ---

    app = QApplication(sys.argv)

    # Create a networkx graph

    widget = MainWindow(view)
    widget.show()
    widget.resize(1000, 800)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
