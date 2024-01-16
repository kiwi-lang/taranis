from collections import defaultdict
from dataclasses import dataclass


import networkx as nx
import torch
import torch.nn as nn


@dataclass
class Memory:
    module: nn.Module = None
    x_in: torch.Tensor = None
    x_out: torch.Tensor = None
    grad_in: torch.Tensor = None
    grad_out: torch.Tensor = None
    node_id: int = None


class DiagnosticView:
    #
    #   Broadcasting rules
    #
    #   Image: (C x W x H) => None
    #   Conv2d             => None
    #   MaxPool            => None
    #   Linear             => Element wise multiplication + b
    #   Flatten            => Keep size
    #
    def __init__(self) -> None:
        self.order = []
        self.fph = None
        self.fh = None
        self.bph = None
        self.bh = None
        self.depth = 0
        self.shape_tracker = []
        self.memory = defaultdict(Memory)
        self.nodes = dict()
        self.graphs = [nx.Graph()]
        self.prev = None

        input_id ="0. Input"
        mem = self.memory["input"]
        mem.node_id = input_id
        self.graph.add_node(input_id)
        self.prev = input_id

    def find_node(self, name) -> Memory:
        for k,v in self.memory.items():
            if name == v.node_id:
                return v
            
        return None

    @property
    def graph(self):
        return self.graphs[-1]

    def register(self, module):
        module.apply(self.module_apply)

        self.fph = nn.modules.module.register_module_forward_pre_hook(self.forward_pre_hook)
        self.fh = nn.modules.module.register_module_forward_hook(self.forward_hook)
        self.bph = nn.modules.module.register_module_full_backward_pre_hook(self.backward_pre_hook)
        self.bh = nn.modules.module.register_module_full_backward_hook(self.backward_hook)

    def unregister(self):
        if self.fph is not None:
            self.fph.remove()
            self.fh.remove()
            self.bph.remove()
            self.bh.remove()

        self.fph = None
        self.fh = None
        self.bph = None
        self.bh = None

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.unregister()

    def __del__(self):
        self.unregister()

    @staticmethod 
    def is_leaf(module: nn.Module):
        return len(list(module.children())) == 0

    @staticmethod
    def show_grad_shape(grad):
        if isinstance(grad, tuple):
            return [i.shape for i in grad if i is not None]
        return grad.shape

    @staticmethod
    def show_module(module: nn.Module):
        params = list(module.parameters(recurse=False))
        shapes = [tuple(param.shape) for param in params]
        return f"{module.__class__.__name__:>20} {str(shapes):>30}"

    @staticmethod
    def show_step(step):
        return f"{step:>20}"
    
    def show_depth(self):
        return " " * self.depth + "|"
    
    def log(self, *args, **kwargs):
        print(*args, **kwargs)

    def get_node(self, module):
        node_id = len(self.memory)

        mem = self.memory[module]
        mem.module = module

        if mem.node_id is None:
            # if not self.is_leaf(module):
            #     g = nx.Graph()
            #     mem.node_id = g
            # else:
                node_id = f"{node_id}. {str(module.__class__.__name__)}"
                self.graph.add_node(node_id)
                mem.node_id = node_id
        
        return mem
        
    def module_apply(self, module: nn.Module):
        self.log(self.show_step('apply'), self.show_module(module), self.is_leaf(module))
        # _ = self.get_node(module)

    def forward_pre_hook(self, module, x_in, *args):
        self.log(self.show_depth(), self.show_step('forward_pre_hook'), self.show_module(module), self.show_grad_shape(x_in))
        self.depth += 1

        mem =  self.get_node(module)
        mem.x_in = x_in
        
        # entering a parent
        # if not self.is_leaf(module):
        #    self.graphs.append(mem.node_id)

        if self.prev is not None: #and self.is_leaf(module):
            self.graph.add_edge(self.prev, mem.node_id)
        self.prev = mem.node_id

    def forward_hook(self, module, x_in, x_out, *args):
        self.depth -= 1
        self.log(self.show_depth(), self.show_step('forward_hook'), self.show_module(module), self.show_grad_shape(x_in), self.show_grad_shape(x_out))
        if self.depth == 0:
            print()
        
        mem = self.get_node(module)
        mem.x_in = x_in
        mem.x_out = x_out

        # if self.prev is not None: #and self.is_leaf(module):
        #    self.graph.add_edge(self.prev, mem.node_id)
    
        # leaving a parent
        # if not self.is_leaf(module):
        #    self.graphs.pop()

        # self.prev = mem.node_id

    def backward_pre_hook(self, module, grad_out):
        """"""
        # weight size
        # self.log(self.show_depth(), self.show_step('backward_pre_hook'), self.show_module(module), self.show_grad_shape(grad_out))
        mem =  self.get_node(module)
        mem.grad_out = grad_out

    def backward_hook(self, module, grad_in, grad_out):
        self.log(self.show_depth(), self.show_step('backward_hook'), self.show_module(module), self.show_grad_shape(grad_in), self.show_grad_shape(grad_out))
        mem = self.get_node(module)
        mem.grad_in = grad_in
        mem.grad_out = grad_out



def test():
    from taranis.core.diagnostic.example import Net2

    view = DiagnosticView()

    net = Net2()
    view.register(net)
    print()


    bs = 1
    input = torch.randn(bs, 1, 32, 32)
    target = torch.randn((bs, 10))    # a dummy target, for example

    out = net(input)

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    loss.backward()