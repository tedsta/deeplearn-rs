use std::cell::RefCell;
use std::rc::Rc;

use matrix::{self, ClMatrix, ClMatrixMode};

use super::operation::Operation;

pub struct Node {
    op: Rc<RefCell<Box<Operation>>>,
    pub inputs: Vec<VarIndex>,
    pub scratch: Vec<VarIndex>, // Scratch variables
    pub outputs: Vec<VarIndex>,
    pub gradients: Vec<VarIndex>, // gradients on inputs
    pub out_events: Vec<matrix::cl_matrix::Event>,
}

pub struct Graph {
    nodes: Vec<Node>,
    vars: Vec<ClMatrix<f32>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
            vars: vec![],
        }
    }

    pub fn add_node(&mut self,
                    ctx: &matrix::Context,
                    op: Rc<RefCell<Box<Operation>>>,
                    inputs: Vec<VarIndex>,
                    out_shapes: &[(u64, u64)])
                    -> NodeIndex {
        let mut outputs = vec![];
        for &(rows, cols) in out_shapes {
            let var_index = VarIndex(self.vars.len());
            self.vars.push(ClMatrix::new(ctx, rows as usize, cols as usize, ClMatrixMode::Mut));
            outputs.push(var_index);
        }
        let mut gradients = vec![];
        for input in &inputs {
            let var_index = VarIndex(self.vars.len());
            let (rows, cols)= (input.get(self).rows(), input.get(self).columns());
            self.vars.push(ClMatrix::new(ctx, rows as usize, cols as usize, ClMatrixMode::Mut));
            gradients.push(var_index);
        }
        self.nodes.push(Node { op: op,
                               inputs: inputs,
                               scratch: vec![],
                               outputs: outputs,
                               gradients: gradients,
                               out_events: vec![] });
        NodeIndex(self.nodes.len()-1)
    }

    pub fn add_variable(&mut self, ctx: &matrix::Context, shape: (u64, u64)) -> VarIndex {
        self.vars.push(ClMatrix::new(ctx, shape.0 as usize, shape.1 as usize, ClMatrixMode::Mut));
        VarIndex(self.vars.len()-1)
    }

    pub fn run(&mut self, ctx: &matrix::Context) {
        // Clear all the out_events
        for node in &mut self.nodes {
            node.out_events.clear();
        }

        let op = self.nodes[0].op.clone();
        op.borrow_mut().forward(ctx, self, NodeIndex(0));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
pub struct VarIndex(usize);

impl VarIndex {
    pub fn get<'a>(&self, g: &'a Graph) -> &'a ClMatrix<f32> {
        &g.vars[self.0]
    }

    pub fn get_mut<'a>(&self, g: &'a mut Graph) -> &'a mut ClMatrix<f32> {
        &mut g.vars[self.0]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
pub struct NodeIndex(usize);

impl NodeIndex {
    pub fn get<'a>(&self, g: &'a Graph) -> &'a Node {
        &g.nodes[self.0]
    }

    pub fn get_mut<'a>(&self, g: &'a mut Graph) -> &'a mut Node {
        &mut g.nodes[self.0]
    }
}

#[test]
fn it_works() {
    use super::operation::MatMul;
    
    let ctx = matrix::Context::new();

    // Setup the graph
    let mut graph = Graph::new();
    let a = graph.add_variable(&ctx, (1, 2));
    let wa = graph.add_variable(&ctx, (2, 3));
    let node = graph.add_node(&ctx,
                              Rc::new(RefCell::new(Box::new(MatMul::new(&ctx, (1, 2), (2, 3))))),
                              vec![a, wa],
                              &[(1, 3)]);

    // Send some input data
    let a_cpu = matrix::Matrix::from_vec(1, 2, vec![1.0, 1.0]);
    let wa_cpu = matrix::Matrix::from_vec(2, 3, vec![0.5, 0.3, 0.2,
                                                     0.6, 0.7, 0.7]);
    a.get(&graph).set(&ctx, &a_cpu);
    wa.get(&graph).set(&ctx, &wa_cpu);

    // Run the network
    graph.run(&ctx);
    let out = {
        let ref out_event = node.get(&graph).out_events[0];
        out_event.get(&ctx, node.get(&graph).outputs[0].get(&graph))
    };
    println!("{:?}", out);
    assert!(false);
}
