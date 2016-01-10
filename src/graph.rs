use std::rc::Rc;
use std::sync::mpsc::channel;

use matrix::{self, ClMatrix, ClMatrixMode};

// A function that takes an opencl context, a list of inputs, and a list of outputs
pub type Operation = Fn(&matrix::Context, &[&ClMatrix<f32>], &[&ClMatrix<f32>]);

pub struct Node {
    op: Rc<Operation>,
    op_d: Vec<Rc<Operation>>, // Derivative ops, with respect to each input, respectively
    pub inputs: Vec<VarIndex>,
    pub outputs: Vec<VarIndex>,
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
                    op: Rc<Operation>,
                    op_d: Vec<Rc<Operation>>,
                    out_shapes: &[(u64, u64)])
                    -> NodeIndex {
        let mut outputs = vec![];
        for &(rows, cols) in out_shapes {
            let var_index = VarIndex(self.vars.len());
            self.vars.push(ClMatrix::new(ctx, rows as usize, cols as usize, ClMatrixMode::Mut));
            outputs.push(var_index);
        }
        self.nodes.push(Node { op: op, op_d: op_d, inputs: vec![], outputs: outputs });
        NodeIndex(self.nodes.len()-1)
    }

    pub fn add_variable(&mut self, ctx: &matrix::Context, shape: (u64, u64)) -> VarIndex {
        self.vars.push(ClMatrix::new(ctx, shape.0 as usize, shape.1 as usize, ClMatrixMode::Mut));
        VarIndex(self.vars.len()-1)
    }

    pub fn set_node_inputs(&mut self, n: NodeIndex, inputs: &[VarIndex]) {
        n.get_mut(self).inputs.extend_from_slice(inputs);
    }

    pub fn run(&mut self) {
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
    let mat_mul =
        Rc::new(move |ctx: &matrix::Context, i: &[&ClMatrix<f32>], o: &[&ClMatrix<f32>]| {
            i[0].multiply(ctx, i[1], o[0]);
        });
    // Derivative with respect to first input
    let mat_mul_d0 =
        Rc::new(move |ctx: &matrix::Context, i: &[&ClMatrix<f32>], o: &[&ClMatrix<f32>]| {
            i[1].copy_to(ctx, o[0]);
        });
    // Derivative with respect to second input
    let mat_mul_d1 =
        Rc::new(move |ctx: &matrix::Context, i: &[&ClMatrix<f32>], o: &[&ClMatrix<f32>]| {
            i[0].copy_to(ctx, o[0]);
        });
    let relu =
        Rc::new(move |ctx: &matrix::Context, i: &[&ClMatrix<f32>], o: &[&ClMatrix<f32>]| {
            i[0].max(ctx, 0.0, o[0]);
        });
    let relu_d =
        Rc::new(move |ctx: &matrix::Context, i: &[&ClMatrix<f32>], o: &[&ClMatrix<f32>]| {
            i[0].dmax(ctx, 0.0, o[0]);
        });

    // Special placeholder op
    /*let placeholder =
        Rc::new(move |ctx: &matrix::Context, i, o| {
            i[0].dmax(ctx, 0.0, o[0]);
        });*/
    
    let mut ctx = matrix::Context::new();

    let mut graph = Graph::new();
    let a = graph.add_variable(&ctx, (1, 2));
    let wa = graph.add_variable(&ctx, (2, 3));
    let node = graph.add_node(&ctx, mat_mul, vec![mat_mul_d0, mat_mul_d1], &[(1, 3)]);
    graph.set_node_inputs(node, &[a, wa]);
    let out = node.get(&graph).outputs[0];
}
