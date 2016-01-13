use matrix::{self, ClMatrix, ClMatrixMode};

use super::operation::Operation;
use super::var_store::{VarIndex, VarStore};

pub struct Node {
    pub inputs: Vec<VarIndex>,
    pub outputs: Vec<VarIndex>,
    pub gradients: Vec<VarIndex>, // gradients on inputs
    pub out_events: Vec<matrix::cl_matrix::Event>,
}

pub struct Graph {
    nodes: Vec<Node>,
    node_ops: Vec<Box<Operation>>,
    pub var_store: VarStore,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
            node_ops: vec![],
            var_store: VarStore::new(),
        }
    }

    pub fn add_node(&mut self,
                    ctx: &matrix::Context,
                    op: Box<Operation>,
                    inputs: Vec<VarIndex>,
                    out_shapes: &[(u64, u64)])
                    -> NodeIndex {
        // Create output variables
        let mut outputs = vec![];
        for &(rows, cols) in out_shapes {
            let var_index = self.var_store.add(ClMatrix::new(ctx, rows as usize, cols as usize, ClMatrixMode::Mut));
            outputs.push(var_index);
        }
        // Create gradient variables
        let mut gradients = vec![];
        for input in &inputs {
            let (rows, cols) = (input.get(self).rows(), input.get(self).columns());
            let var_index = self.var_store.add(ClMatrix::new(ctx, rows as usize, cols as usize, ClMatrixMode::Mut));
            gradients.push(var_index);
        }
        // Create the node
        self.nodes.push(Node { inputs: inputs,
                               outputs: outputs,
                               gradients: gradients,
                               out_events: vec![] });
        // Add the corresponding node op
        self.node_ops.push(op);
        NodeIndex(self.nodes.len()-1)
    }

    pub fn add_variable(&mut self, ctx: &matrix::Context, shape: (u64, u64)) -> VarIndex {
        self.var_store.add(ClMatrix::new(ctx, shape.0 as usize, shape.1 as usize, ClMatrixMode::Mut))
    }

    pub fn run(&mut self, ctx: &matrix::Context) {
        // Clear all the out_events
        for node in &mut self.nodes {
            node.out_events.clear();
        }

        self.node_ops[0].forward(ctx, &mut self.var_store, &mut self.nodes[0]);
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
                              Box::new(MatMul::new(&ctx, (1, 2), (2, 3))),
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
